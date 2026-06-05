/**
 * @file DVCSProcessBMJ12Torch.cpp
 *
 * Implements the PARTONS-registered differentiable DVCS process module.
 * The new tensor entry point pulls CFFs from a DVCSCFFNNPytorch via its
 * computeCFFTensor(gpdType) method (gradients preserved), dresses them
 * with the BMJ12 kinematic helicity coefficients, and evaluates the total
 * σ(λ,φ) using the Theory::* torch implementations.
 */

#include "NNFit/theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterModule.h>

#include "NNFit/theory/Modules/CFFs/DVCS/DVCSCFFNNPytorch.h"
#include "NNFit/theory/Modules/Processes/DVCS/DVCSAmplitudesBMJ12Torch.h"

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const unsigned int DVCSProcessBMJ12Torch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSProcessBMJ12Torch("DVCSProcessBMJ12Torch"));

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

DVCSProcessBMJ12Torch::DVCSProcessBMJ12Torch(const std::string& className)
        : DVCSProcessBMJ12(className),
          m_torchKin(),
          m_torchKinInitialized(false),
          m_lastXB(0.0), m_lastT(0.0), m_lastQ2(0.0), m_lastE(0.0) {
}

DVCSProcessBMJ12Torch::DVCSProcessBMJ12Torch(const DVCSProcessBMJ12Torch& other)
        : DVCSProcessBMJ12(other),
          m_torchKin(other.m_torchKin),
          m_torchKinInitialized(other.m_torchKinInitialized),
          m_lastXB(other.m_lastXB), m_lastT(other.m_lastT),
          m_lastQ2(other.m_lastQ2), m_lastE(other.m_lastE) {
}

DVCSProcessBMJ12Torch::~DVCSProcessBMJ12Torch() {
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

DVCSProcessBMJ12Torch* DVCSProcessBMJ12Torch::clone() const {
    return new DVCSProcessBMJ12Torch(*this);
}

// ---------------------------------------------------------------------------
// PARTONS boilerplate (delegates to the BMJ12 base class)
// ---------------------------------------------------------------------------

void DVCSProcessBMJ12Torch::configure(const ElemUtils::Parameters& parameters) {
    DVCSProcessBMJ12::configure(parameters);
}

void DVCSProcessBMJ12Torch::resolveObjectDependencies() {
    DVCSProcessBMJ12::resolveObjectDependencies();
}

void DVCSProcessBMJ12Torch::prepareSubModules(
        const std::map<std::string, PARTONS::BaseObjectData>& subModulesData) {
    DVCSProcessBMJ12::prepareSubModules(subModulesData);
}

void DVCSProcessBMJ12Torch::initModule() {
    DVCSProcessBMJ12::initModule();
    // Force a rebuild of the torch-side kinematics on the next tensor call:
    // PARTONS may have updated (xB, t, Q2, E) on the parent.
    m_torchKinInitialized = false;
}

void DVCSProcessBMJ12Torch::isModuleWellConfigured() {
    DVCSProcessBMJ12::isModuleWellConfigured();
}

// ---------------------------------------------------------------------------
// buildTorchKinematics
// ---------------------------------------------------------------------------

void DVCSProcessBMJ12Torch::buildTorchKinematics() {

    if (m_torchKinInitialized
            && m_lastXB == m_xB && m_lastT == m_t
            && m_lastQ2 == m_Q2 && m_lastE == m_E)
        return;

    m_torchKin = Theory::computeKinematics(m_xB, m_t, m_Q2, m_E);

    m_lastXB = m_xB;
    m_lastT  = m_t;
    m_lastQ2 = m_Q2;
    m_lastE  = m_E;
    m_torchKinInitialized = true;
}

// ---------------------------------------------------------------------------
// setupKinematics — replacement for pProc->compute(...) when only the
// tensor path will run. setKinematics() is protected on DVCSProcessModule
// but accessible from this derived class.
// ---------------------------------------------------------------------------

void DVCSProcessBMJ12Torch::setupKinematics(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    setKinematics(kinematic);   // inherited; sets m_xB, m_t, m_Q2, m_E, m_phi

    // Push the same kinematic point onto the CFF module. PARTONS' scalar
    // pipeline does this inside computeConvolCoeffFunction() as a side
    // effect of dispatching computeCFF(); we replicate just the state
    // update (no NN forward) so that subsequent computeCFFTensor() calls
    // see correct (xi, t, Q²) instead of construction-default zeros.
    DVCSCFFNNPytorch* pCFF =
            dynamic_cast<DVCSCFFNNPytorch*>(m_pConvolCoeffFunctionModule);

    if (pCFF == nullptr)
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Attached CFF module is not a DVCSCFFNNPytorch.");

    // xB → xi via the configured xi converter (typically DVCSXiConverterXBToXi:
    // xi = xB / (2 − xB)). Going through the module rather than inlining
    // the conversion keeps this code working if the xi converter is ever
    // swapped for a different parametrisation.
    const double xi = m_pXiConverterModule->compute(kinematic).getValue();

    pCFF->setupKinematics(xi, m_t, m_Q2);

    buildTorchKinematics();     // refresh Theory::DVCSKin cache
}

// ---------------------------------------------------------------------------
// computeFourierCoeffsTensor — single NN forward + DressedCFFs +
// Fourier-coefficient build. Designed to be called once per kinematic
// point; the result is reused across all φ and helicity values in the
// observable's quadrature loop, eliminating the (4 × 20)× redundancy
// of computing the same CFFs and coefficients per cross-section call.
// ---------------------------------------------------------------------------

std::pair<Theory::VCSCoeffs, Theory::InterfCoeffs>
DVCSProcessBMJ12Torch::computeFourierCoeffsTensor() {

    // Ensure the φ-independent torch-side kinematics are up to date.
    buildTorchKinematics();

    DVCSCFFNNPytorch* pCFF =
            dynamic_cast<DVCSCFFNNPytorch*>(m_pConvolCoeffFunctionModule);

    if (pCFF == nullptr)
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                    << "Attached CFF module is not a DVCSCFFNNPytorch; "
                    << "the differentiable cross-section path requires the "
                    << "tensor-returning CFF API.");

    // One NN forward pass for all 8 leading-twist CFF components.
    DVCSCFFNNPytorch::AllCFFsTensor cffs = pCFF->computeAllCFFsTensor();

    // Dress with BMJ12 kinematic helicity coefficients and build the
    // φ-independent Fourier coefficients (purely real for unpolarised
    // target, with CFF_FT = CFF_FLT = 0).
    Theory::DressedCFFs dc = Theory::computeDressedCFFs(
            m_torchKin,
            cffs.H_re,  cffs.H_im,
            cffs.E_re,  cffs.E_im,
            cffs.Ht_re, cffs.Ht_im,
            cffs.Et_re, cffs.Et_im);

    return {Theory::computeVCSCoeffs   (m_torchKin, dc),
            Theory::computeInterfCoeffs(m_torchKin, dc)};
}

// ---------------------------------------------------------------------------
// crossSectionAtPhiTensor — now takes pre-computed Fourier coefficients.
// Just the φ-dependent assembly is done per call; everything upstream
// (NN forward, dressed CFFs, Fourier coeffs) lives in the caller and is
// built once per kinematic point.
// ---------------------------------------------------------------------------

torch::Tensor DVCSProcessBMJ12Torch::crossSectionAtPhiTensor(
        const Theory::VCSCoeffs&    vcs,
        const Theory::InterfCoeffs& interf,
        double phi, double beamHelicity) {

    return Theory::crossSectionAtPhi(
            m_torchKin, vcs, interf, phi, beamHelicity);
}