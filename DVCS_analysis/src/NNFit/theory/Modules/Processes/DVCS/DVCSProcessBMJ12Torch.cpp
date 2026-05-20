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
// crossSectionAtPhiTensor
// ---------------------------------------------------------------------------

torch::Tensor DVCSProcessBMJ12Torch::crossSectionAtPhiTensor(
        double phi, double beamHelicity) {

    // (1) Ensure the φ-independent torch-side kinematics are up to date.
    buildTorchKinematics();

    // (2) Get the attached CFF module as a DVCSCFFNNPytorch*.
    DVCSCFFNNPytorch* pCFF =
            dynamic_cast<DVCSCFFNNPytorch*>(m_pConvolCoeffFunctionModule);

    if (pCFF == nullptr)
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                    << "Attached CFF module is not a DVCSCFFNNPytorch; "
                    << "the differentiable cross-section path requires the "
                    << "tensor-returning CFF API.");

    // (3) Pull the 8 leading-twist CFF tensors. One NN forward pass per
    //     GPD type — same redundancy as the existing PARTONS scalar path;
    //     can be optimised later by adding a batched CFF accessor.
    auto [H_re,  H_im ] = pCFF->computeCFFTensor(PARTONS::GPDType::H);
    auto [E_re,  E_im ] = pCFF->computeCFFTensor(PARTONS::GPDType::E);
    auto [Ht_re, Ht_im] = pCFF->computeCFFTensor(PARTONS::GPDType::Ht);
    auto [Et_re, Et_im] = pCFF->computeCFFTensor(PARTONS::GPDType::Et);

    // (4) Dress with BMJ12 kinematic helicity coefficients and build
    //     the φ-independent Fourier coefficients.
    Theory::DressedCFFs dc = Theory::computeDressedCFFs(
            m_torchKin,
            H_re, H_im, E_re, E_im, Ht_re, Ht_im, Et_re, Et_im);

    Theory::VCSCoeffs    vcs    = Theory::computeVCSCoeffs(m_torchKin, dc);
    Theory::InterfCoeffs interf = Theory::computeInterfCoeffs(m_torchKin, dc);

    // (5) Total cross-section at this single φ point.
    return Theory::crossSectionAtPhi(
            m_torchKin, vcs, interf, phi, beamHelicity);
}