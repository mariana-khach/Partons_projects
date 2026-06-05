/**
 * @file DVCSAluMinusSin1PhiTorch.cpp
 *
 * Implements the PARTONS-registered differentiable A_LU^{sin1φ}
 * observable. computeTensor() runs 10-point Gauss–Legendre quadrature
 * over φ ∈ [0, 2π], retrieving per-φ tensor cross-sections from the
 * attached DVCSProcessBMJ12Torch process module so that the autograd
 * graph from NN weights to the integrated asymmetry stays intact.
 *
 * The φ slot of the input kinematic is ignored — φ is integrated out
 * by the quadrature. (xB, t, Q², E) are pushed onto the process module
 * via DVCSProcessBMJ12Torch::setupKinematics(), which calls the
 * inherited protected setKinematics() and refreshes the Theory::DVCSKin
 * cache without running the scalar BMJ12 pipeline.
 *
 * GL nodes/weights are hardcoded from scipy.special.p_roots(10).
 */

#include "NNFit/theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectRegistry.h>

#include <cmath>

#include "NNFit/theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.h"

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const unsigned int DVCSAluMinusSin1PhiTorch::classId =
        PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSAluMinusSin1PhiTorch("DVCSAluMinusSin1PhiTorch"));

// ---------------------------------------------------------------------------
// 10-point Gauss–Legendre nodes & weights on [-1, 1]
// (from scipy.special.p_roots(10))
// ---------------------------------------------------------------------------

const double DVCSAluMinusSin1PhiTorch::s_gl_nodes[10] = {
    -0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
     0.1488743389816312,
     0.4333953941292472,
     0.6794095682990244,
     0.8650633666889845,
     0.9739065285171717
};

const double DVCSAluMinusSin1PhiTorch::s_gl_weights[10] = {
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159820,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159820,
    0.1494513491505806,
    0.0666713443086881
};

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

DVCSAluMinusSin1PhiTorch::DVCSAluMinusSin1PhiTorch(const std::string& className)
        : DVCSAluMinusSin1Phi(className) {
}

DVCSAluMinusSin1PhiTorch::DVCSAluMinusSin1PhiTorch(
        const DVCSAluMinusSin1PhiTorch& other)
        : DVCSAluMinusSin1Phi(other) {
}

DVCSAluMinusSin1PhiTorch::~DVCSAluMinusSin1PhiTorch() {
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

DVCSAluMinusSin1PhiTorch* DVCSAluMinusSin1PhiTorch::clone() const {
    return new DVCSAluMinusSin1PhiTorch(*this);
}

// ---------------------------------------------------------------------------
// PARTONS boilerplate (delegates to the base class)
// ---------------------------------------------------------------------------

void DVCSAluMinusSin1PhiTorch::configure(
        const ElemUtils::Parameters& parameters) {
    DVCSAluMinusSin1Phi::configure(parameters);
}

void DVCSAluMinusSin1PhiTorch::resolveObjectDependencies() {
    DVCSAluMinusSin1Phi::resolveObjectDependencies();
}

void DVCSAluMinusSin1PhiTorch::prepareSubModules(
        const std::map<std::string, PARTONS::BaseObjectData>& subModulesData) {
    DVCSAluMinusSin1Phi::prepareSubModules(subModulesData);
}

void DVCSAluMinusSin1PhiTorch::initModule() {
    DVCSAluMinusSin1Phi::initModule();
}

void DVCSAluMinusSin1PhiTorch::isModuleWellConfigured() {
    DVCSAluMinusSin1Phi::isModuleWellConfigured();
}

// ---------------------------------------------------------------------------
// computeTensor
// ---------------------------------------------------------------------------

torch::Tensor DVCSAluMinusSin1PhiTorch::computeTensor(
        const PARTONS::DVCSObservableKinematic& kinematic) {

    // (1) Cast the attached process module to its tensor-capable subclass.
    DVCSProcessBMJ12Torch* pProc =
            dynamic_cast<DVCSProcessBMJ12Torch*>(m_pProcessModule);

    if (pProc == nullptr)
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                    << "Attached process module is not a "
                    << "DVCSProcessBMJ12Torch; the differentiable "
                    << "observable path requires the tensor-returning "
                    << "cross-section API.");

    // (2) Push (xB, t, Q², E) onto the process module. setupKinematics()
    //     calls the inherited protected setKinematics() and refreshes the
    //     cached Theory::DVCSKin — no scalar BH+VCS+Interf pipeline runs,
    //     so no wasted NN forwards or scalar physics arithmetic.
    pProc->setupKinematics(kinematic);

    // (3) Single NN forward + Fourier-coefficient build, before the
    //     quadrature loop. Both (VCS², BH-DVCS interference) coeffs are
    //     φ-independent, so we compute them once and reuse across all
    //     10 GL nodes × 2 helicities = 20 cross-section evaluations.
    //     This avoids 4× redundant per-GPD NN forwards and the dressed
    //     CFF / Fourier build that would otherwise occur on every call.
    auto [vcs, interf] = pProc->computeFourierCoeffsTensor();

    // (4) 10-point Gauss–Legendre quadrature over φ ∈ [0, 2π].
    //
    //     nodes uₖ ∈ [-1, 1]  →  φₖ = π(1 + uₖ) ∈ [0, 2π]
    //     ∫₀^{2π} f(φ) dφ ≈ π Σₖ wₖ f(φₖ)
    //
    //     A_LU^{sin1φ} = (1/π) ∫₀^{2π} sin(φ) A_LU^–(φ) dφ
    //                  ≈ Σₖ wₖ sin(φₖ) A_LU^–(φₖ)
    //
    //     A_LU^–(φ) = [σ(+1, φ) - σ(-1, φ)] / [σ(+1, φ) + σ(-1, φ)]
    const double PI = 3.141592653589793238462643383279502884;

    torch::Tensor result = torch::zeros({});

    for (int k = 0; k < 10; ++k) {
        const double phi_k = PI * (1. + s_gl_nodes[k]);
        const double w_k   = s_gl_weights[k];

        torch::Tensor sig_p = pProc->crossSectionAtPhiTensor(vcs, interf, phi_k, +1.);
        torch::Tensor sig_m = pProc->crossSectionAtPhiTensor(vcs, interf, phi_k, -1.);
        torch::Tensor denom = sig_p + sig_m;

        // Guard against pathological zero denominator.
        torch::Tensor alu_k = torch::where(
                denom.abs() > 1e-30,
                (sig_p - sig_m) / denom,
                torch::zeros_like(denom));

        result = result + w_k * std::sin(phi_k) * alu_k;
    }

    return result;
}