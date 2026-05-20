#ifndef DVCS_ALU_MINUS_SIN1PHI_TORCH_H
#define DVCS_ALU_MINUS_SIN1PHI_TORCH_H

/**
 * @file DVCSAluMinusSin1PhiTorch.h
 *
 * @brief PARTONS-registered observable for the beam-spin asymmetry
 *
 *     A_LU^{sin1φ}(xB, t, Q²) = (1/π) ∫₀^{2π} dφ sin(φ) A_LU^–(φ)
 *
 * with a differentiable, torch::Tensor-returning entry point that
 * keeps the autograd graph alive from the NN weights all the way to
 * the final asymmetry.
 *
 * Subclasses PARTONS::DVCSAluMinusSin1Phi so the inherited scalar
 * PARTONS pipeline (Math integrator over φ, BH+VCS+Interf cross
 * sections via the attached process module) continues to work
 * unchanged through the existing virtual methods. On top of that,
 * this class adds:
 *
 *   computeTensor(kinematic) — runs a 10-point Gauss–Legendre
 *     quadrature over φ ∈ [0, 2π] using
 *     DVCSProcessBMJ12Torch::crossSectionAtPhiTensor(),
 *     returning a 0-d torch::Tensor with gradients w.r.t. the
 *     NN weights in the upstream DVCSCFFNNPytorch CFF module.
 *
 * The attached PARTONS process module must be a
 * DVCSProcessBMJ12Torch (the only one currently exposing the tensor
 * cross-section); the attached CFF module beneath it must be a
 * DVCSCFFNNPytorch.
 */

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinusSin1Phi.h>

#include <torch/torch.h>

#include <map>
#include <string>

class DVCSAluMinusSin1PhiTorch : public PARTONS::DVCSAluMinusSin1Phi {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    /**
     * Constructor.
     * @param className Name of last child class.
     */
    DVCSAluMinusSin1PhiTorch(const std::string& className);

    virtual ~DVCSAluMinusSin1PhiTorch();

    virtual DVCSAluMinusSin1PhiTorch* clone() const;

    virtual void configure(const ElemUtils::Parameters& parameters);
    virtual void resolveObjectDependencies();
    virtual void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    /**
     * Differentiable evaluation of A_LU^{sin1φ} at a kinematic point.
     *
     * Implementation: 10-point Gauss–Legendre quadrature over
     * φ ∈ [0, 2π] (nodes mapped from scipy.special.p_roots(10)),
     * with the per-φ cross-sections σ(±1, φ) supplied by the
     * attached DVCSProcessBMJ12Torch in tensor form. The φ slot of
     * the input kinematic is ignored (φ is integrated out).
     *
     * @param kinematic (xB, t, Q², E[, φ]) — φ ignored.
     * @return 0-d torch::Tensor carrying gradients to the NN weights.
     */
    torch::Tensor computeTensor(
            const PARTONS::DVCSObservableKinematic& kinematic);

protected:

    /**
     * Copy constructor.
     * @param other Object to be copied.
     */
    DVCSAluMinusSin1PhiTorch(const DVCSAluMinusSin1PhiTorch& other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    /// 10-point Gauss–Legendre nodes on [-1, 1] (scipy.special.p_roots(10)).
    static const double s_gl_nodes[10];

    /// 10-point Gauss–Legendre weights, paired with s_gl_nodes.
    static const double s_gl_weights[10];
};

#endif /* DVCS_ALU_MINUS_SIN1PHI_TORCH_H */