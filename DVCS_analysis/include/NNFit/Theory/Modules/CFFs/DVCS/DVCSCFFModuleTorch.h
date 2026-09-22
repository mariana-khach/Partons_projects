//
// Created by Mariana Khachatryan on 9/21/26.
//

#ifndef DVCS_CFF_MODULE_TORCH_H
#define DVCS_CFF_MODULE_TORCH_H

#include <torch/torch.h>

#include "NNFit/Theory/Modules/CFFs/CFFModuleTorch.h"

namespace PARTONS {
class DVCSObservableKinematic;
} // namespace PARTONS

/**
 * @class DVCSCFFModuleTorch
 *
 * @brief Tensor twin of PARTONS::DVCSConvolCoeffFunctionModule: the CFF link
 * of the differentiable DVCS chain.
 *
 * Completes the mixin pattern the other two links already use. Each link of
 * the torch chain is a PARTONS class for identity/registration PLUS a torch
 * base for the tensor interface:
 *
 *   observable  PARTONS::DVCSAluMinus                + DVCSObservableTorch
 *   process     PARTONS::DVCSProcessBMJ12            + DVCSProcessModuleTorch
 *   CFF         PARTONS::DVCSConvolCoeffFunctionModule + this
 *
 * Until this class existed the CFF link had no torch base, so
 * DVCSProcessBMJ12Torch::setupKinematicsTorchBatch had to dynamic_cast to the
 * CONCRETE DVCSCFFNNTorch -- pinning the whole tensor chain to one CFF
 * implementation. Casting to this interface instead lets any tensor CFF source
 * drive the chain: a different network, or an adapter over a scalar PARTONS
 * CFF model (DVCSCFFStandard, DVCSCFFDispersionRelation, ...) for validating
 * the batched BMJ12 port against PARTONS' native arithmetic across a whole
 * dataset rather than at a single kinematic point.
 *
 * Sits at the channel layer, under the generic CFFModuleTorch<K> -- the same
 * shape as DVCSProcessModuleTorch under ProcessModuleTorch<K>, and as PARTONS'
 * own DVCSConvolCoeffFunctionModule under ConvolCoeffFunctionModule<K,R>. The
 * channel-specific part (which CFFs exist, and the signature that returns
 * them) lives here; the generic base is a marker, for the reasons its header
 * gives.
 *
 * The template argument is DVCSObservableKinematic, not the CCF kinematics
 * PARTONS templates its scalar CFF module on: the batched torch path hands the
 * CFF source observable-level kinematics (xB, t, Q2, E) and lets it do any
 * conversion itself -- the network wants xB directly, and DVCSCFFScalarTorch
 * runs the xi-converter and scales modules internally.
 *
 * Inherits nothing from PARTONS (pure mixin, like DVCSProcessModuleTorch), so
 * reaching it from a DVCSConvolCoeffFunctionModule* is a cross-cast between
 * unrelated base subobjects -- dynamic_cast, never static_cast.
 *
 * Gradients are the implementation's business, not the interface's: the chain
 * downstream multiplies these tensors by no-grad kinematics and works either
 * way. A CFF source built on a torch network returns grad-carrying tensors; a
 * scalar model's values enter as constants and the observable simply comes
 * back detached.
 */
class DVCSCFFModuleTorch
        : public CFFModuleTorch<PARTONS::DVCSObservableKinematic> {

public:

    virtual ~DVCSCFFModuleTorch() = default;

    /**
     * The four standard DVCS CFFs at N kinematic points, as [N] complex
     * (float64) tensors. Components the source does not provide are zero.
     */
    struct AllCFFsTensorBatch {
        torch::Tensor H;  ///< CFF H  ([N] complex double).
        torch::Tensor E;  ///< CFF E.
        torch::Tensor Ht; ///< CFF Ht (H-tilde).
        torch::Tensor Et; ///< CFF Et (E-tilde).
    };

    /**
     * All four CFFs for the whole batch in one evaluation.
     *
     * Takes the CONVOL-COEFF-FUNCTION kinematics, the same five quantities
     * PARTONS::DVCSConvolCoeffFunctionKinematic carries, because that is what
     * the scalar chain hands its CFF module: DVCSProcessModule::
     * computeConvolCoeffFunction runs the xi-converter and the scales module
     * and passes (xi, t, Q2, muF2, muR2) down. The process module converts;
     * the CFF module receives. Mirroring that here keeps the two chains
     * link-for-link and means an implementation wrapping a scalar model has
     * only to build the bean it is handed -- no xi-converter or scales module
     * of its own, and no beam energy it would otherwise have to be given
     * purely to feed them.
     *
     * A CFF source parameterized in xB (the network) converts back itself:
     * xB = 2*xi / (1 + xi), one tensor op.
     *
     * @param xi,t,Q2,muF2,muR2 [N] kinematics tensors, no grad.
     */
    virtual AllCFFsTensorBatch computeAllCFFsTensorBatch(const torch::Tensor& xi,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& muF2, const torch::Tensor& muR2) = 0;
};

#endif /* DVCS_CFF_MODULE_TORCH_H */