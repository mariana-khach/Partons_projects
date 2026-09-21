//
// Created by Mariana Khachatryan on 9/21/26.
//

#ifndef DVCS_CFF_MODULE_TORCH_H
#define DVCS_CFF_MODULE_TORCH_H

#include <torch/torch.h>

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
class DVCSCFFModuleTorch {

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
     * @param xB,t,Q2 [N] raw kinematics tensors (xB directly, NOT skewness:
     *                the batched path does not round-trip through xi).
     */
    virtual AllCFFsTensorBatch computeAllCFFsTensorBatch(const torch::Tensor& xB,
            const torch::Tensor& t, const torch::Tensor& Q2) = 0;
};

#endif /* DVCS_CFF_MODULE_TORCH_H */