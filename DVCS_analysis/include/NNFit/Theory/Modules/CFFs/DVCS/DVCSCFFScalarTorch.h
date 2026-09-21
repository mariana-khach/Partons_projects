//
// Created by Mariana Khachatryan on 9/21/26.
//

#ifndef DVCS_CFF_SCALAR_TORCH_H
#define DVCS_CFF_SCALAR_TORCH_H

#include <partons/beans/automation/BaseObjectData.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>

#include <map>
#include <string>

#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFModuleTorch.h"

/**
 * @class DVCSCFFScalarTorch
 *
 * @brief Drives the tensor chain from an ordinary scalar PARTONS CFF model.
 *
 * Presents any PARTONS::DVCSConvolCoeffFunctionModule (DVCSCFFConstant,
 * DVCSCFFStandard, DVCSCFFDispersionRelation, ...) as a DVCSCFFModuleTorch, by
 * evaluating it per data point and packing the results into [N] complex
 * tensors. Those tensors carry no gradient -- a parametric model has no NN
 * parameters to differentiate -- which the chain downstream does not care
 * about: it multiplies CFF tensors by no-grad kinematics either way, and the
 * observable simply comes back detached.
 *
 * Purpose: a differential test of the batched BMJ12 port. Feeding the SAME
 * scalar CFF model to PARTONS' native process module and to
 * DVCSProcessBMJ12Torch compares two independent transcriptions of BMJ12 over
 * as many kinematic points as you like. The observ_calc_torch_scalar check
 * runs the network on both sides and cannot isolate the process layer.
 *
 * Dual base, exactly like DVCSCFFNNTorch: a PARTONS module for identity,
 * registration and the scalar contract, plus the torch mixin for the tensor
 * interface. That is what lets it be wired the ordinary way --
 * setConvolCoeffFunctionModule(), then found by the same cross-cast every CFF
 * source goes through -- rather than needing an injection path of its own. The
 * torch chain mirrors the scalar chain, so a second way to attach a CFF module
 * would be a deviation, not a convenience.
 *
 * The wrapped model is set with setScalarModule() after construction, because
 * PARTONS modules are created by the factory from a registered prototype and
 * cannot take constructor arguments. Non-owning: the caller creates the
 * wrapped module through the factory and outlives this one.
 */
class DVCSCFFScalarTorch : public PARTONS::DVCSConvolCoeffFunctionModule,
        public DVCSCFFModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    /**
     * Constructor.
     * @param className Name of last child class.
     */
    DVCSCFFScalarTorch(const std::string& className);

    virtual ~DVCSCFFScalarTorch();

    virtual DVCSCFFScalarTorch* clone() const;

    /**
     * The scalar CFF model to evaluate. Must be set before use; non-owning.
     */
    void setScalarModule(PARTONS::DVCSConvolCoeffFunctionModule* pScalarCFF);

    /**
     * Scalar contract: delegate to the wrapped model at this module's current
     * kinematics, so the object behaves like the model it wraps if PARTONS'
     * ordinary pipeline drives it.
     */
    virtual std::complex<double> computeCFF();

    /**
     * Evaluates the wrapped model once per point (N calls, each returning all
     * four CFFs) and stacks the results. Components the model does not provide
     * come back zero, matching the network's behavior for CFFs outside its
     * output layer.
     * @param xi,t,Q2,muF2,muR2 [N] CCF kinematics, as handed down by the
     *        process module (see DVCSCFFModuleTorch).
     * @return four [N] complex float64 tensors, requires_grad = false.
     */
    AllCFFsTensorBatch computeAllCFFsTensorBatch(const torch::Tensor& xi,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& muF2, const torch::Tensor& muR2) override;

protected:

    /**
     * Copy constructor.
     * @param other Object to be copied.
     */
    DVCSCFFScalarTorch(const DVCSCFFScalarTorch& other);

private:

    PARTONS::DVCSConvolCoeffFunctionModule* m_pScalarCFF; ///< Wrapped model (non-owning).
};

#endif /* DVCS_CFF_SCALAR_TORCH_H */