//
// Created by Mariana Khachatryan on 9/21/26.
//

#ifndef DVCS_CFF_SCALAR_TORCH_H
#define DVCS_CFF_SCALAR_TORCH_H

#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <partons/modules/scales/DVCS/DVCSScalesModule.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterModule.h>

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
 * as many kinematic points as you like. The standing observ_calc_torch_scalar
 * check does this at a single point with the network on both sides; this lifts
 * that to a whole dataset with a known, fixed CFF input, so a coefficient error
 * that only shows at large |t| or small xB has somewhere to show up.
 *
 * Kinematics are built exactly as PARTONS' scalar path builds them
 * (DVCSProcessModule::computeConvolCoeffFunction): xi from the xi-converter
 * module, muF2/muR2 from the scales module, both evaluated on the full
 * DVCSObservableKinematic -- the same module instances the process module is
 * wired with, so the adapter cannot drift from the scalar path's conventions.
 *
 * Wiring: this is NOT a PARTONS module (no classId, no registration, not
 * created through the module factory), so it cannot be passed to
 * setConvolCoeffFunctionModule(). Hand it to the process module with
 * DVCSProcessModuleTorch::setCFFModuleTorch() instead, which takes precedence
 * over the wired convol-coeff module for the tensor path. All three pointers
 * it holds are non-owning; the caller outlives the adapter.
 */
class DVCSCFFScalarTorch : public DVCSCFFModuleTorch {

public:

    /**
     * @param pScalarCFF   The scalar CFF model to evaluate (non-owning).
     * @param pXiConverter xB -> xi module, normally the same instance the
     *                     process module is wired with (non-owning).
     * @param pScales      muF2/muR2 module, likewise (non-owning).
     */
    DVCSCFFScalarTorch(PARTONS::DVCSConvolCoeffFunctionModule* pScalarCFF,
            PARTONS::DVCSXiConverterModule* pXiConverter,
            PARTONS::DVCSScalesModule* pScales);

    virtual ~DVCSCFFScalarTorch() = default;

    /**
     * Evaluates the scalar model once per point (N calls, each returning all
     * four CFFs) and stacks the results. Components the model does not provide
     * come back zero, matching the network's behavior for CFFs outside its
     * output layer.
     * @return four [N] complex float64 tensors, requires_grad = false.
     */
    AllCFFsTensorBatch computeAllCFFsTensorBatch(const torch::Tensor& xB,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& E) override;

private:

    PARTONS::DVCSConvolCoeffFunctionModule* m_pScalarCFF;
    PARTONS::DVCSXiConverterModule*         m_pXiConverter;
    PARTONS::DVCSScalesModule*              m_pScales;
};

#endif /* DVCS_CFF_SCALAR_TORCH_H */