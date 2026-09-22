//
// Created by Mariana Khachatryan on 9/22/26.
//

#ifndef DVCS_ALU_DVCS_SIN1PHI_TORCH_H
#define DVCS_ALU_DVCS_SIN1PHI_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluDVCSTorch.h"

/**
 * @class DVCSAluDVCSSin1PhiTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSAluDVCSSin1Phi: the sin(1phi) Fourier
 *        moment of the charge-summed A_LU,
 *
 *          (1/pi) * integral_0^2pi  A_LU(phi) * sin(1phi) dphi
 *
 * mirroring the scalar class exactly -- same weight, same 1/pi normalization,
 * same reuse of the parent's pointwise asymmetry as the integrand.
 *
 * Derives from DVCSAluDVCSTorch (the pointwise layer) plus MathIntegratorModuleTorch
 * (a pure mixin, so no diamond), just as PARTONS::DVCSAluDVCSSin1Phi derives from its own
 * pointwise class plus MathIntegratorModule.
 */
class DVCSAluDVCSSin1PhiTorch: public DVCSAluDVCSTorch, public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAluDVCSSin1PhiTorch(const std::string& className);
    virtual ~DVCSAluDVCSSin1PhiTorch();

    virtual DVCSAluDVCSSin1PhiTorch* clone() const override;

protected:

    DVCSAluDVCSSin1PhiTorch(const DVCSAluDVCSSin1PhiTorch& other);

    /** N=1 wrapper over computeTensorImplBatch(). */
    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * The Fourier moment over N points at once. Unpacks the list into [N]
     * kinematics (each point's own phi is irrelevant -- this integrates over
     * the full phi range) and reduces the parent's aLUTensorBatch() over the
     * shared quadrature grid.
     */
    torch::Tensor computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;
};

#endif /* DVCS_ALU_DVCS_SIN1PHI_TORCH_H */
