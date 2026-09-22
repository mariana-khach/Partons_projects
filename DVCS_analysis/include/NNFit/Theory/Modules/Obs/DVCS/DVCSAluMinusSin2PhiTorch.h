//
// Created by Mariana Khachatryan on 9/22/26.
//

#ifndef DVCS_ALU_MINUS_SIN2PHI_TORCH_H
#define DVCS_ALU_MINUS_SIN2PHI_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusTorch.h"

/**
 * @class DVCSAluMinusSin2PhiTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSAluMinusSin2Phi: the sin(2phi) Fourier
 *        moment of A_LU at beam charge -1,
 *
 *          (1/pi) * integral_0^2pi  A_LU(phi) * sin(2phi) dphi
 *
 * mirroring the scalar class exactly -- same weight, same 1/pi normalization,
 * same reuse of the parent's pointwise asymmetry as the integrand.
 *
 * Derives from DVCSAluMinusTorch (the pointwise layer) plus MathIntegratorModuleTorch
 * (a pure mixin, so no diamond), just as PARTONS::DVCSAluMinusSin2Phi derives from its own
 * pointwise class plus MathIntegratorModule.
 */
class DVCSAluMinusSin2PhiTorch: public DVCSAluMinusTorch, public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAluMinusSin2PhiTorch(const std::string& className);
    virtual ~DVCSAluMinusSin2PhiTorch();

    virtual DVCSAluMinusSin2PhiTorch* clone() const override;

protected:

    DVCSAluMinusSin2PhiTorch(const DVCSAluMinusSin2PhiTorch& other);

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

#endif /* DVCS_ALU_MINUS_SIN2PHI_TORCH_H */
