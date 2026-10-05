//
// Created by Mariana Khachatryan on 9/22/26.
//

#ifndef DVCS_ALU_INT_SIN1PHI_TORCH_H
#define DVCS_ALU_INT_SIN1PHI_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluIntSin1Phi.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluIntTorch.h"

/**
 * @class DVCSAluIntSin1PhiTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSAluIntSin1Phi: the sin(1phi) Fourier
 *        moment of the charge-differenced A_LU,
 *
 *          (1/pi) * integral_0^2pi  A_LU(phi) * sin(1phi) dphi
 *
 * mirroring the scalar class exactly -- same weight, same 1/pi normalization,
 * same reuse of the parent's pointwise asymmetry as the integrand.
 *
 * Derives from PARTONS::DVCSAluIntSin1Phi itself -- and through it from its pointwise
 * PARTONS class, exactly like the scalar class -- plus the DVCSObservableTorch
 * and MathIntegratorModuleTorch mixins. The pointwise tensor layer is reused by
 * calling DVCSAluIntTorch's static aLUTensorBatch(), not by inheriting
 * DVCSAluIntTorch, which would duplicate the PARTONS pointwise base.
 */
class DVCSAluIntSin1PhiTorch: public PARTONS::DVCSAluIntSin1Phi,
        public DVCSObservableTorch, public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAluIntSin1PhiTorch(const std::string& className);
    virtual ~DVCSAluIntSin1PhiTorch();

    virtual DVCSAluIntSin1PhiTorch* clone() const override;

protected:

    DVCSAluIntSin1PhiTorch(const DVCSAluIntSin1PhiTorch& other);

    /** N=1 wrapper over computeTensorImplBatch(). */
    PARTONS::PhysicalType<torch::Tensor> computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * The Fourier moment over N points at once. Unpacks the list into [N]
     * kinematics (each point's own phi is irrelevant -- this integrates over
     * the full phi range) and reduces the parent's aLUTensorBatch() over the
     * shared quadrature grid.
     */
    PARTONS::PhysicalType<torch::Tensor> computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /**
     * Scalar entry point. On a torch process: computeTensor() under
     * NoGradGuard, detached. On any other process: the native
     * PARTONS::DVCSAluIntSin1Phi this class derives from -- its own phi
     * integral over PARTONS::DVCSAluInt, so the leaf composes with any
     * process module, as its scalar twin does.
     */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;
};

#endif /* DVCS_ALU_INT_SIN1PHI_TORCH_H */
