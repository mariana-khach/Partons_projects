//
// Created by Mariana Khachatryan on 9/22/26.
//

#ifndef DVCS_AC_COS0PHI_TORCH_H
#define DVCS_AC_COS0PHI_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAcCos0Phi.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAcTorch.h"

/**
 * @class DVCSAcCos0PhiTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSAcCos0Phi: the average of A_C over phi (weight 1, normalized by 2*pi -- the n = 0
 *        Fourier coefficient carries the other normalization).
 *
 * Mirrors the scalar class exactly -- same weight, same normalization, same
 * reuse of the parent's pointwise asymmetry as the integrand.
 */
class DVCSAcCos0PhiTorch: public PARTONS::DVCSAcCos0Phi,
        public DVCSObservableTorch, public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAcCos0PhiTorch(const std::string& className);
    virtual ~DVCSAcCos0PhiTorch();

    virtual DVCSAcCos0PhiTorch* clone() const override;

protected:

    DVCSAcCos0PhiTorch(const DVCSAcCos0PhiTorch& other);

    PARTONS::PhysicalType<torch::Tensor> computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    PARTONS::PhysicalType<torch::Tensor> computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /**
     * Scalar entry point. On a torch process: computeTensor() under
     * NoGradGuard, detached. On any other process: the native
     * PARTONS::DVCSAcCos0Phi this class derives from -- its own phi
     * integral over PARTONS::DVCSAc, so the leaf composes with any
     * process module, as its scalar twin does.
     */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;
};

#endif /* DVCS_AC_COS0PHI_TORCH_H */
