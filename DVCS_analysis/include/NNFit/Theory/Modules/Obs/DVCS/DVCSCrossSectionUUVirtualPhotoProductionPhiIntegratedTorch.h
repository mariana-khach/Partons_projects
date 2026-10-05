//
// Created by Mariana Khachatryan on 9/23/26.
//

#ifndef DVCSCROSS_SECTION_UUVIRTUAL_PHOTO_PRODUCTION_PHI_INTEGRATED_TORCH_H
#define DVCSCROSS_SECTION_UUVIRTUAL_PHOTO_PRODUCTION_PHI_INTEGRATED_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUVirtualPhotoProductionTorch.h"

/**
 * @class DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated: its parent's
 *        cross section integrated over the full phi range.
 *
 * Mirrors the scalar class exactly -- same interval, and NO normalization
 * (unlike the Fourier moments, which divide by pi or 2pi). Derives from
 * PARTONS::DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated itself (: DVCSCrossSectionUUVirtualPhotoProduction +
 * MathIntegratorModule) plus the torch mixins, and calls the pointwise torch
 * leaf's static cross section rather than inheriting it.
 *
 * ⚠️ The integrand is a CROSS SECTION, not an asymmetry. It is dominated by
 * the Bethe-Heitler peak where the lepton propagators nearly vanish, near
 * phi = 0 and 2pi -- a far harder integrand than the smooth, bounded
 * asymmetries the rest of this family integrates. The GL order here was
 * therefore chosen by its own measurement, not inherited from the A_LU
 * leaves; see the constructor.
 */
class DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch: public PARTONS::DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated,
        public DVCSObservableTorch, public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch(const std::string& className);
    virtual ~DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch();

    virtual DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch* clone() const override;

protected:

    DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch(const DVCSCrossSectionUUVirtualPhotoProductionPhiIntegratedTorch& other);

    PARTONS::PhysicalType<torch::Tensor> computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    PARTONS::PhysicalType<torch::Tensor> computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /**
     * Scalar entry point. On a torch process: computeTensor() under
     * NoGradGuard, detached. On any other process: the native
     * PARTONS::DVCSCrossSectionUUVirtualPhotoProductionPhiIntegrated this class derives from -- its own phi
     * integral over PARTONS::DVCSCrossSectionUUVirtualPhotoProduction, so the leaf composes with any
     * process module, as its scalar twin does.
     */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;
};

#endif /* DVCSCROSS_SECTION_UUVIRTUAL_PHOTO_PRODUCTION_PHI_INTEGRATED_TORCH_H */
