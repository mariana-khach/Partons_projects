//
// Created by Mariana Khachatryan on 9/23/26.
//

#ifndef DVCSCROSS_SECTION_UUDVCSSUB_PROC_PHI_INTEGRATED_TORCH_H
#define DVCSCROSS_SECTION_UUDVCSSUB_PROC_PHI_INTEGRATED_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSCrossSectionUUDVCSSubProcTorch.h"

/**
 * @class DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSCrossSectionUUDVCSSubProcPhiIntegrated: its parent's
 *        cross section integrated over the full phi range.
 *
 * Mirrors the scalar class exactly -- same interval, and NO normalization
 * (unlike the Fourier moments, which divide by pi or 2pi). Derives from the
 * pointwise torch leaf plus the integrator mixin, the same shape as
 * PARTONS::DVCSCrossSectionUUDVCSSubProcPhiIntegrated : DVCSCrossSectionUUDVCSSubProc + MathIntegratorModule.
 *
 * ⚠️ The integrand is a CROSS SECTION, not an asymmetry. It is dominated by
 * the Bethe-Heitler peak where the lepton propagators nearly vanish, near
 * phi = 0 and 2pi -- a far harder integrand than the smooth, bounded
 * asymmetries the rest of this family integrates. The GL order here was
 * therefore chosen by its own measurement, not inherited from the A_LU
 * leaves; see the constructor.
 */
class DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch: public DVCSCrossSectionUUDVCSSubProcTorch,
        public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch(const std::string& className);
    virtual ~DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch();

    virtual DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch* clone() const override;

protected:

    DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch(const DVCSCrossSectionUUDVCSSubProcPhiIntegratedTorch& other);

    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    torch::Tensor computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;
};

#endif /* DVCSCROSS_SECTION_UUDVCSSUB_PROC_PHI_INTEGRATED_TORCH_H */
