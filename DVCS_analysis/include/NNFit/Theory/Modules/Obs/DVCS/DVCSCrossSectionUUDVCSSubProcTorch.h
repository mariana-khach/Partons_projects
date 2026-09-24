//
// Created by Mariana Khachatryan on 9/23/26.
//

#ifndef DVCSCROSS_SECTION_UUDVCSSUB_PROC_TORCH_H
#define DVCSCROSS_SECTION_UUDVCSSUB_PROC_TORCH_H

#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUDVCSSubProc.h>
#include <partons/utils/type/PhysicalType.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h"

class DVCSProcessModuleTorch;

/**
 * @class DVCSCrossSectionUUDVCSSubProcTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSCrossSectionUUDVCSSubProc.
 *
 *        Pure-DVCS (VCS) sub-process alone, unpolarized.
 *
 * Unlike the asymmetry leaves this returns a DIMENSIONFUL quantity, in nb.
 * The process module works in GeV^-2 (PhysicalUnit::GEVm2 on the scalar side),
 * so the tensor result is converted with Constant::CONV_GEVm2_TO_NBARN -- the
 * same factor PARTONS' makeSameUnitAs(PhysicalUnit::NB) applies. The torch
 * chain carries no unit system, so the conversion is explicit here and the
 * scalar wrapper tags its result PhysicalUnit::NB, as the PARTONS class does.
 */
class DVCSCrossSectionUUDVCSSubProcTorch: public PARTONS::DVCSCrossSectionUUDVCSSubProc,
        public DVCSObservableTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSCrossSectionUUDVCSSubProcTorch(const std::string& className);
    virtual ~DVCSCrossSectionUUDVCSSubProcTorch();

    virtual DVCSCrossSectionUUDVCSSubProcTorch* clone() const override;

    /** Pointwise cross section at each kinematic's OWN phi, batched over N. */
    PARTONS::PhysicalType<torch::Tensor> computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /** N=1 wrapper over computeTensorImplBatch(). */
    PARTONS::PhysicalType<torch::Tensor> computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Reusable cross section in nb, [N,M] over shared quadrature nodes or
     * [N,1] at each point's own phi -- the analogue of the asymmetry family's
     * aLUTensorBatch()/aCTensorBatch(). The phi-integrated subclass uses it as
     * its integrand, exactly as the Fourier-moment leaves use theirs.
     */
    PARTONS::PhysicalType<torch::Tensor> crossSectionNbTensorBatch(const torch::Tensor& xB,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& E, const torch::Tensor& phi);

    /** Scalar wrapper over computeTensor() (detached), tagged nb. */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;

    /** Cross-cast the attached process module to its tensor interface. */
    DVCSProcessModuleTorch* torchProcessModule();

protected:

    DVCSCrossSectionUUDVCSSubProcTorch(const DVCSCrossSectionUUDVCSSubProcTorch& other);
};

#endif /* DVCSCROSS_SECTION_UUDVCSSUB_PROC_TORCH_H */
