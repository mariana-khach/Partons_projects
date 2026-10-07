//
// Created by Mariana Khachatryan on 9/23/26.
//

#ifndef DVCSCROSS_SECTION_DIFFERENCE_LUMINUS_TORCH_H
#define DVCSCROSS_SECTION_DIFFERENCE_LUMINUS_TORCH_H

#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionDifferenceLUMinus.h>
#include <partons/utils/type/PhysicalType.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h"

class DVCSProcessModuleTorch;

/**
 * @class DVCSCrossSectionDifferenceLUMinusTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSCrossSectionDifferenceLUMinus.
 *
 *        Beam-spin DIFFERENCE of the cross section at beam charge -1.
 *
 * Unlike the asymmetry leaves this returns a DIMENSIONFUL quantity, in nb.
 * The process module works in GeV^-2 (PhysicalUnit::GEVm2 on the scalar side),
 * so the tensor result is converted with Constant::CONV_GEVm2_TO_NBARN -- the
 * same factor PARTONS' makeSameUnitAs(PhysicalUnit::NB) applies. The torch
 * chain carries no unit system, so the conversion is explicit here and the
 * scalar wrapper tags its result PhysicalUnit::NB, as the PARTONS class does.
 */
class DVCSCrossSectionDifferenceLUMinusTorch: public PARTONS::DVCSCrossSectionDifferenceLUMinus,
        public DVCSObservableTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSCrossSectionDifferenceLUMinusTorch(const std::string& className);
    virtual ~DVCSCrossSectionDifferenceLUMinusTorch();

    virtual DVCSCrossSectionDifferenceLUMinusTorch* clone() const override;

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
    static PARTONS::PhysicalType<torch::Tensor> crossSectionNbTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& xB,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& E, const torch::Tensor& phi);

    /**
     * Scalar entry point. On a torch process: computeTensor() under
     * NoGradGuard, detached. On any other process: the native PARTONS::DVCSCrossSectionDifferenceLUMinus
     * this class derives from, so the leaf composes with any process module.
     */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;


protected:

    DVCSCrossSectionDifferenceLUMinusTorch(const DVCSCrossSectionDifferenceLUMinusTorch& other);
};

#endif /* DVCSCROSS_SECTION_DIFFERENCE_LUMINUS_TORCH_H */
