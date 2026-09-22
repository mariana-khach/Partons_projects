//
// Created by Mariana Khachatryan on 9/22/26.
//

#ifndef DVCS_AC_TORCH_H
#define DVCS_AC_TORCH_H

#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAc.h>
#include <partons/utils/type/PhysicalType.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h"

class DVCSProcessModuleTorch;

/**
 * @class DVCSAcTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSAc -- the pointwise BEAM-CHARGE
 *        asymmetry. Writing sigma(lambda, charge):
 *
 *          A_C(phi) = ((s+++s-+) - (s+-+s--)) / ((s+++s-+) + (s+-+s--))
 *
 * The exact transpose of DVCSAluDVCSTorch: there the charge is summed and the
 * asymmetry taken in beam helicity; here the helicity is summed (unpolarized
 * beam) and the asymmetry taken in beam charge, which isolates the
 * interference term -- odd in charge -- against the charge-even BH+VCS.
 *
 * Its Fourier moments are COSINE moments, A_C being even in phi, where the
 * A_LU family takes sine moments.
 */
class DVCSAcTorch: public PARTONS::DVCSAc, public DVCSObservableTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAcTorch(const std::string& className);
    virtual ~DVCSAcTorch();

    virtual DVCSAcTorch* clone() const override;

    /** Pointwise A_C at each kinematic's OWN phi, batched over N points. */
    torch::Tensor computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /** N=1 wrapper over computeTensorImplBatch(). */
    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Reusable pointwise asymmetry, [N,M] over shared quadrature nodes or
     * [N,1] at each point's own phi. Named aCTensorBatch to match the family
     * it belongs to; the A_LU classes call theirs aLUTensorBatch.
     */
    torch::Tensor aCTensorBatch(const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E,
            const torch::Tensor& phi);

    /** Scalar wrapper over computeTensor() (detached) for the scalar pipeline. */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;

    /** Cross-cast the attached process module to its tensor interface. */
    DVCSProcessModuleTorch* torchProcessModule();

protected:

    DVCSAcTorch(const DVCSAcTorch& other);

    /** The asymmetry formula; see the class brief. */
    virtual torch::Tensor asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& phi);
};

#endif /* DVCS_AC_TORCH_H */
