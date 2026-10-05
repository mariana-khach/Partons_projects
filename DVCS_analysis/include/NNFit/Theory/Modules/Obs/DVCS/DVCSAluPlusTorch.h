//
// Created by Mariana Khachatryan on 9/22/26.
//

#ifndef DVCS_ALU_PLUS_TORCH_H
#define DVCS_ALU_PLUS_TORCH_H

#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluPlus.h>
#include <partons/utils/type/PhysicalType.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h"

class DVCSProcessModuleTorch;

/**
 * @class DVCSAluPlusTorch
 *
 * @brief Differentiable (libtorch) twin of PARTONS::DVCSAluPlus -- the pointwise beam-spin asymmetry for POSITIVE beam charge:
 *          A_LU(phi) = (sigma(+1,+1) - sigma(-1,+1)) / (sigma(+1,+1) + sigma(-1,+1))
 *
 * A sibling of DVCSAluMinusTorch, not a subclass: each A_LU variant must BE its
 * own PARTONS observable for the scalar chain, and an AluPlus is not an
 * AluMinus. PARTONS' own classes are siblings for the same reason, each writing
 * its own formula over ProcessModule::compute(). The repeated machinery here
 * (prepare/assemble, the own-phi unpack) is the price of that mirroring.
 *
 * Fourier-moment leaves reuse the static aLUTensorBatch() without deriving
 * from this class: each derives from its own PARTONS moment class, which
 * derives from PARTONS::DVCSAluPlus -- so every torch class sits directly under
 * the PARTONS class it mirrors.
 */
class DVCSAluPlusTorch: public PARTONS::DVCSAluPlus, public DVCSObservableTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAluPlusTorch(const std::string& className);
    virtual ~DVCSAluPlusTorch();

    virtual DVCSAluPlusTorch* clone() const override;

    /**
     * Pointwise asymmetry at each kinematic's OWN phi, batched over N points.
     * See DVCSAluMinusTorch for why phi's shape selects the mode.
     */
    PARTONS::PhysicalType<torch::Tensor> computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /** N=1 wrapper over computeTensorImplBatch(). */
    PARTONS::PhysicalType<torch::Tensor> computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Reusable pointwise asymmetry, [N,M] over quadrature nodes or [N,1] at
     * each point's own phi. Prepares the process module once, then delegates
     * the formula to asymmetryTensorBatch().
     */
    static PARTONS::PhysicalType<torch::Tensor> aLUTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E,
            const torch::Tensor& phi);

    /**
     * Scalar entry point. On a torch process: computeTensor() under
     * NoGradGuard, detached. On any other process: the native PARTONS::DVCSAluPlus
     * this class derives from, so the leaf composes with any process module.
     */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;


protected:

    DVCSAluPlusTorch(const DVCSAluPlusTorch& other);

    /** The asymmetry formula; see the class brief. */
    static PARTONS::PhysicalType<torch::Tensor> asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& phi);
};

#endif /* DVCS_ALU_PLUS_TORCH_H */
