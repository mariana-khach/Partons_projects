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
 * Fourier-moment leaves derive from this and reuse aLUTensorBatch(), exactly as
 * the scalar moment classes derive from PARTONS::DVCSAluPlus.
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
    torch::Tensor computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /** N=1 wrapper over computeTensorImplBatch(). */
    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Reusable pointwise asymmetry, [N,M] over quadrature nodes or [N,1] at
     * each point's own phi. Prepares the process module once, then delegates
     * the formula to asymmetryTensorBatch().
     */
    torch::Tensor aLUTensorBatch(const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E,
            const torch::Tensor& phi);

    /** Scalar wrapper over computeTensor() (detached) for the scalar pipeline. */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;

    /** Cross-cast the attached process module to its tensor interface. */
    DVCSProcessModuleTorch* torchProcessModule();

protected:

    DVCSAluPlusTorch(const DVCSAluPlusTorch& other);

    /** The asymmetry formula; see the class brief. */
    virtual torch::Tensor asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& phi);
};

#endif /* DVCS_ALU_PLUS_TORCH_H */
