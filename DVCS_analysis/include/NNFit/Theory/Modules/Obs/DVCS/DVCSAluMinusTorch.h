//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef DVCS_ALU_MINUS_TORCH_H
#define DVCS_ALU_MINUS_TORCH_H

#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinus.h>
#include <partons/utils/type/PhysicalType.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h"

class DVCSProcessModuleTorch;

/**
 * @class DVCSAluMinusTorch
 *
 * @brief Differentiable (libtorch) twin of PARTONS::DVCSAluMinus — the pointwise
 *        beam-spin asymmetry for negative beam charge:
 *          A_LU(phi) = (sigma(+) - sigma(-)) / (sigma(+) + sigma(-)).
 *
 * Mirrors the scalar hierarchy exactly: this is the reusable pointwise layer
 * (sibling of DVCSAluMinus), and Fourier-moment observables
 * (DVCSAluMinusSin1PhiTorch, a future DVCSAluMinusCos0PhiTorch, ...) derive from
 * it and reuse aLUTensor() — just as the scalar moment classes derive from
 * DVCSAluMinus and reuse its computeObservable().
 *
 * Subclasses PARTONS::DVCSAluMinus so it is a scalar drop-in; the inherited
 * scalar computeObservable() is overridden to return computeTensor().item()
 * (no gradient), making the tensor computation the single source of truth.
 */
class DVCSAluMinusTorch: public PARTONS::DVCSAluMinus,
        public DVCSObservableTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAluMinusTorch(const std::string& className);
    virtual ~DVCSAluMinusTorch();

    virtual DVCSAluMinusTorch* clone() const override;

protected:

    DVCSAluMinusTorch(const DVCSAluMinusTorch& other);

    /**
     * Pointwise A_LU at the kinematic's stored phi (the ObservableTorch hook;
     * tensor twin of DVCSAluMinus::computeObservable). Moment subclasses override
     * this with their Fourier integral.
     * @return 0-d torch::Tensor, grad-connected to the NN parameters.
     */
    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Reusable pointwise asymmetry A_LU(phi), batched over a [N] tensor of phi.
     * Shared by every Fourier-moment subclass.
     * @return [N] tensor A_LU(phi), grad-connected to the NN CFF parameters.
     */
    torch::Tensor aLUTensor(const PARTONS::DVCSObservableKinematic& kinematic,
            const torch::Tensor& phi);

    /** Scalar wrapper over computeTensor() (detached) for the scalar pipeline. */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;

    /** Cross-cast the attached process module to its tensor interface. */
    DVCSProcessModuleTorch* torchProcessModule();
};

#endif /* DVCS_ALU_MINUS_TORCH_H */