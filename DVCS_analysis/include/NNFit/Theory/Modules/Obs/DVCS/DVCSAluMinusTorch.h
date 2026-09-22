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
 * it and reuse aLUTensorBatch() — just as the scalar moment classes derive from
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
     * tensor twin of DVCSAluMinus::computeObservable). A thin N=1 wrapper
     * around computeTensorImplBatch() (mirrors DVCSAluMinusSin1PhiTorch's own
     * N=1 wrapper) — not yet implemented at this base class, since
     * computeTensorImplBatch() itself throws here (see its doc comment).
     * Moment subclasses override this with their Fourier integral instead.
     * @return 0-d torch::Tensor, grad-connected to the NN parameters.
     */
    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Reusable pointwise asymmetry A_LU(phi), batched over N data points x M
     * phi nodes. Shared by every Fourier-moment subclass: prepare once
     * (hoisting the helicity-independent setup), assemble per helicity --
     * driven through the same abstract DVCSProcessModuleTorch* base
     * (prepareTensorBatch()/crossSectionTensorBatch()), no concrete
     * process-module type needed.
     * @return [N,M] tensor A_LU(phi), grad-connected to the NN CFF parameters.
     */
    torch::Tensor aLUTensorBatch(const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E,
            const torch::Tensor& phi);

    /**
     * Batched (N-point) sibling of computeTensorImpl() -- the
     * ObservableTorch<K> hook (channel-generic List<K>).
     *
     * NOT IMPLEMENTED at this pointwise base -- throws. A meaningful batched
     * pointwise A_LU would need each of the N kinematics' own phi matched
     * 1:1 (an [N] phi broadcast), whereas aLUTensorBatch()/
     * crossSectionTensorBatch() broadcast phi as a [M] axis shared by every
     * data point (an [N,M] outer product) -- built for the Fourier-moment
     * leaf (DVCSAluMinusSin1PhiTorch), which integrates every point over the
     * same quadrature nodes. Reusing it here would need either a new
     * per-point-phi broadcasting mode or a wasteful O(N^2) diagonal
     * extraction; skipped since no current consumer needs a batched
     * pointwise leaf. DVCSAluMinusSin1PhiTorch overrides this with a real,
     * O(N) implementation reusing aLUTensorBatch() as it was built for.
     * Kept as a concrete (non-pure) override only so this class -- which
     * self-registers its own prototype -- remains instantiable.
     */
    torch::Tensor computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /** Scalar wrapper over computeTensor() (detached) for the scalar pipeline. */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;

    /** Cross-cast the attached process module to its tensor interface. */
    DVCSProcessModuleTorch* torchProcessModule();
};

#endif /* DVCS_ALU_MINUS_TORCH_H */