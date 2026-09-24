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
    PARTONS::PhysicalType<torch::Tensor> computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Reusable pointwise asymmetry A_LU(phi), batched over N data points x M
     * phi nodes. Shared by every Fourier-moment subclass: prepare once
     * (hoisting the helicity-independent setup), then hand off to the
     * asymmetryTensorBatch() hook, which assembles the cross sections its own
     * formula needs -- driven through the abstract DVCSProcessModuleTorch*
     * base, no concrete process-module type required.
     * @return [N,M] tensor A_LU(phi), grad-connected to the NN CFF parameters.
     */
    PARTONS::PhysicalType<torch::Tensor> aLUTensorBatch(const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E,
            const torch::Tensor& phi);

    /**
     * Batched (N-point) sibling of computeTensorImpl(): pointwise A_LU(phi)
     * with each kinematic evaluated at ITS OWN phi.
     *
     * Shares aLUTensorBatch() with the Fourier-moment leaves; only phi's shape
     * selects the mode, since every phi-dependent term downstream broadcasts
     * [N] kinematics (unsqueezed to [N,1]) against whatever phi is:
     *
     *   phi [M]    -> [N,M]   every point at every quadrature node (moments)
     *   phi [N,1]  -> [N,1]   point i at its own phi_i (this)
     *
     * One batched evaluation over all N points -- no per-point loop and no
     * O(N^2) diagonal extraction, which earlier notes assumed would be needed.
     * Cheaper than a moment: same operation count with M = 1 rather than 20.
     */
    PARTONS::PhysicalType<torch::Tensor> computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;

    /** Scalar wrapper over computeTensor() (detached) for the scalar pipeline. */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;

    /** Cross-cast the attached process module to its tensor interface. */
    DVCSProcessModuleTorch* torchProcessModule();

protected:

    /**
     * The asymmetry formula itself, given a process module on which
     * prepareTensorBatch() has already run. Each A_LU variant in PARTONS is a
     * different combination of sigma(lambda, charge), not a different
     * sub-process selection, so this hook assembles exactly the terms its own
     * formula needs -- mirroring the scalar classes, where each calls
     * ProcessModule::compute() as many times as its expression requires.
     * Writing sigma(lambda, charge):
     *
     *   AluMinus (this)  (s+- - s--) / (s+- + s--)
     *   AluPlus          (s++ - s-+) / (s++ + s-+)
     *   AluDVCS          ((s+++s+-) - (s-++s--)) / ((s+++s+-) + (s-++s--))
     *   AluInt           ((s++-s+-) - (s-+-s--)) / ((s+++s+-) + (s-++s--))
     *
     * The charge SUM cancels the interference term (odd in beam charge),
     * leaving the BH+VCS part; the charge DIFFERENCE isolates it. That is why
     * the DVCS/Int variants need four cross sections rather than a
     * VCSSubProcessType selector.
     *
     * @param proc Prepared process module (prepareTensorBatch already called).
     * @param phi  [M] shared quadrature nodes, or [N,1] per-point own phi.
     * @return Same shape as phi broadcast against [N]: [N,M] or [N,1].
     */
    virtual PARTONS::PhysicalType<torch::Tensor> asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& phi);
};

#endif /* DVCS_ALU_MINUS_TORCH_H */