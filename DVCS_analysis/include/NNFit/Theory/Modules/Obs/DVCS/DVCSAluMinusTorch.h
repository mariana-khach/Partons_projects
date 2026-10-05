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
 * This is the reusable pointwise layer. The Fourier-moment leaves
 * (DVCSAluMinusSin1PhiTorch, DVCSAluMinusSin2PhiTorch) reuse its static
 * aLUTensorBatch() but do NOT derive from it: each derives from its own PARTONS
 * moment class, which derives from PARTONS::DVCSAluMinus -- so every torch
 * class sits directly under the PARTONS class it mirrors.
 *
 * Subclasses PARTONS::DVCSAluMinus so it is a scalar drop-in. The scalar
 * computeObservable() runs the tensor chain (detached) on a torch process
 * module, and PARTONS' own DVCSAluMinus on any other.
 */
class DVCSAluMinusTorch: public PARTONS::DVCSAluMinus,
        public DVCSObservableTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAluMinusTorch(const std::string& className);
    virtual ~DVCSAluMinusTorch();

    virtual DVCSAluMinusTorch* clone() const override;

    /**
     * Reusable pointwise asymmetry A_LU(phi), batched over N data points x M
     * phi nodes: prepare the process once (hoisting the helicity-independent
     * setup), then hand off to asymmetryTensorBatch(), which assembles the
     * cross sections its own formula needs.
     *
     * Static and public because the Fourier-moment leaves call it WITHOUT
     * deriving from this class: they derive from their PARTONS moment class
     * (DVCSAluMinusSin1Phi, ...), which already derives from
     * PARTONS::DVCSAluMinus, so inheriting this class too would give them two
     * DVCSAluMinus subobjects.
     * @return [N,M] tensor A_LU(phi), grad-connected to the NN CFF parameters.
     */
    static PARTONS::PhysicalType<torch::Tensor> aLUTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E,
            const torch::Tensor& phi);

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

    /**
     * Scalar entry point. On a torch process: computeTensor() under
     * NoGradGuard, detached. On any other process: the native PARTONS::DVCSAluMinus
     * this class derives from, so the leaf composes with any process module.
     */
    virtual PARTONS::PhysicalType<double> computeObservable(
            const PARTONS::DVCSObservableKinematic& kinematic,
            const PARTONS::List<PARTONS::GPDType>& gpdType) override;


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
    static PARTONS::PhysicalType<torch::Tensor> asymmetryTensorBatch(DVCSProcessModuleTorch& proc,
            const torch::Tensor& phi);
};

#endif /* DVCS_ALU_MINUS_TORCH_H */