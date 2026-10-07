//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef OBSERVABLE_TORCH_H
#define OBSERVABLE_TORCH_H

#include <partons/beans/List.h>
#include <partons/utils/type/PhysicalType.h>
#include <torch/torch.h>

#include "NNFit/Theory/Beans/Obs/ObservableResultTorch.h"

/**
 * @class ObservableTorch
 *
 * @brief Channel-agnostic interface for a differentiable (libtorch) observable.
 *
 * Tensor counterpart of PARTONS' Observable<KinematicType, ResultType>. The
 * ResultType parameter collapses to ObservableResultTorch<KinematicType>
 * (there is exactly one tensor result shape), so only the KinematicType is
 * templated.
 *
 * Mirrors the scalar compute()/computeObservable() split via the non-virtual
 * interface (NVI) idiom:
 *   - computeTensor()      — public template method, the differentiable sibling
 *                            of Observable::compute(). The single entry point.
 *   - computeTensorImpl()  — protected pure-virtual hook, the sibling of
 *                            Observable::computeObservable(), supplied by leaves.
 *
 * Instantiated per channel exactly as PARTONS instantiates Observable<K,R>:
 *   using DVCSObservableTorch = ObservableTorch<PARTONS::DVCSObservableKinematic>;
 */
template <typename KinematicType>
class ObservableTorch {

public:

    virtual ~ObservableTorch() = default;

    /**
     * Differentiable observable value at the given kinematics (template method).
     * Delegates to the computeTensorImpl() hook; kept as a distinct layer so the
     * tensor chain matches the scalar compute()/computeObservable() split and so
     * shared pre/post work can be added here without touching the leaves.
     * @return Result bean carrying a 0-d unit-tagged tensor, grad-connected
     *         to the NN parameters.
     */
    ObservableResultTorch<KinematicType> computeTensor(
            const KinematicType& kinematic) {
        PARTONS::List<KinematicType> one;
        one.add(kinematic);
        return ObservableResultTorch<KinematicType>(computeTensorImpl(kinematic),
                one);
    }

    /**
     * Batched (N-point) sibling of computeTensor(): the differentiable value
     * at N kinematic points at once. Delegates to the computeTensorImplBatch()
     * hook. Channel-agnostic: takes PARTONS' own List<KinematicType>,
     * mirroring the scalar chain's computeManyKinematic bean-list convention.
     * @return Result bean carrying an [N] unit-tagged tensor, grad-connected
     *         to the NN parameters, alongside the kinematics it was evaluated
     *         at. One bean for the whole batch — the scalar chain returns
     *         List<ObservableResult>, one per point, because Result<K>'s
     *         kinematic is singular.
     */
    ObservableResultTorch<KinematicType> computeTensorBatch(
            const PARTONS::List<KinematicType>& kinematics) {
        return ObservableResultTorch<KinematicType>(
                computeTensorImplBatch(kinematics), kinematics);
    }

protected:

    /**
     * Physics hook implementing the observable, supplied by the concrete leaf —
     * the tensor sibling of Observable::computeObservable(). Returns a
     * unit-tagged value exactly as computeObservable() returns
     * PhysicalType<double>; the template method above wraps it in the bean.
     */
    virtual PARTONS::PhysicalType<torch::Tensor> computeTensorImpl(
            const KinematicType& kinematic) = 0;

    /**
     * Batched (N-point) sibling of computeTensorImpl(), supplied by the
     * concrete leaf.
     */
    virtual PARTONS::PhysicalType<torch::Tensor> computeTensorImplBatch(
            const PARTONS::List<KinematicType>& kinematics) = 0;
};

#endif /* OBSERVABLE_TORCH_H */