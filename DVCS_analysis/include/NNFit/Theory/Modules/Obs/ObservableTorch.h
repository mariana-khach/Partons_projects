//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef OBSERVABLE_TORCH_H
#define OBSERVABLE_TORCH_H

#include <torch/torch.h>

/**
 * @class ObservableTorch
 *
 * @brief Channel-agnostic interface for a differentiable (libtorch) observable.
 *
 * Tensor counterpart of PARTONS' Observable<KinematicType, ResultType>. The
 * ResultType template parameter collapses to torch::Tensor (every tensor
 * observable returns a 0-d tensor), so only the KinematicType is templated.
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
     * @return 0-d torch::Tensor, grad-connected to the NN parameters.
     */
    torch::Tensor computeTensor(const KinematicType& kinematic) {
        return computeTensorImpl(kinematic);
    }

protected:

    /**
     * Physics hook implementing the observable, supplied by the concrete leaf —
     * the tensor sibling of Observable::computeObservable().
     */
    virtual torch::Tensor computeTensorImpl(const KinematicType& kinematic) = 0;
};

#endif /* OBSERVABLE_TORCH_H */