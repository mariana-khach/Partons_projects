//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef OBSERVABLE_SERVICE_TORCH_H
#define OBSERVABLE_SERVICE_TORCH_H

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/beans/List.h>
#include <torch/torch.h>

#include "NNFit/Theory/Beans/Obs/ObservableResultTorch.h"
#include "NNFit/Theory/Modules/Obs/ObservableTorch.h"

/**
 * @class ObservableServiceTorch
 *
 * @brief Channel-agnostic differentiable driver, mixed into a PARTONS service.
 *
 * Tensor counterpart of ObservableService<KinematicType, ResultType>::
 * computeSingleKinematic(). It is a thin generic dispatch — the differentiable
 * twin of `pObservable->compute(...)` — taking a base ObservableTorch pointer so
 * it drives any tensor observable of the channel polymorphically (like the
 * scalar service taking Observable<K,R>*). The result type collapses to
 * ObservableResultTorch<K>, so only KinematicType is templated.
 *
 * Intended as a mixin alongside the channel PARTONS service, e.g.
 *   class DVCSObservableServiceTorch
 *       : public PARTONS::DVCSObservableService,
 *         public ObservableServiceTorch<PARTONS::DVCSObservableKinematic> {};
 */
template <typename KinematicType>
class ObservableServiceTorch {

public:

    virtual ~ObservableServiceTorch() = default;

    /**
     * Differentiable single-kinematic computation.
     *
     * Mirrors the scalar computeSingleKinematic(), returning a result bean as
     * it does — but one carrying a live torch::Tensor rather than a detached
     * double, so gradients propagate from the returned value back to the NN
     * parameters that parametrize the CFFs.
     *
     * @param kinematic   Observable kinematics.
     * @param pObservable Tensor observable to drive (base-typed for polymorphism).
     * @return Result bean holding a 0-d unit-tagged tensor.
     */
    ObservableResultTorch<KinematicType> computeSingleKinematicTorch(
            const KinematicType& kinematic,
            ObservableTorch<KinematicType>* pObservable) const {

        if (!pObservable) {
            throw ElemUtils::CustomException("ObservableServiceTorch", __func__,
                    "Null tensor observable passed to computeSingleKinematicTorch.");
        }

        return pObservable->computeTensor(kinematic);
    }

    /**
     * Differentiable many-kinematic (batched) computation.
     *
     * Mirrors computeSingleKinematicTorch() but drives the observable's batched
     * hook (ObservableTorch<K>::computeTensorBatch(), channel-generic
     * List<K>) instead of the single-point one, so gradients still propagate
     * from the returned [N] tensor back to the NN parameters.
     *
     * @param kinematics  List of N observable kinematics.
     * @param pObservable Tensor observable to drive (base-typed for polymorphism).
     * @return Result bean holding an [N] unit-tagged tensor plus the
     *         kinematics it was evaluated at.
     */
    ObservableResultTorch<KinematicType> computeManyKinematicTorch(
            const PARTONS::List<KinematicType>& kinematics,
            ObservableTorch<KinematicType>* pObservable) const {

        if (!pObservable) {
            throw ElemUtils::CustomException("ObservableServiceTorch", __func__,
                    "Null tensor observable passed to computeManyKinematicTorch.");
        }

        return pObservable->computeTensorBatch(kinematics);
    }
};

#endif /* OBSERVABLE_SERVICE_TORCH_H */