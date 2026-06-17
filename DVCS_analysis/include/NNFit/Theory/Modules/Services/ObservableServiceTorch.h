//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef OBSERVABLE_SERVICE_TORCH_H
#define OBSERVABLE_SERVICE_TORCH_H

#include <ElementaryUtils/logger/CustomException.h>
#include <torch/torch.h>

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
 * torch::Tensor, so only KinematicType is templated.
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
     * Mirrors the scalar computeSingleKinematic() but returns the observable's
     * tensor result instead of a scalar bean, so gradients propagate from the
     * returned tensor back to the NN parameters that parametrize the CFFs.
     *
     * @param kinematic   Observable kinematics.
     * @param pObservable Tensor observable to drive (base-typed for polymorphism).
     * @return 0-d torch::Tensor holding the observable value.
     */
    torch::Tensor computeSingleKinematicTorch(const KinematicType& kinematic,
            ObservableTorch<KinematicType>* pObservable) const {

        if (!pObservable) {
            throw ElemUtils::CustomException("ObservableServiceTorch", __func__,
                    "Null tensor observable passed to computeSingleKinematicTorch.");
        }

        return pObservable->computeTensor(kinematic);
    }
};

#endif /* OBSERVABLE_SERVICE_TORCH_H */