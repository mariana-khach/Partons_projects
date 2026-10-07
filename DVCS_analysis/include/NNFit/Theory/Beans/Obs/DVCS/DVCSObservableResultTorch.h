//
// Created by Mariana Khachatryan on 9/23/26.
//

#ifndef DVCS_OBSERVABLE_RESULT_TORCH_H
#define DVCS_OBSERVABLE_RESULT_TORCH_H

#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>

#include "NNFit/Theory/Beans/Obs/ObservableResultTorch.h"

/**
 * DVCS instantiation of the channel-agnostic tensor result bean — the tensor
 * twin of PARTONS::DVCSObservableResult. A plain alias, exactly as
 * DVCSObservableTorch is for ObservableTorch: the bean carries no
 * channel-specific API, only the value, its unit and the kinematics.
 *
 * Note the scalar side needs a real class here (DVCSObservableResult derives
 * from ObservableResult<DVCSObservableKinematic>) because PARTONS' factory and
 * database layers dispatch on the concrete type. Nothing on the tensor path
 * does, so an alias is enough.
 */
using DVCSObservableResultTorch =
        ObservableResultTorch<PARTONS::DVCSObservableKinematic>;

#endif /* DVCS_OBSERVABLE_RESULT_TORCH_H */