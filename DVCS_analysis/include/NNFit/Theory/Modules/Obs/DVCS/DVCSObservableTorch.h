//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef DVCS_OBSERVABLE_TORCH_H
#define DVCS_OBSERVABLE_TORCH_H

#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>

#include "NNFit/Theory/Modules/Obs/ObservableTorch.h"

/**
 * DVCS instantiation of the channel-agnostic tensor observable interface —
 * the tensor twin of PARTONS::DVCSObservable. computeTensor() is the whole
 * contract, so this is a plain alias (no DVCS-specific tensor API to add).
 */
using DVCSObservableTorch = ObservableTorch<PARTONS::DVCSObservableKinematic>;

#endif /* DVCS_OBSERVABLE_TORCH_H */