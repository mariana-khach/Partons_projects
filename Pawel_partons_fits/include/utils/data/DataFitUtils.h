/*
 * DataUtils.h
 *
 *  Created on: Nov 17, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef DATAFITUTILS_H_
#define DATAFITUTILS_H_

#include <partons/beans/List.h>

#include "../../beans/observable_data_fit/ObservableDataFit.h"

namespace DataFitUtils {

/**
 * Calculate average kinematics for given data list
 * @param data List of data points
 */
PARTONS::DVCSObservableKinematic getAverageKinematic(
        const PARTONS::List<ObservableDataFit>& data);

/**
 * Get minimum probed kinematics for given data list
 * @param data List of data points
 */
PARTONS::DVCSObservableKinematic getMinimumKinematic(
        const PARTONS::List<ObservableDataFit>& data);

/**
 * Get maximum probed kinematics for given data list
 * @param data List of data points
 */
PARTONS::DVCSObservableKinematic getMaximumKinematic(
        const PARTONS::List<ObservableDataFit>& data);

}

#endif /* DATAUTILS_H_ */
