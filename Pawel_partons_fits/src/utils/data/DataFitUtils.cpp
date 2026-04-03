/*
 * DataUtils.cpp
 *
 *  Created on: Nov 17, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#include "../../../include/utils/data/DataFitUtils.h"

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <stddef.h>

#include "../../../include/beans/observable_data_fit/ObservableDataFit.h"

using namespace PARTONS;

namespace DataFitUtils {

DVCSObservableKinematic getAverageKinematic(const List<ObservableDataFit>& data) {

    //variables
    double meanXb = 0.;
    double meanT = 0.;
    double meanQ2 = 0.;
    double meanE = 0.;
    double meanPhi = 0.;

    //loop over points
    for (size_t i = 0; i < data.size(); i++) {

        meanXb += data.get(i).getObservableKinematic().getXB();
        meanT += data.get(i).getObservableKinematic().getT();
        meanQ2 += data.get(i).getObservableKinematic().getQ2();
        meanE += data.get(i).getObservableKinematic().getE();
        meanPhi += data.get(i).getObservableKinematic().getPhi().getValue();
    }

    //finish
    meanXb /= double(data.size());
    meanT /= double(data.size());
    meanQ2 /= double(data.size());
    meanE /= double(data.size());
    meanPhi /= double(data.size());

    //return
    return DVCSObservableKinematic(meanXb, meanT, meanQ2, meanE, meanPhi);
}

DVCSObservableKinematic getMinimumKinematic(const List<ObservableDataFit>& data) {

    //variables
    double minXb;
    double minT;
    double minQ2;
    double minE;
    double minPhi;

    //loop over points
    for (size_t i = 0; i < data.size(); i++) {

        if (i == 0) {

            minXb = data.get(i).getObservableKinematic().getXB();
            minT = data.get(i).getObservableKinematic().getT();
            minQ2 = data.get(i).getObservableKinematic().getQ2();
            minE = data.get(i).getObservableKinematic().getE();
            minPhi = data.get(i).getObservableKinematic().getPhi().getValue();

        } else {

            if (data.get(i).getObservableKinematic().getXB() < minXb)
                minXb = data.get(i).getObservableKinematic().getXB();
            if (data.get(i).getObservableKinematic().getT() < minT)
                minT = data.get(i).getObservableKinematic().getT();
            if (data.get(i).getObservableKinematic().getQ2() < minQ2)
                minQ2 = data.get(i).getObservableKinematic().getQ2();
            if (data.get(i).getObservableKinematic().getE() < minE)
                minE = data.get(i).getObservableKinematic().getE();
            if (data.get(i).getObservableKinematic().getPhi().getValue() < minPhi)
                minPhi = data.get(i).getObservableKinematic().getPhi().getValue();
        }
    }

    //return
    return DVCSObservableKinematic(minXb, minT, minQ2, minE, minPhi);
}

DVCSObservableKinematic getMaximumKinematic(const List<ObservableDataFit>& data) {

    //variables
    double maxXb;
    double maxT;
    double maxQ2;
    double maxE;
    double maxPhi;

    //loop over points
    for (size_t i = 0; i < data.size(); i++) {

        if (i == 0) {

            maxXb = data.get(i).getObservableKinematic().getXB();
            maxT = data.get(i).getObservableKinematic().getT();
            maxQ2 = data.get(i).getObservableKinematic().getQ2();
            maxE = data.get(i).getObservableKinematic().getE();
            maxPhi = data.get(i).getObservableKinematic().getPhi().getValue();

        } else {

            if (data.get(i).getObservableKinematic().getXB() > maxXb)
                maxXb = data.get(i).getObservableKinematic().getXB();
            if (data.get(i).getObservableKinematic().getT() > maxT)
                maxT = data.get(i).getObservableKinematic().getT();
            if (data.get(i).getObservableKinematic().getQ2() > maxQ2)
                maxQ2 = data.get(i).getObservableKinematic().getQ2();
            if (data.get(i).getObservableKinematic().getE() > maxE)
                maxE = data.get(i).getObservableKinematic().getE();
            if (data.get(i).getObservableKinematic().getPhi().getValue() > maxPhi)
                maxPhi = data.get(i).getObservableKinematic().getPhi().getValue();
        }
    }

    //return
    return DVCSObservableKinematic(maxXb, maxT, maxQ2, maxE, maxPhi);
}

}
