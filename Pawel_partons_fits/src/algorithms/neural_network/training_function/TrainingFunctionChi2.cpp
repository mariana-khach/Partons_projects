/*
 * TrainingFunctionChi2.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/training_function/TrainingFunctionChi2.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "../../../../include/algorithms/neural_network/beans/TrainingFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int TrainingFunctionChi2::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerTrainingFunction(
                TrainingFunctionType::Chi2, new TrainingFunctionChi2());

TrainingFunctionChi2::TrainingFunctionChi2() :
        TrainingFunction("TrainingFunctionChi2") {
}

TrainingFunctionChi2::TrainingFunctionChi2(
        const TrainingFunctionChi2& other) :
        TrainingFunction(other) {
}

TrainingFunctionChi2::~TrainingFunctionChi2() {
}

TrainingFunctionChi2* TrainingFunctionChi2::clone() const {
    return new TrainingFunctionChi2(*this);
}

double TrainingFunctionChi2::evaluate(const Data& output,
        const Data& outputReference) const {

    if (output.getNVariables() != outputReference.getNVariables()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Size of input and output objects different");
    }

    if (output.getNPoints() != outputReference.getNPoints()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Number of points in output and output objects different");
    }

    double result = 0.;

    for (unsigned int i = 0; i < output.getNPoints(); i++) {

        std::map<unsigned int, std::vector<double> >::const_iterator it1, it2;
        double resultForOneDaton = 0.;

        for (it1 = output.getData().begin(), it2 =
                outputReference.getData().begin();
                it1 != output.getData().end()
                        || it2 != outputReference.getData().end();
                it1++, it2++) {
            resultForOneDaton += 0.5
                    * pow((it1->second).at(i) - (it2->second).at(i), 2);
        }

        result += resultForOneDaton;
    }

    return result;
}

double TrainingFunctionChi2::partialDerivative(double output,
        double outputReference) const {
    return output - outputReference;
}
