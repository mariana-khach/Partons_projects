/*
 * TrainingFunction.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/training_function/TrainingFunction.h"

#include <ElementaryUtils/logger/CustomException.h>

TrainingFunction::TrainingFunction() :
        BaseObject("TrainingFunction") {
}

TrainingFunction::TrainingFunction(const std::string& name) :
        BaseObject(name) {
}

TrainingFunction::TrainingFunction(const TrainingFunction& other) :
        BaseObject(other) {
}

TrainingFunction::~TrainingFunction() {
}

TrainingFunction* TrainingFunction::clone() const {
    return new TrainingFunction(*this);
}

double TrainingFunction::evaluate(const Data& output,
        const Data& outputReference) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

double TrainingFunction::partialDerivative(double output,
        double outputReference) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}
