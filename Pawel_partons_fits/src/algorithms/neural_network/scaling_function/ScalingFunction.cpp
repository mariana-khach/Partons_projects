/*
 * ScalingFunction.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/scaling_function/ScalingFunction.h"

#include <ElementaryUtils/logger/CustomException.h>

ScalingFunction::ScalingFunction() :
        BaseObject("ScalingFunction") {
}

ScalingFunction::ScalingFunction(const std::string& name) :
        BaseObject(name) {
}

ScalingFunction::ScalingFunction(const ScalingFunction& other) :
        BaseObject(other) {
}

ScalingFunction::~ScalingFunction() {
}

ScalingFunction* ScalingFunction::clone() const {
    return new ScalingFunction(*this);
}

double ScalingFunction::evaluate(ScalingModeType::Type mode, double input,
        const std::pair<double, double>& parameters) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

double ScalingFunction::evaluateFirstDerivative(ScalingModeType::Type mode,
        double input, const std::pair<double, double>& parameters) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

std::pair<double, double> ScalingFunction::evaluateParameters(
        const std::vector<double>& input) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return std::pair<double, double>(0., 0.);
}
