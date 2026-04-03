/*
 * ScalingFunctionMinMax.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/scaling_function/ScalingFunctionMinMax.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <iterator>
#include <limits>

#include "../../../../include/algorithms/neural_network/beans/ScalingFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int ScalingFunctionMinMax::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerScalingFunction(
                ScalingFunctionType::MinMax, new ScalingFunctionMinMax());

ScalingFunctionMinMax::ScalingFunctionMinMax() :
        ScalingFunction("ScalingFunctionMinMax") {
}

ScalingFunctionMinMax::ScalingFunctionMinMax(const ScalingFunctionMinMax& other) :
        ScalingFunction(other) {
}

ScalingFunctionMinMax::~ScalingFunctionMinMax() {
}

ScalingFunctionMinMax* ScalingFunctionMinMax::clone() const {
    return new ScalingFunctionMinMax(*this);
}

double ScalingFunctionMinMax::evaluate(ScalingModeType::Type mode, double input,
        const std::pair<double, double>& parameters) const {

    switch (mode) {

    case ScalingModeType::Scale: {
        return 2 * (input - parameters.first)
                / (parameters.second - parameters.first) - 1.;
    }
        break;

    case ScalingModeType::Unscale: {
        return 0.5 * (input + 1.) * (parameters.second - parameters.first)
                + parameters.first;
    }
        break;

    default: {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Scaling mode neither scale nor unscale");
        return 0.;
    }

        break;
    }
}

double ScalingFunctionMinMax::evaluateFirstDerivative(
        ScalingModeType::Type mode, double input,
        const std::pair<double, double>& parameters) const {

    switch (mode) {

    case ScalingModeType::Scale: {
        return 2 / (parameters.second - parameters.first);
    }
        break;

    case ScalingModeType::Unscale: {
        return 0.5 * (parameters.second - parameters.first);
    }
        break;

    default: {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Scaling mode neither scale nor unscale");
        return 0.;
    }

        break;
    }
}

std::pair<double, double> ScalingFunctionMinMax::evaluateParameters(
        const std::vector<double>& input) const {

    std::vector<double>::const_iterator it;

    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();

    for (it = input.begin(); it != input.end(); it++) {
        if ((*it) < min)
            min = (*it);
        if ((*it) > max)
            max = (*it);
    }

    return std::make_pair(min, max);
}
