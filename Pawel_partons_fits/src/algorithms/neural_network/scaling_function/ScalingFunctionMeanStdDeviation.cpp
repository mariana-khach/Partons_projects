/*
 * ScalingFunctionMeanStdDeviation.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/scaling_function/ScalingFunctionMeanStdDeviation.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <cmath>
#include <iterator>

#include "../../../../include/algorithms/neural_network/beans/ScalingFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int ScalingFunctionMeanStdDeviation::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerScalingFunction(
                ScalingFunctionType::MeanStdDeviation,
                new ScalingFunctionMeanStdDeviation());

ScalingFunctionMeanStdDeviation::ScalingFunctionMeanStdDeviation() :
        ScalingFunction("ScalingFunctionMeanStdDeviation") {
}

ScalingFunctionMeanStdDeviation::ScalingFunctionMeanStdDeviation(
        const ScalingFunctionMeanStdDeviation& other) :
        ScalingFunction(other) {
}

ScalingFunctionMeanStdDeviation::~ScalingFunctionMeanStdDeviation() {
}

ScalingFunctionMeanStdDeviation* ScalingFunctionMeanStdDeviation::clone() const {
    return new ScalingFunctionMeanStdDeviation(*this);
}

double ScalingFunctionMeanStdDeviation::evaluate(ScalingModeType::Type mode,
        double input, const std::pair<double, double>& parameters) const {

    switch (mode) {

    case ScalingModeType::Scale: {
        return (input - parameters.first) / parameters.second;
    }
        break;

    case ScalingModeType::Unscale: {
        return parameters.first + input * parameters.second;
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

double ScalingFunctionMeanStdDeviation::evaluateFirstDerivative(
        ScalingModeType::Type mode, double input,
        const std::pair<double, double>& parameters) const {

    switch (mode) {

    case ScalingModeType::Scale: {
        return 1. / parameters.second;
    }
        break;

    case ScalingModeType::Unscale: {
        return parameters.second;
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

std::pair<double, double> ScalingFunctionMeanStdDeviation::evaluateParameters(
        const std::vector<double>& input) const {

    std::vector<double>::const_iterator it;

    double mean = 0.;
    double sigma = 0.;

    for (it = input.begin(); it != input.end(); it++) {
        mean += (*it);
    }

    mean /= double(input.size());

    for (it = input.begin(); it != input.end(); it++) {
        sigma += pow(mean - *it, 2);
    }

    sigma = sqrt(sigma / double(input.size()));

    return std::make_pair(mean, sigma);
}
