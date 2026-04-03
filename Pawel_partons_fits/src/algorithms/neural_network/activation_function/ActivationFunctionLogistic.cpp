/*
 * ActivationFunctionLogistic.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/activation_function/ActivationFunctionLogistic.h"

#include <cmath>

#include "../../../../include/algorithms/neural_network/beans/ActivationFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int ActivationFunctionLogistic::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerActivationFunction(
                ActivationFunctionType::Logistic,
                new ActivationFunctionLogistic());

ActivationFunctionLogistic::ActivationFunctionLogistic() :
        ActivationFunction("ActivationFunctionLogistic") {
}

ActivationFunctionLogistic::ActivationFunctionLogistic(
        const ActivationFunctionLogistic& other) :
        ActivationFunction(other) {
}

ActivationFunctionLogistic::~ActivationFunctionLogistic() {
}

ActivationFunctionLogistic* ActivationFunctionLogistic::clone() const {
    return new ActivationFunctionLogistic(*this);
}

double ActivationFunctionLogistic::evaluate(double input) {
    return 1. / (1. + exp(-1 * input));
}

double ActivationFunctionLogistic::evaluateFirstDerivative(double input) {

    double mInput = -1 * input;

    return exp(mInput) * pow(1. + exp(mInput), -2);
}

double ActivationFunctionLogistic::evaluateSecondDerivative(double input) {

    double mInput = -1 * input;

    return -1 * exp(mInput) * pow(1. + exp(mInput), -2)
            + 2 * pow(exp(mInput), 2) * pow(1. + exp(mInput), -3);
}
