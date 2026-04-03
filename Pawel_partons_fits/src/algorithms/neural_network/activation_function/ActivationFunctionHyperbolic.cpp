/*
 * ActivationFunctionHyperbolic.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/activation_function/ActivationFunctionHyperbolic.h"

#include <cmath>

#include "../../../../include/algorithms/neural_network/beans/ActivationFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int ActivationFunctionHyperbolic::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerActivationFunction(
                ActivationFunctionType::Hyperbolic,
                new ActivationFunctionHyperbolic());

ActivationFunctionHyperbolic::ActivationFunctionHyperbolic() :
        ActivationFunction("ActivationFunctionHyperbolic") {
}

ActivationFunctionHyperbolic::ActivationFunctionHyperbolic(
        const ActivationFunctionHyperbolic& other) :
        ActivationFunction(other) {
}

ActivationFunctionHyperbolic::~ActivationFunctionHyperbolic() {
}

ActivationFunctionHyperbolic* ActivationFunctionHyperbolic::clone() const {
    return new ActivationFunctionHyperbolic(*this);
}

double ActivationFunctionHyperbolic::evaluate(double input) {
    return tanh(input);
}

double ActivationFunctionHyperbolic::evaluateFirstDerivative(double input) {
    return 1. - pow(tanh(input), 2);
}

double ActivationFunctionHyperbolic::evaluateSecondDerivative(double input) {
    return -2 * tanh(input) * (1. - pow(tanh(input), 2));
}
