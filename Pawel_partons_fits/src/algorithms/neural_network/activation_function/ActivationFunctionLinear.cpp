/*
 * ActivationFunctionLinear.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/activation_function/ActivationFunctionLinear.h"

#include "../../../../include/algorithms/neural_network/beans/ActivationFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int ActivationFunctionLinear::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerActivationFunction(
                ActivationFunctionType::Linear, new ActivationFunctionLinear());

ActivationFunctionLinear::ActivationFunctionLinear() :
        ActivationFunction("ActivationFunctionLinear") {
}

ActivationFunctionLinear::ActivationFunctionLinear(
        const ActivationFunctionLinear& other) :
        ActivationFunction(other) {
}

ActivationFunctionLinear::~ActivationFunctionLinear() {
}

ActivationFunctionLinear* ActivationFunctionLinear::clone() const {
    return new ActivationFunctionLinear(*this);
}

double ActivationFunctionLinear::evaluate(double input) {
    return input;
}

double ActivationFunctionLinear::evaluateFirstDerivative(double input) {
    return 1.;
}

double ActivationFunctionLinear::evaluateSecondDerivative(double input) {
    return 0.;
}
