/*
 * ActivationFunctionSymetricThreshold.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/activation_function/ActivationFunctionSymetricThreshold.h"

#include <ElementaryUtils/logger/CustomException.h>

#include "../../../../include/algorithms/neural_network/beans/ActivationFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int ActivationFunctionSymetricThreshold::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerActivationFunction(
                ActivationFunctionType::SymetricThreshold,
                new ActivationFunctionSymetricThreshold());

ActivationFunctionSymetricThreshold::ActivationFunctionSymetricThreshold() :
        ActivationFunction("ActivationFunctionSymetricThreshold") {
}

ActivationFunctionSymetricThreshold::ActivationFunctionSymetricThreshold(
        const ActivationFunctionSymetricThreshold& other) :
        ActivationFunction(other) {
}

ActivationFunctionSymetricThreshold::~ActivationFunctionSymetricThreshold() {
}

ActivationFunctionSymetricThreshold* ActivationFunctionSymetricThreshold::clone() const {
    return new ActivationFunctionSymetricThreshold(*this);
}

double ActivationFunctionSymetricThreshold::evaluate(double input) {
    return (input < 0.) ? (-1.) : (1.);
}

double ActivationFunctionSymetricThreshold::evaluateFirstDerivative(
        double input) {
    return (input == 0.) ? (1.) : (0.);
}

double ActivationFunctionSymetricThreshold::evaluateSecondDerivative(
        double input) {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "No second derivative for this activation function");
    return 0.;
}
