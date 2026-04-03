/*
 * ActivationFunctionThreshold.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/activation_function/ActivationFunctionThreshold.h"

#include <ElementaryUtils/logger/CustomException.h>

#include "../../../../include/algorithms/neural_network/beans/ActivationFunctionType.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int ActivationFunctionThreshold::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerActivationFunction(
				ActivationFunctionType::Threshold, new ActivationFunctionThreshold());

ActivationFunctionThreshold::ActivationFunctionThreshold() :
		ActivationFunction("ActivationFunctionThreshold") {
}

ActivationFunctionThreshold::ActivationFunctionThreshold(
        const ActivationFunctionThreshold& other) :
        ActivationFunction(other) {
}

ActivationFunctionThreshold::~ActivationFunctionThreshold() {
}

ActivationFunctionThreshold* ActivationFunctionThreshold::clone() const {
    return new ActivationFunctionThreshold(*this);
}

double ActivationFunctionThreshold::evaluate(double input) {
	return (input < 0.) ? (0.) : (1.);
}

double ActivationFunctionThreshold::evaluateFirstDerivative(double input) {
	return (input == 0.) ? (1.) : (0.);
}

double ActivationFunctionThreshold::evaluateSecondDerivative(double input) {

    throw ElemUtils::CustomException(getClassName(), __func__, "No second derivative for this activation function");
	return 0.;
}
