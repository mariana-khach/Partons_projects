/*
 * ActivationFunction.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/activation_function/ActivationFunction.h"

#include <ElementaryUtils/logger/CustomException.h>

ActivationFunction::ActivationFunction() :
        BaseObject("ActivationFunction") {
}

ActivationFunction::ActivationFunction(const std::string& name) :
        BaseObject(name) {
}

ActivationFunction::ActivationFunction(const ActivationFunction& other) :
        BaseObject(other) {
}

ActivationFunction::~ActivationFunction() {
}

ActivationFunction* ActivationFunction::clone() const {
    return new ActivationFunction(*this);
}

double ActivationFunction::evaluate(double input) {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

double ActivationFunction::evaluateFirstDerivative(double input) {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

double ActivationFunction::evaluateSecondDerivative(double input) {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}
