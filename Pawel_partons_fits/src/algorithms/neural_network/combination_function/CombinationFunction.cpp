/*
 * CombinationFunction.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/combination_function/CombinationFunction.h"

#include <ElementaryUtils/logger/CustomException.h>

CombinationFunction::CombinationFunction() :
        BaseObject("CombinationFunction") {
}

CombinationFunction::CombinationFunction(const std::string& name) :
        BaseObject(name) {
}

CombinationFunction::CombinationFunction(const CombinationFunction& other) :
        BaseObject(other) {
}

CombinationFunction::~CombinationFunction() {
}

CombinationFunction* CombinationFunction::clone() const {
    return new CombinationFunction(*this);
}

double CombinationFunction::evaluate(
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

double CombinationFunction::evaluateFirstPartialDerivativeForCell(
        const NeuralNetworkNeuron* const neuron,
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

double CombinationFunction::evaluateFirstPartialDerivativeForNeuron(
        const NeuralNetworkNeuron* const neuron,
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

double CombinationFunction::evaluateFirstPartialDerivativeForBias(
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

