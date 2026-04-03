/*
 * CombinationFunctionScalarProduct.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/combination_function/CombinationFunctionScalarProduct.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <iterator>

#include "../../../../include/algorithms/neural_network/beans/CombinationFunctionType.h"
#include "../../../../include/algorithms/neural_network/neural_network_cell/NeuralNetworkCell.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

const unsigned int CombinationFunctionScalarProduct::classId =
        NeuralNetworkTypeRegistry::getInstance()->registerCombinationFunction(
                CombinationFunctionType::ScalarProduct,
                new CombinationFunctionScalarProduct());

CombinationFunctionScalarProduct::CombinationFunctionScalarProduct() :
        CombinationFunction("CombinationFunctionScalarProduct") {
}

CombinationFunctionScalarProduct::CombinationFunctionScalarProduct(
        const CombinationFunctionScalarProduct& other) :
        CombinationFunction(other) {
}

CombinationFunctionScalarProduct::~CombinationFunctionScalarProduct() {
}

CombinationFunctionScalarProduct* CombinationFunctionScalarProduct::clone() const {
    return new CombinationFunctionScalarProduct(*this);
}

double CombinationFunctionScalarProduct::evaluate(
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    double result = bias;
    std::vector<NeuralNetworkNeuron*>::const_iterator it;

    for (it = neurons.begin(); it != neurons.end(); it++) {
        result += (*it)->getNeuralNetworkCellIn()->getOutput()
                * (*it)->getWeight();
    }

    return result;
}

double CombinationFunctionScalarProduct::evaluateFirstPartialDerivativeForCell(
        const NeuralNetworkNeuron* const neuron,
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    std::vector<NeuralNetworkNeuron*>::const_iterator it;

    for (it = neurons.begin(); it != neurons.end(); it++) {
        if (neuron == *it)
            return (*it)->getWeight();
    }

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Neuron not found");

    return 0.;
}

double CombinationFunctionScalarProduct::evaluateFirstPartialDerivativeForNeuron(
        const NeuralNetworkNeuron* const neuron,
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    std::vector<NeuralNetworkNeuron*>::const_iterator it;

    for (it = neurons.begin(); it != neurons.end(); it++) {
        if (neuron == *it)
            return (*it)->getNeuralNetworkCellIn()->getOutput();
    }

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Neuron not found");

    return 0.;
}

double CombinationFunctionScalarProduct::evaluateFirstPartialDerivativeForBias(
        const std::vector<NeuralNetworkNeuron*>& neurons, double bias) const {

    return bias;
}
