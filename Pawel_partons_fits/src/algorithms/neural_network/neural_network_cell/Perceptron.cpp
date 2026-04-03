/*
 * Perceptron.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_cell/Perceptron.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <vector>

#include "../../../../include/algorithms/neural_network/activation_function/ActivationFunction.h"
#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellPropertyType.h"
#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"
#include "../../../../include/algorithms/neural_network/combination_function/CombinationFunction.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"
#include "../../../../include/algorithms/random_generator/RandomGenerator.h"

Perceptron::Perceptron() :
        NeuralNetworkCell("Perceptron", NeuralNetworkCellType::Perceptron) {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Constructor not supporter");

    m_pRandomGenerator = 0;

    m_bias = 0.;
    m_fixedBias = false;
    m_activationFunctionType = ActivationFunctionType::UNDEFINED;
    m_combinationFunctionType = CombinationFunctionType::UNDEFINED;
    m_pActivationFunction = 0;
    m_pCombinationFunction = 0;
}

Perceptron::Perceptron(ActivationFunctionType::Type activationFunctionType,
        CombinationFunctionType::Type combinationFunctionType,
        InitializationType::Type initializationType) :
        NeuralNetworkCell("Perceptron", NeuralNetworkCellType::Perceptron) {

    m_pRandomGenerator = RandomGenerator::getInstance();

    switch (initializationType) {

    case InitializationType::Static: {
        m_bias = 0.;
    }

        break;

    case InitializationType::Random: {
        m_bias = m_pRandomGenerator->diceFlat(-1., 1.);
    }

        break;

    default: {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "InitializationType not supported");
    }

        break;
    }

    m_fixedBias = false;
    m_activationFunctionType = activationFunctionType;
    m_combinationFunctionType = combinationFunctionType;

    m_pActivationFunction =
            NeuralNetworkTypeRegistry::getInstance()->getActivationFunction(
                    m_activationFunctionType);

    m_pCombinationFunction =
            NeuralNetworkTypeRegistry::getInstance()->getCombinationFunction(
                    m_combinationFunctionType);

    m_properties.push_back(NeuralNetworkCellPropertyType::ManyInputNeurons);
    m_properties.push_back(NeuralNetworkCellPropertyType::ManyOutputNeurons);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequairesFixedInputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequairesFixedOutputNeuronWeights);
    m_properties.push_back(NeuralNetworkCellPropertyType::RequiresTraining);
}

Perceptron::Perceptron(const Perceptron& other) :
        NeuralNetworkCell(other) {

    m_pRandomGenerator = other.m_pRandomGenerator;

    m_bias = other.m_bias;
    m_fixedBias = other.m_fixedBias;
    m_activationFunctionType = other.m_activationFunctionType;
    m_combinationFunctionType = other.m_combinationFunctionType;

    if (other.m_pActivationFunction) {
        m_pActivationFunction = other.m_pActivationFunction->clone();
    }

    if (other.m_pCombinationFunction) {
        m_pCombinationFunction = other.m_pCombinationFunction->clone();
    }
}

Perceptron::~Perceptron() {

    if (m_pActivationFunction) {
        delete m_pActivationFunction;
        m_pActivationFunction = 0;
    }

    if (m_pCombinationFunction) {
        delete m_pCombinationFunction;
        m_pCombinationFunction = 0;
    }
}

Perceptron* Perceptron::clone() const {
    return new Perceptron(*this);
}

std::string Perceptron::toString() const {

    ElemUtils::Formatter formatter;

    formatter << NeuralNetworkCell::toString();
    formatter << "\t" << "activationFunctionType: "
            << ActivationFunctionType(m_activationFunctionType).toString()
            << "\n";
    formatter << "\t" << "combinationFunctionType: "
            << CombinationFunctionType(m_combinationFunctionType).toString()
            << "\n";
    formatter << "\t" << "bias [fixed]: " << m_bias << " [" << m_fixedBias
            << "]" << "\n";

    return formatter.str();
}

double Perceptron::getBias() const {
    return m_bias;
}

void Perceptron::setBias(double bias) {
    m_bias = bias;
}

void Perceptron::fixBias() {
    m_fixedBias = true;
}

void Perceptron::releaseBias() {
    m_fixedBias = false;
}

bool Perceptron::isBiasFixed() const {
    return m_fixedBias;
}

ActivationFunctionType::Type Perceptron::getActivationFunctionType() const {
    return m_activationFunctionType;
}

CombinationFunctionType::Type Perceptron::getCombinationFunctionType() const {
    return m_combinationFunctionType;
}

void Perceptron::checkConsistency() const {
    NeuralNetworkCell::checkConsistency();
}

void Perceptron::evaluate() {

    setOutput(
            m_pActivationFunction->evaluate(
                    m_pCombinationFunction->evaluate(m_neuronsIn, m_bias)));
}

double Perceptron::evaluateDerivativeBackward(
        NeuralNetworkNeuron* const neuron) const {

//	if (neuron->getNeuralNetworkCellOut() != this) {
//		throw ElemUtils::CustomException(__func__, "Neuron not connected to this cell");
//	}

    double firstDerivativeOfActivationFunction =
            m_pActivationFunction->evaluateFirstDerivative(
                    m_pCombinationFunction->evaluate(m_neuronsIn, m_bias));

    double firstDerivativeOfCombinationFunction =
            m_pCombinationFunction->evaluateFirstPartialDerivativeForCell(
                    neuron, m_neuronsIn, m_bias);

    return firstDerivativeOfActivationFunction
            * firstDerivativeOfCombinationFunction;
}

double Perceptron::evaluateFirstDerivatieOfActivationFunction() const {
    return m_pActivationFunction->evaluateFirstDerivative(
            m_pCombinationFunction->evaluate(m_neuronsIn, m_bias));
}

double Perceptron::evaluateFirstPartialDerivatieOfCombinationFunctionForNeuron(
        const NeuralNetworkNeuron* const neuron) const {
    return m_pCombinationFunction->evaluateFirstPartialDerivativeForNeuron(
            neuron, m_neuronsIn, m_bias);
}

double Perceptron::evaluateFirstPartialDerivatieOfCombinationFunctionForBias() const {
    return m_pCombinationFunction->evaluateFirstPartialDerivativeForBias(
            m_neuronsIn, m_bias);
}
