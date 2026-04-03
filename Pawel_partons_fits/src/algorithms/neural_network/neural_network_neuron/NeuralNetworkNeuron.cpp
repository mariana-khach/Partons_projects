/*
 * NeuralNetworkNeuron.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_neuron/NeuralNetworkNeuron.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>

#include "../../../../include/algorithms/neural_network/neural_network_cell/NeuralNetworkCell.h"
#include "../../../../include/algorithms/random_generator/RandomGenerator.h"

class Perceptron;

NeuralNetworkNeuron::NeuralNetworkNeuron() :
        BaseObject("NeuralNetworkNeuron") {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Constructor not supporter");

    m_pRandomGenerator = 0;

    m_weight = 0.;
    m_fixedWeight = false;
    m_neuralNetworkCellIn = 0;
    m_neuralNetwokCellOut = 0;
}

NeuralNetworkNeuron::NeuralNetworkNeuron(
        NeuralNetworkCell* const NeuralNetworkCellIn,
        NeuralNetworkCell* const NeuralNetworkCellOut,
        InitializationType::Type initializationType) :
        BaseObject("Neuron") {

    m_pRandomGenerator = RandomGenerator::getInstance();

    switch (initializationType) {

    case InitializationType::Static: {
        m_weight = 1.;
    }

        break;

    case InitializationType::Random: {
        m_weight = m_pRandomGenerator->diceFlat(-1., 1.);
    }

        break;

    default: {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "InitializationType not supported");
    }

        break;
    }

    m_fixedWeight = false;
    m_neuralNetworkCellIn = NeuralNetworkCellIn;
    m_neuralNetwokCellOut = NeuralNetworkCellOut;
}

NeuralNetworkNeuron::NeuralNetworkNeuron(const NeuralNetworkNeuron& other) :
        BaseObject(other) {

    m_pRandomGenerator = other.m_pRandomGenerator;
    m_weight = other.m_weight;
    m_fixedWeight = other.m_fixedWeight;
    m_neuralNetworkCellIn = other.m_neuralNetworkCellIn;
    m_neuralNetwokCellOut = other.m_neuralNetwokCellOut;
}

NeuralNetworkNeuron::~NeuralNetworkNeuron() {
}

NeuralNetworkNeuron* NeuralNetworkNeuron::clone() const {
    return new NeuralNetworkNeuron(*this);
}

std::string NeuralNetworkNeuron::toString() const {

    ElemUtils::Formatter formatter;

    formatter << "Neuron: " << this << "\n";
    formatter << "\t" << "weight [fixed]: " << m_weight << " [" << m_fixedWeight
            << "]" << "\n";
    formatter << "\t" << "cell in: " << m_neuralNetworkCellIn << "\n";
    formatter << "\t" << "cell out: " << m_neuralNetwokCellOut << "\n";

    return formatter.str();
}

double NeuralNetworkNeuron::getWeight() const {
    return m_weight;
}

void NeuralNetworkNeuron::setWeight(double weight) {
    m_weight = weight;
}

bool NeuralNetworkNeuron::isWeightFixed() const {
    return m_fixedWeight;
}

void NeuralNetworkNeuron::fixWeight() {
    m_fixedWeight = true;
}

void NeuralNetworkNeuron::releaseWeight() {
    m_fixedWeight = false;
}

NeuralNetworkCell* NeuralNetworkNeuron::getNeuralNetworkCellIn() const {
    return m_neuralNetworkCellIn;
}

NeuralNetworkCell* NeuralNetworkNeuron::getNeuralNetworkCellOut() const {
    return m_neuralNetwokCellOut;
}

