/*
 * InputCell.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_cell/InputCell.h"

#include <vector>

#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellPropertyType.h"
#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"

InputCell::InputCell() :
        NeuralNetworkCell("InputCell", NeuralNetworkCellType::InputCell) {

    m_input = 0.;

    m_properties.push_back(NeuralNetworkCellPropertyType::NoInputNeurons);
    m_properties.push_back(NeuralNetworkCellPropertyType::ManyOutputNeurons);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::RequairesFixedInputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequairesFixedOutputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequiresTraining);
}

InputCell::InputCell(const InputCell& other) :
        NeuralNetworkCell(other) {

    m_input = other.m_input;
}

InputCell::~InputCell() {
}

InputCell* InputCell::clone() const {
    return new InputCell(*this);
}

std::string InputCell::toString() const{
    return NeuralNetworkCell::toString();
}

void InputCell::checkConsistency() const {
    NeuralNetworkCell::checkConsistency();
}

void InputCell::evaluate() {
    setOutput(m_input);
}

double InputCell::evaluateDerivativeBackward(
        NeuralNetworkNeuron* const neuron) const {

    //throw ElemUtils::CustomException(__func__, "This type of cell can not have input neurons");
    return 0.;
}

void InputCell::setInput(double input) {
    m_input = input;
}

double InputCell::getInput() const {
    return m_input;
}
