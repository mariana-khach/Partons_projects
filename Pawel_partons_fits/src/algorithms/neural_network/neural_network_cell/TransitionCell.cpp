/*
 * TransitionCell.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_cell/TransitionCell.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <vector>

#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellPropertyType.h"
#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"
#include "../../../../include/algorithms/neural_network/neural_network_neuron/NeuralNetworkNeuron.h"

TransitionCell::TransitionCell() :
        NeuralNetworkCell("TransitionCell",
                NeuralNetworkCellType::TransitionCell) {

    m_properties.push_back(NeuralNetworkCellPropertyType::OnlyOneInputNeuron);
    m_properties.push_back(NeuralNetworkCellPropertyType::ManyOutputNeurons);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::RequairesFixedInputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequairesFixedOutputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequiresTraining);
}

TransitionCell::TransitionCell(const TransitionCell& other) :
        NeuralNetworkCell(other) {
}

TransitionCell::~TransitionCell() {
}

TransitionCell* TransitionCell::clone() const {
    return new TransitionCell(*this);
}

std::string TransitionCell::toString() const {
    return NeuralNetworkCell::toString();
}

void TransitionCell::checkConsistency() const {
    NeuralNetworkCell::checkConsistency();
}

void TransitionCell::evaluate() {

    setOutput(m_neuronsIn.at(0)->getNeuralNetworkCellIn()->getOutput());
}

double TransitionCell::evaluateDerivativeBackward(
        NeuralNetworkNeuron* const neuron) const {

    if (neuron->getNeuralNetworkCellOut() != this) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Neuron not connected to this cell");
    }

    return 1.;
}
