/*
 * OutputCell.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_cell/OutputCell.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <vector>

#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellPropertyType.h"
#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"
#include "../../../../include/algorithms/neural_network/neural_network_neuron/NeuralNetworkNeuron.h"

OutputCell::OutputCell() :
        NeuralNetworkCell("OutputCell", NeuralNetworkCellType::OutputCell) {

    m_properties.push_back(NeuralNetworkCellPropertyType::OnlyOneInputNeuron);
    m_properties.push_back(NeuralNetworkCellPropertyType::NoOutputNeurons);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::RequairesFixedInputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::RequairesFixedOutputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequiresTraining);
}

OutputCell::OutputCell(const OutputCell& other) :
        NeuralNetworkCell(other) {
}

OutputCell::~OutputCell() {
}

OutputCell* OutputCell::clone() const {
    return new OutputCell(*this);
}

std::string OutputCell::toString() const {
    return NeuralNetworkCell::toString();
}

void OutputCell::checkConsistency() const {
    NeuralNetworkCell::checkConsistency();
}

void OutputCell::evaluate() {

    setOutput(m_neuronsIn.at(0)->getNeuralNetworkCellIn()->getOutput());
}

double OutputCell::evaluateDerivativeBackward(
        NeuralNetworkNeuron* const neuron) const {

    if (neuron->getNeuralNetworkCellOut() != this) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Neuron not connected to this cell");
    }

    return 1.;
}
