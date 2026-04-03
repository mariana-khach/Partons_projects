/*
 * NeuralNetworkLayer.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_layer/NeuralNetworkLayer.h"

#include <ElementaryUtils/string_utils/Formatter.h>
#include <iterator>

NeuralNetworkLayer::NeuralNetworkLayer() :
        BaseObject("NeuralNetworkLayer") {
    m_cells.clear();
}

NeuralNetworkLayer::NeuralNetworkLayer(const NeuralNetworkLayer& other) :
        BaseObject(other) {
    m_cells = other.m_cells;
}

NeuralNetworkLayer::~NeuralNetworkLayer() {
}

NeuralNetworkLayer* NeuralNetworkLayer::clone() const {
    return new NeuralNetworkLayer(*this);
}

std::string NeuralNetworkLayer::toString() const {

    ElemUtils::Formatter formatter;
    std::vector<NeuralNetworkCell*>::const_iterator it;

    formatter << "Layer: " << this << "\n";
    formatter << "\t" << "content [" << m_cells.size() << "]: ";
    for (it = m_cells.begin(); it != m_cells.end(); it++)
        formatter << *it << " ";
    formatter << "\n";

    return formatter.str();
}

const std::vector<NeuralNetworkCell*>& NeuralNetworkLayer::getCells() const {
    return m_cells;
}

void NeuralNetworkLayer::addCell(NeuralNetworkCell* const cell) {
    m_cells.push_back(cell);
}

void NeuralNetworkLayer::evaluate() const {

    std::vector<NeuralNetworkCell*>::const_iterator it;

    for (it = m_cells.begin(); it != m_cells.end(); it++) {
        (*it)->evaluate();
    }
}
