/*
 * NeuralNetworkCellType.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"

NeuralNetworkCellType::NeuralNetworkCellType() :
        BaseObject("NeuralNetworkCellType") {
    m_type = NeuralNetworkCellType::UNDEFINED;
}

NeuralNetworkCellType::NeuralNetworkCellType(NeuralNetworkCellType::Type type) :
        BaseObject("NeuralNetworkCellType") {
    m_type = type;
}

NeuralNetworkCellType::NeuralNetworkCellType(
        const NeuralNetworkCellType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

NeuralNetworkCellType::~NeuralNetworkCellType() {
}

NeuralNetworkCellType* NeuralNetworkCellType::clone() const {
    return new NeuralNetworkCellType(*this);
}

std::string NeuralNetworkCellType::toString() const {

    switch (m_type) {

    case NeuralNetworkCellType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case NeuralNetworkCellType::InputCell: {
        return "InputCell";
    }

        break;

    case NeuralNetworkCellType::OutputCell: {
        return "OutputCell";
    }

        break;

    case NeuralNetworkCellType::ScalingCell: {
        return "ScalingCell";
    }

        break;

    case NeuralNetworkCellType::Perceptron: {
        return "Perceptron";
    }

        break;

    case NeuralNetworkCellType::TransitionCell: {
        return "TransitionCell";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}

