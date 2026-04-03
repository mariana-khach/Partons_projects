/*
 * NeuralNetworkCellPropertyType.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellPropertyType.h"

NeuralNetworkCellPropertyType::NeuralNetworkCellPropertyType() :
        BaseObject("NeuralNetworkCellPropertyType") {
    m_type = NeuralNetworkCellPropertyType::UNDEFINED;
}

NeuralNetworkCellPropertyType::NeuralNetworkCellPropertyType(
        NeuralNetworkCellPropertyType::Type type) :
        BaseObject("NeuralNetworkCellPropertyType") {
    m_type = type;
}

NeuralNetworkCellPropertyType::NeuralNetworkCellPropertyType(
        const NeuralNetworkCellPropertyType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

NeuralNetworkCellPropertyType::~NeuralNetworkCellPropertyType() {
}

NeuralNetworkCellPropertyType* NeuralNetworkCellPropertyType::clone() const {
    return new NeuralNetworkCellPropertyType(*this);
}

std::string NeuralNetworkCellPropertyType::toString() const {

    switch (m_type) {

    case NeuralNetworkCellPropertyType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case NeuralNetworkCellPropertyType::OnlyOneInputNeuron: {
        return "OnlyOneInputNeuron";
    }

        break;

    case NeuralNetworkCellPropertyType::ManyInputNeurons: {
        return "ManyInputNeurons";
    }

        break;

    case NeuralNetworkCellPropertyType::NoInputNeurons: {
        return "NoInputNeurons";
    }

        break;

    case NeuralNetworkCellPropertyType::OnlyOneOutputNeuron: {
        return "OnlyOneOutputNeuron";
    }

        break;

    case NeuralNetworkCellPropertyType::ManyOutputNeurons: {
        return "ManyOutputNeurons";
    }

        break;

    case NeuralNetworkCellPropertyType::NoOutputNeurons: {
        return "NoOutputNeurons";
    }

        break;

    case NeuralNetworkCellPropertyType::RequairesFixedInputNeuronWeights: {
        return "RequairesFixedInputNeuronWeights";
    }

        break;

    case NeuralNetworkCellPropertyType::DoNotRequairesFixedInputNeuronWeights: {
        return "DoNotRequairesFixedInputNeuronWeights";
    }

        break;

    case NeuralNetworkCellPropertyType::RequairesFixedOutputNeuronWeights: {
        return "RequairesFixedOutputNeuronWeights";
    }

        break;
    case NeuralNetworkCellPropertyType::DoNotRequairesFixedOutputNeuronWeights: {
        return "DoNotRequairesFixedutputNeuronWeights";
    }

        break;

    case NeuralNetworkCellPropertyType::RequiresTraining: {
        return "RequiresTraining";
    }

        break;

    case NeuralNetworkCellPropertyType::DoNotRequiresTraining: {
        return "DoNotRequiresTraining";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}

