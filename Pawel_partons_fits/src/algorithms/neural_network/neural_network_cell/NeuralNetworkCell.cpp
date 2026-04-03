/*
 * NeuralNetworkCell.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_cell/NeuralNetworkCell.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <iterator>

NeuralNetworkCell::NeuralNetworkCell() :
        BaseObject("NeuralNetworkCell") {
    m_type = NeuralNetworkCellType::UNDEFINED;
    m_output = 0.;
    m_neuronsIn.clear();
    m_neuronsOut.clear();
    m_properties.clear();
}

NeuralNetworkCell::NeuralNetworkCell(const std::string& name,
        NeuralNetworkCellType::Type type) :
        BaseObject(name) {
    m_type = type;
    m_output = 0.;
    m_neuronsIn.clear();
    m_neuronsOut.clear();
    m_properties.clear();
}

NeuralNetworkCell::NeuralNetworkCell(const NeuralNetworkCell& other) :
        BaseObject(other) {
    m_type = other.m_type;
    m_output = other.m_output;
    m_neuronsIn = other.m_neuronsIn;
    m_neuronsOut = other.m_neuronsOut;
    m_properties = other.m_properties;
}

NeuralNetworkCell::~NeuralNetworkCell() {
}

NeuralNetworkCell* NeuralNetworkCell::clone() const {
    return new NeuralNetworkCell(*this);
}

std::string NeuralNetworkCell::toString() const {

    ElemUtils::Formatter formatter;
    std::vector<NeuralNetworkNeuron*>::const_iterator it;

    formatter << "Cell: " << this << " [" << getClassName() << "]\n";
    formatter << "\t" << "input [" << m_neuronsIn.size() << "]: ";
    for (it = m_neuronsIn.begin(); it != m_neuronsIn.end(); it++)
        formatter << *it << " ";
    formatter << "\n";
    formatter << "\t" << "output [" << m_neuronsOut.size() << "]: ";
    for (it = m_neuronsOut.begin(); it != m_neuronsOut.end(); it++)
        formatter << *it << " ";
    formatter << "\n";

    return formatter.str();
}

NeuralNetworkCellType::Type NeuralNetworkCell::getType() const {
    return m_type;
}

void NeuralNetworkCell::evaluate() {
    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
}

double NeuralNetworkCell::evaluateDerivativeBackward(
        NeuralNetworkNeuron* const neuron) const {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
    return 0.;
}

const std::vector<NeuralNetworkNeuron*>& NeuralNetworkCell::getNeuronsIn() const {
    return m_neuronsIn;
}

void NeuralNetworkCell::addNeuronIn(NeuralNetworkNeuron* const neuron) {
    m_neuronsIn.push_back(neuron);
}

const std::vector<NeuralNetworkNeuron*>& NeuralNetworkCell::getNeuronsOut() const {
    return m_neuronsOut;
}

void NeuralNetworkCell::addNeuronOut(NeuralNetworkNeuron* const neuron) {
    m_neuronsOut.push_back(neuron);
}

double NeuralNetworkCell::getOutput() const {
    return m_output;
}

void NeuralNetworkCell::setOutput(double output) {
    m_output = output;
}

bool NeuralNetworkCell::checkProperty(
        NeuralNetworkCellPropertyType::Type property) const {

    std::vector<NeuralNetworkCellPropertyType::Type>::const_iterator it;

    for (it = m_properties.begin(); it != m_properties.end(); it++) {
        if ((*it) == property)
            return true;
    }

    return false;
}

void NeuralNetworkCell::checkConsistency() const {

    //number of input neurons
    if (int(checkProperty(NeuralNetworkCellPropertyType::OnlyOneInputNeuron))
            + int(
                    checkProperty(
                            NeuralNetworkCellPropertyType::ManyInputNeurons))
            + int(checkProperty(NeuralNetworkCellPropertyType::NoInputNeurons))
            != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "No or too many properties related to number of input neurons");
    }

    //number of output neurons
    if (int(checkProperty(NeuralNetworkCellPropertyType::OnlyOneOutputNeuron))
            + int(
                    checkProperty(
                            NeuralNetworkCellPropertyType::ManyOutputNeurons))
            + int(checkProperty(NeuralNetworkCellPropertyType::NoOutputNeurons))
            != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "No or too many properties related to number of output neurons");
    }

    //fixed weight of input neurons
    if (int(
            checkProperty(
                    NeuralNetworkCellPropertyType::RequairesFixedInputNeuronWeights))
            + int(
                    checkProperty(
                            NeuralNetworkCellPropertyType::DoNotRequairesFixedInputNeuronWeights))
            != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "No or too many properties related to weights of input neurons");
    }

    //fixed weight of output neurons
    if (int(
            checkProperty(
                    NeuralNetworkCellPropertyType::RequairesFixedOutputNeuronWeights))
            + int(
                    checkProperty(
                            NeuralNetworkCellPropertyType::DoNotRequairesFixedOutputNeuronWeights))
            != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "No or too many properties related to weights of output neurons");
    }

    //check properties
    std::vector<NeuralNetworkCellPropertyType::Type>::const_iterator it;

    for (it = m_properties.begin(); it != m_properties.end(); it++) {

        switch (*it) {

        case NeuralNetworkCellPropertyType::OnlyOneInputNeuron: {
            if (m_neuronsIn.size() != 1) {
                throw ElemUtils::CustomException(getClassName(), __func__,
                        "Property OnlyOneInputNeuron failed");
            }
        }

            break;

        case NeuralNetworkCellPropertyType::ManyInputNeurons: {
        }

            break;

        case NeuralNetworkCellPropertyType::NoInputNeurons: {
            if (m_neuronsIn.size() != 0) {
                throw ElemUtils::CustomException(getClassName(), __func__,
                        "Property NoInputNeurons failed");
            }
        }

            break;

        case NeuralNetworkCellPropertyType::OnlyOneOutputNeuron: {
            if (m_neuronsOut.size() != 1) {
                throw ElemUtils::CustomException(getClassName(), __func__,
                        "Property OnlyOneOutputNeuron failed");
            }
        }

            break;

        case NeuralNetworkCellPropertyType::ManyOutputNeurons: {
        }

            break;

        case NeuralNetworkCellPropertyType::NoOutputNeurons: {
            if (m_neuronsOut.size() != 0) {
                throw ElemUtils::CustomException(getClassName(), __func__,
                        "Property NoOutputNeurons failed");
            }
        }

            break;

        case NeuralNetworkCellPropertyType::RequairesFixedInputNeuronWeights: {

            std::vector<NeuralNetworkNeuron*>::const_iterator it;

            for (it = m_neuronsIn.begin(); it != m_neuronsIn.end(); it++) {

                if (!((*it)->isWeightFixed())) {
                    throw ElemUtils::CustomException(getClassName(), __func__,
                            "Property RequairesFixedInputNeuronWeights failed");
                }
            }
        }

            break;

        case NeuralNetworkCellPropertyType::DoNotRequairesFixedInputNeuronWeights: {
        }

            break;

        case NeuralNetworkCellPropertyType::RequairesFixedOutputNeuronWeights: {

            std::vector<NeuralNetworkNeuron*>::const_iterator it;

            for (it = m_neuronsOut.begin(); it != m_neuronsOut.end(); it++) {

                if (!((*it)->isWeightFixed())) {
                    throw ElemUtils::CustomException(getClassName(), __func__,
                            "Property RequairesFixedOutputNeuronWeights failed");
                }
            }
        }

            break;

        case NeuralNetworkCellPropertyType::DoNotRequairesFixedOutputNeuronWeights: {
        }

            break;

        default: {
        }

            break;
        }
    }
}

