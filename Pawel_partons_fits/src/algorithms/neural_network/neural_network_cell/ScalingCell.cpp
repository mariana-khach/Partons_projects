/*
 * ScalingCell.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network_cell/ScalingCell.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>

#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellPropertyType.h"
#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"
#include "../../../../include/algorithms/neural_network/neural_network_neuron/NeuralNetworkNeuron.h"
#include "../../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"
#include "../../../../include/algorithms/neural_network/scaling_function/ScalingFunction.h"

ScalingCell::ScalingCell() :
        NeuralNetworkCell("ScalingCell", NeuralNetworkCellType::ScalingCell) {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Constructor not supporter");

    m_scalingModeType = ScalingModeType::UNDEFINED;
    m_scalingFunctionType = ScalingFunctionType::UNDEFINED;
    m_pScalingFunction = 0;
    m_scalingParameters = std::make_pair(0., 0.);
}

ScalingCell::ScalingCell(ScalingFunctionType::Type scalingFunctionType,
        ScalingModeType::Type mode) :
        NeuralNetworkCell("ScalingCell", NeuralNetworkCellType::ScalingCell) {

    m_scalingModeType = mode;
    m_scalingFunctionType = scalingFunctionType;
    m_pScalingFunction =
            NeuralNetworkTypeRegistry::getInstance()->getScalingFunction(
                    m_scalingFunctionType);
    m_scalingParameters = std::make_pair(0., 0.);

    m_properties.push_back(NeuralNetworkCellPropertyType::OnlyOneInputNeuron);
    m_properties.push_back(NeuralNetworkCellPropertyType::ManyOutputNeurons);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::RequairesFixedInputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequairesFixedOutputNeuronWeights);
    m_properties.push_back(
            NeuralNetworkCellPropertyType::DoNotRequiresTraining);
}

ScalingCell::ScalingCell(const ScalingCell& other) :
        NeuralNetworkCell(other) {

    m_scalingModeType = other.m_scalingModeType;
    m_scalingFunctionType = other.m_scalingFunctionType;

    if (other.m_pScalingFunction) {
        m_pScalingFunction = other.m_pScalingFunction->clone();
    }

    m_scalingParameters = other.m_scalingParameters;
}

ScalingCell::~ScalingCell() {

    if (m_pScalingFunction) {
        delete m_pScalingFunction;
        m_pScalingFunction = 0;
    }
}

ScalingCell* ScalingCell::clone() const {
    return new ScalingCell(*this);
}

std::string ScalingCell::toString() const {

    ElemUtils::Formatter formatter;

    formatter << NeuralNetworkCell::toString();
    formatter << "\t" << "scalingModeType: "
            << ScalingModeType(m_scalingModeType).toString() << "\n";
    formatter << "\t" << "scalingFunctionType: "
            << ScalingFunctionType(m_scalingFunctionType).toString() << "\n";
    formatter << "\t" << "scalingParameters: " << m_scalingParameters.first
            << " " << m_scalingParameters.second << "\n";

    return formatter.str();
}

ScalingModeType::Type ScalingCell::getScalingModeType() const {
    return m_scalingModeType;
}

ScalingFunctionType::Type ScalingCell::getScalingFunctionType() const {
    return m_scalingFunctionType;
}

const std::pair<double, double>& ScalingCell::getScalingParameters() const {
    return m_scalingParameters;
}

void ScalingCell::setScalingParameters(const std::pair<double, double>& input) {
    m_scalingParameters = input;
}

void ScalingCell::setScalingParameters(const std::vector<double>& input) {
    m_scalingParameters = m_pScalingFunction->evaluateParameters(input);
}

void ScalingCell::checkConsistency() const {
    NeuralNetworkCell::checkConsistency();
}

void ScalingCell::evaluate() {

    if (m_scalingFunctionType == ScalingFunctionType::NoScaling) {

        setOutput(m_neuronsIn.at(0)->getNeuralNetworkCellIn()->getOutput());

    } else {

        if (m_scalingParameters.first == m_scalingParameters.second) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "Scaling parameters not set or equal");
        }

        setOutput(
                m_pScalingFunction->evaluate(m_scalingModeType,
                        m_neuronsIn.at(0)->getNeuralNetworkCellIn()->getOutput(),
                        m_scalingParameters));
    }
}

double ScalingCell::evaluateDerivativeBackward(
        NeuralNetworkNeuron* const neuron) const {

//	if (neuron->getNeuralNetworkCellOut() != this) {
//		throw ElemUtils::CustomException(__func__, "Neuron not connected to this cell");
//	}

    if (m_scalingFunctionType == ScalingFunctionType::NoScaling) {

        return 1.;

    } else {

        if (m_scalingParameters.first == m_scalingParameters.second) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "Scaling parameters not set or equal");
        }

        return m_pScalingFunction->evaluateFirstDerivative(m_scalingModeType,
                m_neuronsIn.at(0)->getNeuralNetworkCellIn()->getOutput(),
                m_scalingParameters);
    }

}

