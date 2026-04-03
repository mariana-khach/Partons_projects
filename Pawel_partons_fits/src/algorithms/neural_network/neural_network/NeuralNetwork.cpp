/*
 * NeuralNetwork.cpp
 *
 *  Created on: May 1, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/neural_network/NeuralNetwork.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <iterator>
#include <map>
#include <utility>

#include "../../../../include/algorithms/neural_network/beans/NeuralNetworkCellType.h"
#include "../../../../include/algorithms/neural_network/beans/ScalingModeType.h"
#include "../../../../include/algorithms/neural_network/neural_network_cell/InputCell.h"
#include "../../../../include/algorithms/neural_network/neural_network_cell/OutputCell.h"
#include "../../../../include/algorithms/neural_network/neural_network_cell/Perceptron.h"
#include "../../../../include/algorithms/neural_network/neural_network_cell/ScalingCell.h"

NeuralNetwork::NeuralNetwork() :
        BaseObject("NeuralNetwork") {

    m_nIn = 0;
    m_nOut = 0;

    m_neuralNetworkCells.clear();
    m_neuralNetworkNeurons.clear();
    m_neuralNetworkLayers.clear();
}

NeuralNetwork::NeuralNetwork(unsigned int nIn, unsigned int nOut,
        const std::vector<unsigned int>& perceptronLayers,
        ActivationFunctionType::Type hiddenActivationFunctionType,
        CombinationFunctionType::Type hiddenCombinationFunctionType,
        ActivationFunctionType::Type outputActivationFunctionType,
        CombinationFunctionType::Type outputCombinationFunctionType,
        ScalingFunctionType::Type scalingFunctionType,
        InitializationType::Type initializationType) :
        BaseObject("NeuralNetwork") {

    //iterator
    std::vector<unsigned int>::const_iterator it;

    //check input parameters
    if (nIn == 0 || nOut == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Number of input or output variables is zero");
    }

    if (perceptronLayers.size() == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Vector describing perceptron layers is empty");
    }

    for (it = perceptronLayers.begin(); it != perceptronLayers.end(); it++) {
        if (*it == 0) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "Size of perceptron layer is zero");
        }
    }

    //pointers
    NeuralNetworkLayer* p_layer = 0;
    NeuralNetworkCell* p_cell = 0;

    //input and output parameters
    m_nIn = nIn;
    m_nOut = nOut;

    //input layer
    p_layer = new NeuralNetworkLayer();
    m_neuralNetworkLayers.push_back(p_layer);

    for (unsigned int i = 0; i < nIn; i++) {
        p_cell = new InputCell();
        m_neuralNetworkCells.push_back(p_cell);

        p_layer->addCell(p_cell);

        p_cell = 0;
    }

    p_layer = 0;

    //scaling layer
    p_layer = new NeuralNetworkLayer();
    m_neuralNetworkLayers.push_back(p_layer);

    for (unsigned int i = 0; i < nIn; i++) {

        p_cell = new ScalingCell(scalingFunctionType, ScalingModeType::Scale);
        m_neuralNetworkCells.push_back(p_cell);

        p_layer->addCell(p_cell);

        p_cell = 0;
    }

    p_layer = 0;

    //neurons between input and scaling layer
    connectLayersOneToOne(*(m_neuralNetworkLayers.end() - 2),
            *(m_neuralNetworkLayers.end() - 1), true,
            InitializationType::Static);

    //perceptron layers
    for (it = perceptronLayers.begin(); it != perceptronLayers.end(); it++) {

        p_layer = new NeuralNetworkLayer();
        m_neuralNetworkLayers.push_back(p_layer);

        for (unsigned int i = 0; i < *it; i++) {

            p_cell = new Perceptron(hiddenActivationFunctionType,
                    hiddenCombinationFunctionType, initializationType);
            m_neuralNetworkCells.push_back(p_cell);

            p_layer->addCell(p_cell);

            p_cell = 0;
        }

        //neurons between two layer
        connectLayersPermutive(*(m_neuralNetworkLayers.end() - 2),
                *(m_neuralNetworkLayers.end() - 1), false, initializationType);

        p_layer = 0;
    }

    //last perceptron layer
    p_layer = new NeuralNetworkLayer();
    m_neuralNetworkLayers.push_back(p_layer);

    for (unsigned int i = 0; i < nOut; i++) {

        p_cell = new Perceptron(outputActivationFunctionType,
                outputCombinationFunctionType, initializationType);
        m_neuralNetworkCells.push_back(p_cell);

        p_layer->addCell(p_cell);

        p_cell = 0;
    }

    p_layer = 0;

    //neurons between two layer
    connectLayersPermutive(*(m_neuralNetworkLayers.end() - 2),
            *(m_neuralNetworkLayers.end() - 1), false, initializationType);

    //unscaling layer
    p_layer = new NeuralNetworkLayer();
    m_neuralNetworkLayers.push_back(p_layer);

    for (unsigned int i = 0; i < nOut; i++) {

        p_cell = new ScalingCell(scalingFunctionType, ScalingModeType::Unscale);
        m_neuralNetworkCells.push_back(p_cell);

        p_layer->addCell(p_cell);

        p_cell = 0;
    }

    p_layer = 0;

    //neurons between last perceptron layer and unscaling layer
    connectLayersOneToOne(*(m_neuralNetworkLayers.end() - 2),
            *(m_neuralNetworkLayers.end() - 1), true,
            InitializationType::Static);

    //output layer
    p_layer = new NeuralNetworkLayer();
    m_neuralNetworkLayers.push_back(p_layer);

    for (unsigned int i = 0; i < nOut; i++) {
        p_cell = new OutputCell();
        m_neuralNetworkCells.push_back(p_cell);

        p_layer->addCell(p_cell);

        p_cell = 0;
    }

    p_layer = 0;

    //neurons between unscaling and output layers
    connectLayersOneToOne(*(m_neuralNetworkLayers.end() - 2),
            *(m_neuralNetworkLayers.end() - 1), true,
            InitializationType::Static);

    //check consistency
    checkConsistency();
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other) :
        BaseObject(other) {

    m_nIn = other.m_nIn;
    m_nOut = other.m_nOut;

    m_neuralNetworkCells = other.m_neuralNetworkCells;
    m_neuralNetworkNeurons = other.m_neuralNetworkNeurons;
    m_neuralNetworkLayers = other.m_neuralNetworkLayers;
}

NeuralNetwork::~NeuralNetwork() {

    std::vector<NeuralNetworkCell*>::iterator itCell;

    for (itCell = m_neuralNetworkCells.begin();
            itCell != m_neuralNetworkCells.end(); itCell++) {

        if (*itCell) {
            delete *itCell;
            *itCell = 0;
        }
    }
    m_neuralNetworkCells.clear();

    std::vector<NeuralNetworkNeuron*>::iterator itNeuron;

    for (itNeuron = m_neuralNetworkNeurons.begin();
            itNeuron != m_neuralNetworkNeurons.end(); itNeuron++) {

        if (*itNeuron) {
            delete *itNeuron;
            *itNeuron = 0;
        }
    }
    m_neuralNetworkNeurons.clear();

    std::vector<NeuralNetworkLayer*>::iterator itLayer;

    for (itLayer = m_neuralNetworkLayers.begin();
            itLayer != m_neuralNetworkLayers.end(); itLayer++) {

        if (*itLayer) {
            delete *itLayer;
            *itLayer = 0;
        }
    }
    m_neuralNetworkLayers.clear();
}

NeuralNetwork* NeuralNetwork::clone() const {
    return new NeuralNetwork(*this);
}

std::string NeuralNetwork::toString() const {

    ElemUtils::Formatter formatter;
    std::vector<NeuralNetworkLayer*>::const_iterator it1;
    std::vector<NeuralNetworkCell*>::const_iterator it2;
    std::vector<NeuralNetworkNeuron*>::const_iterator it3;

    formatter << "\n";
    formatter << "=== Layers ===================" << "\n";
    for (it1 = m_neuralNetworkLayers.begin();
            it1 != m_neuralNetworkLayers.end(); it1++)
        formatter << (*it1)->toString();
    formatter << "=== Cells ====================" << "\n";
    for (it2 = m_neuralNetworkCells.begin(); it2 != m_neuralNetworkCells.end();
            it2++)
        formatter << (*it2)->toString();
    formatter << "=== Neurons ==================" << "\n";
    for (it3 = m_neuralNetworkNeurons.begin();
            it3 != m_neuralNetworkNeurons.end(); it3++)
        formatter << (*it3)->toString();

    return formatter.str();

}

void NeuralNetwork::connectLayersOneToOne(NeuralNetworkLayer* const layerIn,
        NeuralNetworkLayer* const layerOut, bool fixNeurons,
        InitializationType::Type initializationType) {

    if ((layerIn->getCells()).size() == 0
            || (layerOut->getCells()).size() == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Size of layer is zero");
    }

    if ((layerIn->getCells()).size() != (layerOut->getCells()).size()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Sizes of layers are different");
    }

    std::vector<NeuralNetworkCell*>::const_iterator itIn, itOut;

    for (itIn = (layerIn->getCells()).begin(), itOut =
            (layerOut->getCells()).begin();
            itIn != (layerIn->getCells()).end()
                    || itOut != (layerOut->getCells()).end(); itIn++, itOut++) {

        connectTwoCellsByNeuron(*itIn, *itOut, fixNeurons, initializationType);
    }
}

void NeuralNetwork::connectLayersPermutive(NeuralNetworkLayer* const layerIn,
        NeuralNetworkLayer* const layerOut, bool fixNeurons,
        InitializationType::Type initializationType) {

    if ((layerIn->getCells()).size() == 0
            || (layerOut->getCells()).size() == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Size of layer is zero");
    }

    std::vector<NeuralNetworkCell*>::const_iterator itIn, itOut;

    for (itIn = (layerIn->getCells()).begin();
            itIn != (layerIn->getCells()).end(); itIn++) {
        for (itOut = (layerOut->getCells()).begin();
                itOut != (layerOut->getCells()).end(); itOut++) {

            connectTwoCellsByNeuron(*itIn, *itOut, fixNeurons,
                    initializationType);
        }
    }
}

void NeuralNetwork::connectTwoCellsByNeuron(NeuralNetworkCell* const cell1,
        NeuralNetworkCell* const cell2, bool fixNeuron,
        InitializationType::Type initializationType) {

    if (cell1 == 0 || cell2 == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "One of pointer is null");
    }

    NeuralNetworkNeuron* p_neuron = new NeuralNetworkNeuron(cell1, cell2,
            initializationType);
    m_neuralNetworkNeurons.push_back(p_neuron);

    cell1->addNeuronOut(p_neuron);
    cell2->addNeuronIn(p_neuron);

    if (fixNeuron)
        p_neuron->fixWeight();
}

void NeuralNetwork::setScalingParameters(const Data& input,
        const Data& output) const {

    if (input.getNPoints() == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Size of input vector is zero");
    }

    if (output.getNPoints() == 0) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Size of output vector is zero");
    }

    std::vector<NeuralNetworkCell*>::const_iterator itCell;
    std::map<unsigned int, std::vector<double> >::const_iterator itInput,
            itOutput;

    for (itCell = ((*(m_neuralNetworkLayers.begin()))->getCells()).begin(), itInput =
            (input.getData()).begin();
            itCell != ((*(m_neuralNetworkLayers.begin()))->getCells()).end()
                    || itInput != (input.getData()).end();
            itCell++, itInput++) {

        if ((*itCell)->getType() != NeuralNetworkCellType::InputCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No input cell when expected");
        }

        if (((*itCell)->getNeuronsOut()).size() != 1) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "None or more than one output neuron");
        }

        NeuralNetworkCell* scalingCell =
                ((*itCell)->getNeuronsOut())[0]->getNeuralNetworkCellOut();

        if (scalingCell->getType() != NeuralNetworkCellType::ScalingCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No scaling cell when expected");
        }

        (static_cast<ScalingCell*>(scalingCell))->setScalingParameters(
                itInput->second);
    }

    for (itCell = ((*(m_neuralNetworkLayers.end() - 1))->getCells()).begin(), itOutput =
            (output.getData()).begin();
            itCell != ((*(m_neuralNetworkLayers.end() - 1))->getCells()).end()
                    || itOutput != (output.getData()).end();
            itCell++, itOutput++) {

        if ((*itCell)->getType() != NeuralNetworkCellType::OutputCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No output cell when expected");
        }

        if (((*itCell)->getNeuronsIn()).size() != 1) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "None or more than one output neuron");
        }

        NeuralNetworkCell* scalingCell =
                ((*itCell)->getNeuronsIn())[0]->getNeuralNetworkCellIn();

        if (scalingCell->getType() != NeuralNetworkCellType::ScalingCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No scaling cell when expected");
        }

        (static_cast<ScalingCell*>(scalingCell))->setScalingParameters(
                itOutput->second);
    }
}

void NeuralNetwork::checkConsistency() const {

    std::vector<NeuralNetworkCell*>::const_iterator it;

    for (it = m_neuralNetworkCells.begin(); it != m_neuralNetworkCells.end();
            it++) {
        (*it)->checkConsistency();
    }
}

Data NeuralNetwork::evaluateForOnePoint(const Data& input,
        unsigned int iPoint) const {

    //check input parameters
    if (input.getNVariables() != m_nIn) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Size of input vector different than that of input parameters to neural network");
    }

    //check i
    if (input.getNPoints() <= iPoint) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Illegal point index");
    }

    //output
    Data output(m_nOut);

    //loop over input data
    std::vector<NeuralNetworkCell*>::const_iterator itCell;
    std::map<unsigned int, std::vector<double> >::const_iterator itInput;

    //set input
    for (itCell = ((*(m_neuralNetworkLayers.begin()))->getCells()).begin(), itInput =
            (input.getData()).begin();
            itCell != ((*(m_neuralNetworkLayers.begin()))->getCells()).end()
                    || itInput != (input.getData()).end();
            itCell++, itInput++) {

        if ((*itCell)->getType() != NeuralNetworkCellType::InputCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No input cell when expected");
        }

        (static_cast<InputCell*>(*itCell))->setInput(
                (itInput->second).at(iPoint));
    }

    //run
    std::vector<NeuralNetworkLayer*>::const_iterator itLayer;

    for (itLayer = m_neuralNetworkLayers.begin();
            itLayer != m_neuralNetworkLayers.end(); itLayer++) {
        (*itLayer)->evaluate();
    }

    //get result
    std::vector<double> outputData;

    for (itCell = ((*(m_neuralNetworkLayers.end() - 1))->getCells()).begin();
            itCell != ((*(m_neuralNetworkLayers.end() - 1))->getCells()).end();
            itCell++) {

        if ((*itCell)->getType() != NeuralNetworkCellType::OutputCell) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No output cell when expected");
        }

        outputData.push_back((static_cast<OutputCell*>(*itCell))->getOutput());
    }

    //add data
    output.addData(outputData);

    //return
    return output;
}

Data NeuralNetwork::evaluate(const Data& input) const {

    //output
    Data output(m_nOut);

    //loop over points
    for (unsigned int i = 0; i < input.getNPoints(); i++) {

        //evaluate
        output += evaluateForOnePoint(input, i);
    }

    //return
    return output;
}

void NeuralNetwork::train(const Data& input, const Data& output,
        TrainingAlgorithmType::Type trainingAlgorithmType,
        TrainingFunctionType::Type trainingFunctionType,
        const std::vector<double>& parameters) {

//    NeuralNetworkTypeRegistry::getInstance()->getTrainingAlgorithm(
//            trainingAlgorithmType)->train(*this, input, output,
//            trainingFunctionType, parameters);
}

const std::vector<NeuralNetworkCell*>& NeuralNetwork::getNeuralNetworkCells() const {
    return m_neuralNetworkCells;
}

const std::vector<NeuralNetworkLayer*>& NeuralNetwork::getNeuralNetworkLayers() const {
    return m_neuralNetworkLayers;
}

const std::vector<NeuralNetworkNeuron*>& NeuralNetwork::getNeuralNetworkNeurons() const {
    return m_neuralNetworkNeurons;
}

unsigned int NeuralNetwork::getNIn() const {
    return m_nIn;
}

unsigned int NeuralNetwork::getNOut() const {
    return m_nOut;
}
