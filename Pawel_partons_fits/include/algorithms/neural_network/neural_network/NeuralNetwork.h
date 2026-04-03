/*
 * NeuralNetwork.h
 *
 *  Created on: May 1, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <string>
#include <vector>

#include "../beans/ActivationFunctionType.h"
#include "../beans/CombinationFunctionType.h"
#include "../beans/Data.h"
#include "../beans/InitializationType.h"
#include "../beans/ScalingFunctionType.h"
#include "../beans/TrainingAlgorithmType.h"
#include "../beans/TrainingFunctionType.h"
#include "../neural_network_cell/NeuralNetworkCell.h"
#include "../neural_network_layer/NeuralNetworkLayer.h"
#include "../neural_network_neuron/NeuralNetworkNeuron.h"

class NeuralNetwork: public PARTONS::BaseObject {

public:

    NeuralNetwork();
    NeuralNetwork(unsigned int nIn, unsigned int nOut,
            const std::vector<unsigned int>& perceptronLayers,
            ActivationFunctionType::Type hiddenActivationFunctionType,
            CombinationFunctionType::Type hiddenCombinationFunctionType,
            ActivationFunctionType::Type outputActivationFunctionType,
            CombinationFunctionType::Type outputCombinationFunctionType,
            ScalingFunctionType::Type scalingFunctionType,
            InitializationType::Type initializationType);
    NeuralNetwork(const NeuralNetwork& other);
    ~NeuralNetwork();
    virtual NeuralNetwork* clone() const;
    virtual std::string toString() const;

    Data evaluate(const Data& input) const;
    Data evaluateForOnePoint(const Data& input, unsigned int iPoint) const;
    void setScalingParameters(const Data& input, const Data& output) const;
    void train(const Data& input, const Data& output,
            TrainingAlgorithmType::Type trainingAlgorithmType,
            TrainingFunctionType::Type trainingFunctionType,
            const std::vector<double>& parameters);

    const std::vector<NeuralNetworkCell*>& getNeuralNetworkCells() const;
    const std::vector<NeuralNetworkLayer*>& getNeuralNetworkLayers() const;
    const std::vector<NeuralNetworkNeuron*>& getNeuralNetworkNeurons() const;
    unsigned int getNIn() const;
    unsigned int getNOut() const;

private:

    void connectLayersOneToOne(NeuralNetworkLayer* const layerIn,
            NeuralNetworkLayer* const layerOut, bool fixNeurons,
            InitializationType::Type initializationType);
    void connectLayersPermutive(NeuralNetworkLayer* const layerIn,
            NeuralNetworkLayer* const layerOut, bool fixNeurons,
            InitializationType::Type initializationType);
    void connectTwoCellsByNeuron(NeuralNetworkCell* const cell1,
            NeuralNetworkCell* const cell2, bool fixNeuron,
            InitializationType::Type initializationType);

    void checkConsistency() const;

    unsigned int m_nIn;
    unsigned int m_nOut;

    std::vector<NeuralNetworkCell*> m_neuralNetworkCells;
    std::vector<NeuralNetworkNeuron*> m_neuralNetworkNeurons;
    std::vector<NeuralNetworkLayer*> m_neuralNetworkLayers;
};

#endif /* NEURALNETWORK_H_ */
