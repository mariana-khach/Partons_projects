/*
 * NeuralNetworkNeuron.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURALNETWORKNEURON_H_
#define NEURALNETWORKNEURON_H_

#include <string>

#include "../beans/InitializationType.h"

class RandomGenerator;

class NeuralNetworkCell;

class NeuralNetworkNeuron: public PARTONS::BaseObject {

public:

    NeuralNetworkNeuron();
    NeuralNetworkNeuron(NeuralNetworkCell* const NeuralNetworkCellIn,
            NeuralNetworkCell* const NeuralNetworkCellOut,
            InitializationType::Type initializationType);
    NeuralNetworkNeuron(const NeuralNetworkNeuron& other);
    ~NeuralNetworkNeuron();
    virtual NeuralNetworkNeuron* clone() const;
    virtual std::string toString() const;

    double getWeight() const;
    void setWeight(double weight);

    void fixWeight();
    void releaseWeight();
    bool isWeightFixed() const;

    NeuralNetworkCell* getNeuralNetworkCellIn() const;
    NeuralNetworkCell* getNeuralNetworkCellOut() const;

private:

    RandomGenerator* m_pRandomGenerator;

    double m_weight;
    bool m_fixedWeight;
    NeuralNetworkCell* m_neuralNetworkCellIn;
    NeuralNetworkCell* m_neuralNetwokCellOut;
};

#endif /* NEURALNETWORKNEURON_H_ */
