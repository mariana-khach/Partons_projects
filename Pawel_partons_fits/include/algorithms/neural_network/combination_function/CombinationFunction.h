/*
 * CombinationFunction.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef COMBINATIONFUNCTION_H_
#define COMBINATIONFUNCTION_H_

#include <string>
#include <vector>

#include "../neural_network_neuron/NeuralNetworkNeuron.h"

class CombinationFunction: public PARTONS::BaseObject {

public:

    CombinationFunction();
    CombinationFunction(const std::string& name);
    virtual ~CombinationFunction();
    virtual CombinationFunction* clone() const;

    virtual double evaluate(const std::vector<NeuralNetworkNeuron*>& neurons,
            double bias) const;

    virtual double evaluateFirstPartialDerivativeForCell(
            const NeuralNetworkNeuron* const neuron,
            const std::vector<NeuralNetworkNeuron*>& neurons,
            double bias) const;
    virtual double evaluateFirstPartialDerivativeForNeuron(
            const NeuralNetworkNeuron* const neuron,
            const std::vector<NeuralNetworkNeuron*>& neurons,
            double bias) const;
    virtual double evaluateFirstPartialDerivativeForBias(
            const std::vector<NeuralNetworkNeuron*>& neurons,
            double bias) const;

protected:

    CombinationFunction(const CombinationFunction& other);
};

#endif /* COMBINATIONFUNCTION_H_ */
