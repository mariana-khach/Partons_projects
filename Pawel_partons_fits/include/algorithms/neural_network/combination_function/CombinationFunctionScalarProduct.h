/*
 * CombinationFunctionScalarProduct.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef COMBINATIONFUNCTIONSCALARPRODUCT_H_
#define COMBINATIONFUNCTIONSCALARPRODUCT_H_

#include <vector>

#include "../neural_network_neuron/NeuralNetworkNeuron.h"
#include "CombinationFunction.h"

class CombinationFunctionScalarProduct: public CombinationFunction {

public:

    static const unsigned int classId;

    CombinationFunctionScalarProduct();
    ~CombinationFunctionScalarProduct();
    virtual CombinationFunctionScalarProduct* clone() const;

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

    CombinationFunctionScalarProduct(
            const CombinationFunctionScalarProduct& other);
};

#endif /* COMBINATIONFUNCTIONSCALARPRODUCT_H_ */
