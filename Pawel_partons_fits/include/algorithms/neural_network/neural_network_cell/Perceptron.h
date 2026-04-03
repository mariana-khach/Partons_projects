/*
 * Perceptron.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_

#include <string>

#include "../beans/ActivationFunctionType.h"
#include "../beans/CombinationFunctionType.h"
#include "../beans/InitializationType.h"
#include "NeuralNetworkCell.h"

class ActivationFunction;
class CombinationFunction;

class Perceptron: public NeuralNetworkCell {

public:

    static const unsigned int classId;

    Perceptron();
    Perceptron(ActivationFunctionType::Type activationFunctionType,
            CombinationFunctionType::Type combinationFunctionType,
            InitializationType::Type initializationType);
    ~Perceptron();
    virtual Perceptron* clone() const;
    virtual std::string toString() const;

    virtual void evaluate();
    virtual double evaluateDerivativeBackward(
            NeuralNetworkNeuron* const neuron) const;

    virtual void checkConsistency() const;

    ActivationFunctionType::Type getActivationFunctionType() const;
    CombinationFunctionType::Type getCombinationFunctionType() const;

    double getBias() const;
    void setBias(double bias);

    void fixBias();
    void releaseBias();
    bool isBiasFixed() const;

    double evaluateFirstDerivatieOfActivationFunction() const;
    double evaluateFirstPartialDerivatieOfCombinationFunctionForNeuron(
            const NeuralNetworkNeuron* const neuron) const;
    double evaluateFirstPartialDerivatieOfCombinationFunctionForBias() const;

protected:

    Perceptron(const Perceptron& other);

private:

    RandomGenerator* m_pRandomGenerator;

    double m_bias;
    bool m_fixedBias;
    ActivationFunctionType::Type m_activationFunctionType;
    CombinationFunctionType::Type m_combinationFunctionType;
    ActivationFunction* m_pActivationFunction;
    CombinationFunction* m_pCombinationFunction;
};

#endif /* PERCEPTRON_H_ */
