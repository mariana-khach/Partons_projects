/*
 * NeuralNetworkCell.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURALNETWORKCELL_H_
#define NEURALNETWORKCELL_H_

#include <string>
#include <vector>

#include "../beans/NeuralNetworkCellPropertyType.h"
#include "../beans/NeuralNetworkCellType.h"
#include "../neural_network_neuron/NeuralNetworkNeuron.h"

class NeuralNetworkCell: public PARTONS::BaseObject {

public:

    NeuralNetworkCell();
    NeuralNetworkCell(const std::string& name,
            NeuralNetworkCellType::Type type);
    virtual ~NeuralNetworkCell();
    virtual NeuralNetworkCell* clone() const;
    virtual std::string toString() const;

    NeuralNetworkCellType::Type getType() const;

    virtual void evaluate();
    virtual double evaluateDerivativeBackward(
            NeuralNetworkNeuron* const neuron) const;

    const std::vector<NeuralNetworkNeuron*>& getNeuronsIn() const;
    void addNeuronIn(NeuralNetworkNeuron* const neuron);
    const std::vector<NeuralNetworkNeuron*>& getNeuronsOut() const;
    void addNeuronOut(NeuralNetworkNeuron* const neuron);

    double getOutput() const;
    void setOutput(double output);

    virtual void checkConsistency() const;
    bool checkProperty(NeuralNetworkCellPropertyType::Type property) const;

protected:

    NeuralNetworkCell(const NeuralNetworkCell& other);

    double m_output;
    std::vector<NeuralNetworkNeuron*> m_neuronsIn;
    std::vector<NeuralNetworkNeuron*> m_neuronsOut;
    std::vector<NeuralNetworkCellPropertyType::Type> m_properties;

private:

    NeuralNetworkCellType::Type m_type;
};

#endif /* NEURALNETWORKCELL_H_ */
