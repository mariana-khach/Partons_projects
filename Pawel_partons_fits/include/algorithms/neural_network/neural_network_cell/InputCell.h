/*
 * InputCell.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef INPUTCELL_H_
#define INPUTCELL_H_

#include <string>

#include "NeuralNetworkCell.h"

class InputCell: public NeuralNetworkCell {

public:

    InputCell();
    ~InputCell();
    virtual InputCell* clone() const;
    virtual std::string toString() const;

    virtual void evaluate();
    virtual double evaluateDerivativeBackward(
            NeuralNetworkNeuron* const neuron) const;

    virtual void checkConsistency() const;

    void setInput(double input);
    double getInput() const;

protected:

    InputCell(const InputCell& other);

private:

    double m_input;
};

#endif /* INPUTCELL_H_ */
