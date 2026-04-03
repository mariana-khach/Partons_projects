/*
 * TransitionCell.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef TRANSITIONCELL_H_
#define TRANSITIONCELL_H_

#include <string>

#include "NeuralNetworkCell.h"

class TransitionCell: public NeuralNetworkCell {

public:

    TransitionCell();
    ~TransitionCell();
    virtual TransitionCell* clone() const;
    virtual std::string toString() const;

    virtual void evaluate();
    virtual double evaluateDerivativeBackward(
            NeuralNetworkNeuron* const neuron) const;

    virtual void checkConsistency() const;

protected:

    TransitionCell(const TransitionCell& other);
};

#endif /* TRANSITIONCELL_H_ */
