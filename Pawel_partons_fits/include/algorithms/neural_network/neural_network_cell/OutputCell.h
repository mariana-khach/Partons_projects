/*
 * OutputCell.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef OUTPUTCELL_H_
#define OUTPUTCELL_H_

#include <string>

#include "NeuralNetworkCell.h"

class OutputCell: public NeuralNetworkCell {

public:

    OutputCell();
    ~OutputCell();
    virtual OutputCell* clone() const;
    virtual std::string toString() const;

    virtual void evaluate();
    virtual double evaluateDerivativeBackward(
            NeuralNetworkNeuron* const neuron) const;

    virtual void checkConsistency() const;

protected:

    OutputCell(const OutputCell& other);
};

#endif /* OUTPUTCELL_H_ */
