/*
 * NeuralNetworkLayer.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURALNETWORKLAYER_H_
#define NEURALNETWORKLAYER_H_

#include <string>
#include <vector>

#include "../neural_network_cell/NeuralNetworkCell.h"

class NeuralNetworkLayer: public PARTONS::BaseObject {

public:

    NeuralNetworkLayer();
    NeuralNetworkLayer(const NeuralNetworkLayer& other);
    virtual ~NeuralNetworkLayer();
    virtual NeuralNetworkLayer* clone() const;
    virtual std::string toString() const;

    const std::vector<NeuralNetworkCell*>& getCells() const;
    void addCell(NeuralNetworkCell* const cell);

    void evaluate() const;

private:

    std::vector<NeuralNetworkCell*> m_cells;
};

#endif /* NEURALNETWORKLAYER_H_ */
