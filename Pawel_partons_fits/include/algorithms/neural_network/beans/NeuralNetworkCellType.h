/*
 * NeuralNetworkCellType.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURALNETWORKCELLTYPE_H_
#define NEURALNETWORKCELLTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class NeuralNetworkCellType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0,
        InputCell = 1,
        OutputCell = 2,
        ScalingCell = 3,
        Perceptron = 4,
        TransitionCell = 5
    };

    NeuralNetworkCellType();
    NeuralNetworkCellType(NeuralNetworkCellType::Type type);
    NeuralNetworkCellType(const NeuralNetworkCellType& other);
    ~NeuralNetworkCellType();
    NeuralNetworkCellType* clone() const;

    std::string toString() const;

private:

    NeuralNetworkCellType::Type m_type;
};

#endif /* NEURALNETWORKCELLTYPE_H_ */
