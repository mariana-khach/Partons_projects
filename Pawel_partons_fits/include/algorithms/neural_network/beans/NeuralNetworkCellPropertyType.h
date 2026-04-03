/*
 * NeuralNetworkCellPropertyType.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURALNETWORKCELLPROPERTYTYPE_H_
#define NEURALNETWORKCELLPROPERTYTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class NeuralNetworkCellPropertyType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0,
        OnlyOneInputNeuron = 1,
        ManyInputNeurons = 2,
        NoInputNeurons = 3,
        OnlyOneOutputNeuron = 4,
        ManyOutputNeurons = 5,
        NoOutputNeurons = 6,
        RequairesFixedInputNeuronWeights = 7,
        DoNotRequairesFixedInputNeuronWeights = 8,
        RequairesFixedOutputNeuronWeights = 9,
        DoNotRequairesFixedOutputNeuronWeights = 10,
        RequiresTraining = 11,
        DoNotRequiresTraining = 12
    };

    NeuralNetworkCellPropertyType();
    NeuralNetworkCellPropertyType(NeuralNetworkCellPropertyType::Type type);
    NeuralNetworkCellPropertyType(const NeuralNetworkCellPropertyType& other);
    ~NeuralNetworkCellPropertyType();
    NeuralNetworkCellPropertyType* clone() const;

    std::string toString() const;

private:

    NeuralNetworkCellPropertyType::Type m_type;
};

#endif /* NEURALNETWORKCELLPROPERTYTYPE_H_ */
