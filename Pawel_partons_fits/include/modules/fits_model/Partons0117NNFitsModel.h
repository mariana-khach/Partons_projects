/*
 * Partons0117NNFitsModel.h
 *
 *  Created on: 17 Jun 2017
 *      Author: Pawel Sznajder (NCBJ)
 */

#ifndef PARTONS0117NNMODEL_H_
#define PARTONS0117NNMODEL_H_

#include <ElementaryUtils/parameters/Parameters.h>
#include <string>
#include <vector>

#include "../../algorithms/neural_network/neural_network_cell/Perceptron.h"
#include "../../algorithms/neural_network/neural_network_neuron/NeuralNetworkNeuron.h"
#include "FitsModelModule.h"

class DVCSCFFNN;

class Partons0117NNFitsModel: public FitsModelModule {

public:

    static const unsigned int classId;

    Partons0117NNFitsModel(const std::string& className);
    virtual ~Partons0117NNFitsModel();

    virtual void resolveObjectDependencies();
    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual Partons0117NNFitsModel* clone() const;

    virtual void initParameters();
    virtual void updateParameters(const double* Var);

    virtual void initModule();
    virtual void isModuleWellConfigured();

protected:

    Partons0117NNFitsModel(const Partons0117NNFitsModel &other);

private:

    DVCSCFFNN* m_pDVCSNNCFFModel;
    std::vector<Perceptron*> m_perceptonMap;
    std::vector<NeuralNetworkNeuron*> m_neuronMap;

    std::vector<Perceptron*>::iterator itPerceptron;
    std::vector<NeuralNetworkNeuron*>::iterator itNeuron;

};

#endif /* PARTONS0117NNMODEL_H_ */
