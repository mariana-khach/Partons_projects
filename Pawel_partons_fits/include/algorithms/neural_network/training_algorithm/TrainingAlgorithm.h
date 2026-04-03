/*
 * TrainingAlgorithm.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef TRAININGALGORITHM_H_
#define TRAININGALGORITHM_H_

#include <string>
#include <vector>

#include "../beans/Data.h"
#include "../beans/TrainingFunctionType.h"

class NeuralNetwork;

class TrainingAlgorithm: public PARTONS::BaseObject {

public:

    TrainingAlgorithm();
    TrainingAlgorithm(const std::string& name);
    virtual ~TrainingAlgorithm();
    virtual TrainingAlgorithm* clone() const;

    virtual void train(const NeuralNetwork& nn, const Data& input,
            const Data& output, TrainingFunctionType::Type trainingFunctionType,
            const std::vector<double>& parameters);

protected:

    TrainingAlgorithm(const TrainingAlgorithm& other);
};

#endif /* TRAININGALGORITHM_H_ */
