/*
 * TrainingType.h
 *
 *  Created on: Jul 28, 2016
 *      Author: Pawel Sznajder
 */

#ifndef TRAININGALGORITHMTYPE_H_
#define TRAININGALGORITHMTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class TrainingAlgorithmType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0, BackPropagation = 1, GeneticAlgorithm = 2
    };

    TrainingAlgorithmType();
    TrainingAlgorithmType(TrainingAlgorithmType::Type type);
    TrainingAlgorithmType(const TrainingAlgorithmType& other);
    ~TrainingAlgorithmType();
    TrainingAlgorithmType* clone() const;

    std::string toString() const;

private:

    TrainingAlgorithmType::Type m_type;
};

#endif /* TRAININGALGORITHMTYPE_H_ */
