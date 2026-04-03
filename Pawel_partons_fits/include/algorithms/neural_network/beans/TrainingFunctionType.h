/*
 * TrainingFunctionType.h
 *
 *  Created on: Jul 28, 2016
 *      Author: Pawel Sznajder
 */

#ifndef TRAININGFUNCTIONTYPE_H_
#define TRAININGFUNCTIONTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class TrainingFunctionType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0, Chi2 = 1
    };

    TrainingFunctionType();
    TrainingFunctionType(TrainingFunctionType::Type type);
    TrainingFunctionType(const TrainingFunctionType& other);
    ~TrainingFunctionType();
    TrainingFunctionType* clone() const;

    std::string toString() const;

private:

    TrainingFunctionType::Type m_type;
};

#endif /* TRAININGFUNCTIONTYPE_H_ */
