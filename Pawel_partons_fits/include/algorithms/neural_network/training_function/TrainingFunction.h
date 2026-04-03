/*
 * TrainingFunction.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef TRAININGFUNCTION_H_
#define TRAININGFUNCTION_H_

#include <string>

#include "../beans/Data.h"

class TrainingFunction: public PARTONS::BaseObject {

public:

    TrainingFunction();
    TrainingFunction(const std::string& name);
    virtual ~TrainingFunction();
    virtual TrainingFunction* clone() const;

    virtual double evaluate(const Data& output,
            const Data& outputReference) const;
    virtual double partialDerivative(double output,
            double outputReference) const;

protected:

    TrainingFunction(const TrainingFunction& other);
};

#endif /* TRAININGFUNCTION_H_ */
