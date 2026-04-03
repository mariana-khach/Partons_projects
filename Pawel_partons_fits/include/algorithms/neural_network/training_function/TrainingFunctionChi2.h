/*
 * TrainingFunctionChi2.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef TRAININGFUNCTIONCHI2_H_
#define TRAININGFUNCTIONCHI2_H_

#include "../beans/Data.h"
#include "TrainingFunction.h"

class TrainingFunctionChi2: public TrainingFunction {

public:

    static const unsigned int classId;

    TrainingFunctionChi2();
    ~TrainingFunctionChi2();
    virtual TrainingFunctionChi2* clone() const;

    virtual double evaluate(const Data& output,
            const Data& outputReference) const;
    virtual double partialDerivative(double output,
            double outputReference) const;

protected:

    TrainingFunctionChi2(const TrainingFunctionChi2& other);
};

#endif /* TRAININGFUNCTIONCHI2_H_ */
