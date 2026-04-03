/*
 * ActivationFunctionThreshold.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef ACTIVATIONFUNCTIONTHRESHOLD_H_
#define ACTIVATIONFUNCTIONTHRESHOLD_H_

#include "ActivationFunction.h"

class ActivationFunctionThreshold: public ActivationFunction {

public:

    static const unsigned int classId;

    ActivationFunctionThreshold();
    ~ActivationFunctionThreshold();
    virtual ActivationFunctionThreshold* clone() const;

    virtual double evaluate(double input);
    virtual double evaluateFirstDerivative(double input);
    virtual double evaluateSecondDerivative(double input);

protected:

    ActivationFunctionThreshold(const ActivationFunctionThreshold& other);
};

#endif /* ACTIVATIONFUNCTIONTHRESHOLD_H_ */
