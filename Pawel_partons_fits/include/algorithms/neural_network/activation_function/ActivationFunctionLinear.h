/*
 * ActivationFunctionLinear.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef ACTIVATIONFUNCTIONLINEAR_H_
#define ACTIVATIONFUNCTIONLINEAR_H_

#include "ActivationFunction.h"

class ActivationFunctionLinear: public ActivationFunction {

public:

    static const unsigned int classId;

    ActivationFunctionLinear();
    ~ActivationFunctionLinear();
    virtual ActivationFunctionLinear* clone() const;

    virtual double evaluate(double input);
    virtual double evaluateFirstDerivative(double input);
    virtual double evaluateSecondDerivative(double input);

protected:

    ActivationFunctionLinear(const ActivationFunctionLinear& other);
};

#endif /* ACTIVATIONFUNCTIONLINEAR_H_ */
