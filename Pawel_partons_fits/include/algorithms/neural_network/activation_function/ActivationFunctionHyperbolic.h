/*
 * ActivationFunctionHyperbolic.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef ACTIVATIONFUNCTIONHYPERBOLIC_H_
#define ACTIVATIONFUNCTIONHYPERBOLIC_H_

#include "ActivationFunction.h"

class ActivationFunctionHyperbolic: public ActivationFunction {

public:

    static const unsigned int classId;

    ActivationFunctionHyperbolic();
    ~ActivationFunctionHyperbolic();
    virtual ActivationFunctionHyperbolic* clone() const;

    virtual double evaluate(double input);
    virtual double evaluateFirstDerivative(double input);
    virtual double evaluateSecondDerivative(double input);

protected:

    ActivationFunctionHyperbolic(const ActivationFunctionHyperbolic& other);
};

#endif /* ACTIVATIONFUNCTIONHYPERBOLIC_H_ */
