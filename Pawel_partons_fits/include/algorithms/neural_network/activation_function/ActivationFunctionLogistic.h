/*
 * ActivationFunctionLogistic.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef ACTIVATIONFUNCTIONLOGISTIC_H_
#define ACTIVATIONFUNCTIONLOGISTIC_H_

#include "ActivationFunction.h"

class ActivationFunctionLogistic: public ActivationFunction {

public:

    static const unsigned int classId;

    ActivationFunctionLogistic();
    ~ActivationFunctionLogistic();
    virtual ActivationFunctionLogistic* clone() const;

    virtual double evaluate(double input);
    virtual double evaluateFirstDerivative(double input);
    virtual double evaluateSecondDerivative(double input);

protected:

    ActivationFunctionLogistic(const ActivationFunctionLogistic& other);
};

#endif /* ACTIVATIONFUNCTIONLOGISTIC_H_ */
