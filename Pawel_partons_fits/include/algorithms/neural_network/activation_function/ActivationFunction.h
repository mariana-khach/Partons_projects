/*
 * ActivationFunction.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_

#include <partons/BaseObject.h>
#include <string>

class ActivationFunction: public PARTONS::BaseObject {

public:

    ActivationFunction();
    ActivationFunction(const std::string& name);
    virtual ~ActivationFunction();
    virtual ActivationFunction* clone() const;

    virtual double evaluate(double input);
    virtual double evaluateFirstDerivative(double input);
    virtual double evaluateSecondDerivative(double input);

protected:

    ActivationFunction(const ActivationFunction& other);
};

#endif /* ACTIVATIONFUNCTION_H_ */
