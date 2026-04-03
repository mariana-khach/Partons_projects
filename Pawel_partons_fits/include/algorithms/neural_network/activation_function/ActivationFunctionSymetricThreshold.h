/*
 * ActivationFunctionSymetricThreshold.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef ACTIVATIONFUNCTIONSYMETRICTHRESHOLD_H_
#define ACTIVATIONFUNCTIONSYMETRICTHRESHOLD_H_

#include "ActivationFunction.h"

class ActivationFunctionSymetricThreshold: public ActivationFunction {

public:

    static const unsigned int classId;

    ActivationFunctionSymetricThreshold();
    ~ActivationFunctionSymetricThreshold();
    virtual ActivationFunctionSymetricThreshold* clone() const;

    virtual double evaluate(double input);
    virtual double evaluateFirstDerivative(double input);
    virtual double evaluateSecondDerivative(double input);

protected:

    ActivationFunctionSymetricThreshold(const ActivationFunctionSymetricThreshold& other);
};

#endif /* ACTIVATIONFUNCTIONSYMETRICTHRESHOLD_H_ */
