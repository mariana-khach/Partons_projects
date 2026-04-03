/*
 * ScalingFunction.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef SCALINGFUNCTION_H_
#define SCALINGFUNCTION_H_

#include <string>
#include <utility>
#include <vector>

#include "../beans/ScalingModeType.h"

class ScalingFunction: public PARTONS::BaseObject {

public:

    ScalingFunction();
    ScalingFunction(const std::string& name);
    virtual ~ScalingFunction();
    virtual ScalingFunction* clone() const;

    virtual double evaluate(ScalingModeType::Type mode, double input,
            const std::pair<double, double>& parameters) const;
    virtual double evaluateFirstDerivative(ScalingModeType::Type mode,
            double input, const std::pair<double, double>& parameters) const;
    virtual std::pair<double, double> evaluateParameters(
            const std::vector<double>& input) const;

protected:

    ScalingFunction(const ScalingFunction& other);
};

#endif /* SCALINGFUNCTION_H_ */
