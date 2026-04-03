/*
 * ScalingFunctionMinMax.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef SCALINGFUNCTIONMINMAX_H_
#define SCALINGFUNCTIONMINMAX_H_

#include <utility>
#include <vector>

#include "../beans/ScalingModeType.h"
#include "ScalingFunction.h"

class ScalingFunctionMinMax: public ScalingFunction {

public:

    static const unsigned int classId;

    ScalingFunctionMinMax();
    ~ScalingFunctionMinMax();
    virtual ScalingFunctionMinMax* clone() const;

    virtual double evaluate(ScalingModeType::Type mode, double input,
            const std::pair<double, double>& parameters) const;
    virtual double evaluateFirstDerivative(ScalingModeType::Type mode,
            double input, const std::pair<double, double>& parameters) const;
    virtual std::pair<double, double> evaluateParameters(
            const std::vector<double>& input) const;

protected:

    ScalingFunctionMinMax(const ScalingFunctionMinMax& other);
};

#endif /* SCALINGFUNCTIONMINMAX_H_ */
