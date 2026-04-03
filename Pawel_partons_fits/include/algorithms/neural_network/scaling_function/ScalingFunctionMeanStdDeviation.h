/*
 * ScalingFunctionMeanStdDeviation.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef SCALINGFUNCTIONMEANSTDDEVIATION_H_
#define SCALINGFUNCTIONMEANSTDDEVIATION_H_

#include <utility>
#include <vector>

#include "../beans/ScalingModeType.h"
#include "ScalingFunction.h"

class ScalingFunctionMeanStdDeviation: public ScalingFunction {

public:

    static const unsigned int classId;

    ScalingFunctionMeanStdDeviation();
    ~ScalingFunctionMeanStdDeviation();
    virtual ScalingFunctionMeanStdDeviation* clone() const;

    virtual double evaluate(ScalingModeType::Type mode, double input,
            const std::pair<double, double>& parameters) const;
    virtual double evaluateFirstDerivative(ScalingModeType::Type mode,
            double input, const std::pair<double, double>& parameters) const;
    virtual std::pair<double, double> evaluateParameters(
            const std::vector<double>& input) const;

protected:

    ScalingFunctionMeanStdDeviation(
            const ScalingFunctionMeanStdDeviation& other);
};

#endif /* SCALINGFUNCTIONMEANSTDDEVIATION_H_ */
