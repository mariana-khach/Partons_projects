/*
 * ScalingCell.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef SCALINGCELL_H_
#define SCALINGCELL_H_

#include <string>
#include <utility>
#include <vector>

#include "../beans/ScalingFunctionType.h"
#include "../beans/ScalingModeType.h"
#include "NeuralNetworkCell.h"

class ScalingFunction;

class ScalingCell: public NeuralNetworkCell {

public:

    ScalingCell();
    ScalingCell(ScalingFunctionType::Type scalingFunctionType,
            ScalingModeType::Type mode);
    ~ScalingCell();
    virtual ScalingCell* clone() const;
    virtual std::string toString() const;

    virtual void evaluate();
    virtual double evaluateDerivativeBackward(
            NeuralNetworkNeuron* const neuron) const;

    virtual void checkConsistency() const;

    ScalingModeType::Type getScalingModeType() const;
    ScalingFunctionType::Type getScalingFunctionType() const;

    const std::pair<double, double>& getScalingParameters() const;
    void setScalingParameters(const std::pair<double, double>& input);
    void setScalingParameters(const std::vector<double>& input);

protected:

    ScalingCell(const ScalingCell& other);

private:

    ScalingModeType::Type m_scalingModeType;
    ScalingFunctionType::Type m_scalingFunctionType;
    ScalingFunction* m_pScalingFunction;
    std::pair<double, double> m_scalingParameters;
};

#endif /* SCALINGCELL_H_ */
