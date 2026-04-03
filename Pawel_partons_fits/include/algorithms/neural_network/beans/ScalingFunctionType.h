/*
 * ScalingFunctionType.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef SCALINGFUNCTIONTYPE_H_
#define SCALINGFUNCTIONTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class ScalingFunctionType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0, NoScaling = 1, MeanStdDeviation = 2, MinMax = 3
    };

    ScalingFunctionType();
    ScalingFunctionType(ScalingFunctionType::Type type);
    ScalingFunctionType(const ScalingFunctionType& other);
    ~ScalingFunctionType();
    ScalingFunctionType* clone() const;

    std::string toString() const;

private:

    ScalingFunctionType::Type m_type;
};

#endif /* SCALINGFUNCTIONTYPE_H_ */
