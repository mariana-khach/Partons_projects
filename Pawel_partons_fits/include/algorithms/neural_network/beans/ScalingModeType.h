/*
 * ScalingModeType.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef SCALINGMODETYPE_H_
#define SCALINGMODETYPE_H_

#include <partons/BaseObject.h>
#include <string>

class ScalingModeType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0, Scale = 1, Unscale = 2
    };

    ScalingModeType();
    ScalingModeType(ScalingModeType::Type type);
    ScalingModeType(const ScalingModeType& other);
    ~ScalingModeType();
    ScalingModeType* clone() const;

    std::string toString() const;

private:

    ScalingModeType::Type m_type;
};

#endif /* SCALINGMODETYPE_H_ */
