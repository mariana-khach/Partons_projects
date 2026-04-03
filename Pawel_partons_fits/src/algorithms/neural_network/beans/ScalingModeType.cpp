/*
 * ScalingModeType.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/ScalingModeType.h"

ScalingModeType::ScalingModeType() :
        BaseObject("ScalingModeType") {
    m_type = ScalingModeType::UNDEFINED;
}

ScalingModeType::ScalingModeType(ScalingModeType::Type type) :
        BaseObject("ScalingModeType") {
    m_type = type;
}

ScalingModeType::ScalingModeType(
        const ScalingModeType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

ScalingModeType::~ScalingModeType() {
}

ScalingModeType* ScalingModeType::clone() const {
    return new ScalingModeType(*this);
}

std::string ScalingModeType::toString() const {

    switch (m_type) {

    case ScalingModeType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case ScalingModeType::Scale: {
        return "Scale";
    }

        break;

    case ScalingModeType::Unscale: {
        return "Unscale";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}

