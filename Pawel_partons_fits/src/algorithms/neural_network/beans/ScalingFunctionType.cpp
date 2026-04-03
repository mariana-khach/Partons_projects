/*
 * ScalingFunctionType.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/ScalingFunctionType.h"

ScalingFunctionType::ScalingFunctionType() :
        BaseObject("ScalingFunctionType") {
    m_type = ScalingFunctionType::UNDEFINED;
}

ScalingFunctionType::ScalingFunctionType(ScalingFunctionType::Type type) :
        BaseObject("ScalingFunctionType") {
    m_type = type;
}

ScalingFunctionType::ScalingFunctionType(
        const ScalingFunctionType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

ScalingFunctionType::~ScalingFunctionType() {
}

ScalingFunctionType* ScalingFunctionType::clone() const {
    return new ScalingFunctionType(*this);
}

std::string ScalingFunctionType::toString() const {

    switch (m_type) {

    case ScalingFunctionType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case ScalingFunctionType::NoScaling: {
        return "NoScaling";
    }

        break;

    case ScalingFunctionType::MeanStdDeviation: {
        return "MeanStdDeviation";
    }

        break;

    case ScalingFunctionType::MinMax: {
        return "MinMax";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}

