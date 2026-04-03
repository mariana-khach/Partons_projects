/*
 * TrainingFunctionType.cpp
 *
 *  Created on: Jul 28, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/TrainingFunctionType.h"

TrainingFunctionType::TrainingFunctionType() :
        BaseObject("TrainingFunctionType") {
    m_type = TrainingFunctionType::UNDEFINED;
}

TrainingFunctionType::TrainingFunctionType(TrainingFunctionType::Type type) :
        BaseObject("TrainingFunctionType") {
    m_type = type;
}

TrainingFunctionType::TrainingFunctionType(
        const TrainingFunctionType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

TrainingFunctionType::~TrainingFunctionType() {
}

TrainingFunctionType* TrainingFunctionType::clone() const {
    return new TrainingFunctionType(*this);
}

std::string TrainingFunctionType::toString() const {

    switch (m_type) {

    case TrainingFunctionType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case TrainingFunctionType::Chi2: {
        return "BackPropagation";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}
