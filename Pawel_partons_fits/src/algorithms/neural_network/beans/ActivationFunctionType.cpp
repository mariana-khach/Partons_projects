/*
 * ActivationFunctionType.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/ActivationFunctionType.h"

ActivationFunctionType::ActivationFunctionType() :
        BaseObject("ActivationFunctionType") {
    m_type = ActivationFunctionType::UNDEFINED;
}

ActivationFunctionType::ActivationFunctionType(
        ActivationFunctionType::Type type) :
        BaseObject("ActivationFunctionType") {
    m_type = type;
}

ActivationFunctionType::ActivationFunctionType(
        const ActivationFunctionType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

ActivationFunctionType::~ActivationFunctionType() {
}

ActivationFunctionType* ActivationFunctionType::clone() const {
    return new ActivationFunctionType(*this);
}

std::string ActivationFunctionType::toString() const {

    switch (m_type) {

    case ActivationFunctionType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case ActivationFunctionType::Hyperbolic: {
        return "Hyperbolic";
    }

        break;

    case ActivationFunctionType::Linear: {
        return "Linear";
    }

        break;

    case ActivationFunctionType::Logistic: {
        return "Logistic";
    }

        break;

    case ActivationFunctionType::SymetricThreshold: {
        return "SymetricThreshold";
    }

        break;

    case ActivationFunctionType::Threshold: {
        return "Threshold";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}

