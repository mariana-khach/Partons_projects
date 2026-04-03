/*
 * InitializationType.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/InitializationType.h"

InitializationType::InitializationType() :
        BaseObject("InitializationType") {
    m_type = InitializationType::UNDEFINED;
}

InitializationType::InitializationType(InitializationType::Type type) :
        BaseObject("InitializationType") {
    m_type = type;
}

InitializationType::InitializationType(
        const InitializationType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

InitializationType::~InitializationType() {
}

InitializationType* InitializationType::clone() const {
    return new InitializationType(*this);
}

std::string InitializationType::toString() const {

    switch (m_type) {

    case InitializationType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case InitializationType::Static: {
        return "Static";
    }

        break;

    case InitializationType::Random: {
        return "Random";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}

