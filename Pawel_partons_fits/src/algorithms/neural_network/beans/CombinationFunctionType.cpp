/*
 * CombinationFunctionType.cpp
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/CombinationFunctionType.h"

CombinationFunctionType::CombinationFunctionType() :
        BaseObject("CombinationFunctionType") {
    m_type = CombinationFunctionType::UNDEFINED;
}

CombinationFunctionType::CombinationFunctionType(
        CombinationFunctionType::Type type) :
        BaseObject("CombinationFunctionType") {
    m_type = type;
}

CombinationFunctionType::CombinationFunctionType(
        const CombinationFunctionType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

CombinationFunctionType::~CombinationFunctionType() {
}

CombinationFunctionType* CombinationFunctionType::clone() const {
    return new CombinationFunctionType(*this);
}

std::string CombinationFunctionType::toString() const {

    switch (m_type) {

    case CombinationFunctionType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case CombinationFunctionType::ScalarProduct: {
        return "ScalarProduct";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}

