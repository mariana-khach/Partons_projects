/*
 * KinematicCutArithmeticOperator.cpp
 *
 *  Created on: Jan 22, 2017
 *      Author: Pawel Sznajder
 */

#include "../../../include/beans/kinematic_cut/KinematicCutArithmeticOperator.h"

KinematicCutArithmeticOperator::KinematicCutArithmeticOperator() :
        BaseObject("KinematicCutArithmeticOperator") {
    m_type = UNDEFINED;
}

KinematicCutArithmeticOperator::KinematicCutArithmeticOperator(
        KinematicCutArithmeticOperator::Type type) :
        BaseObject("KinematicCutArithmeticOperator") {
    m_type = type;
}

KinematicCutArithmeticOperator::KinematicCutArithmeticOperator(
        const KinematicCutArithmeticOperator& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

KinematicCutArithmeticOperator::~KinematicCutArithmeticOperator() {
}

KinematicCutArithmeticOperator* KinematicCutArithmeticOperator::clone() {
    return new KinematicCutArithmeticOperator(*this);
}

std::string KinematicCutArithmeticOperator::toString() const {

    switch (m_type) {

    case KinematicCutArithmeticOperator::eq: {
        return "eq";
    }

    case KinematicCutArithmeticOperator::ne: {
        return "ne";
    }

    case KinematicCutArithmeticOperator::lt: {
        return "lt";
    }

    case KinematicCutArithmeticOperator::le: {
        return "lt";
    }

    case KinematicCutArithmeticOperator::gt: {
        return "gt";
    }

    case KinematicCutArithmeticOperator::ge: {
        return "ge";
    }

    default: {
        return "UNDEFINED";
    }
    }
}
