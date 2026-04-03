/*
 * KinematicCutVariable.cpp
 *
 *  Created on: Jan 22, 2017
 *      Author: Pawel Sznajder
 */

#include "../../../include/beans/kinematic_cut/KinematicCutVariable.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/utils/type/PhysicalType.h>

using namespace PARTONS;

KinematicCutVariable::KinematicCutVariable() :
        BaseObject("KinematicCutVariable") {
    m_type = KinematicCutVariable::UNDEFINED;
}

KinematicCutVariable::KinematicCutVariable(KinematicCutVariable::Type type) :
        BaseObject("KinematicCutVariable") {
    m_type = type;
}

KinematicCutVariable::KinematicCutVariable(const KinematicCutVariable& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

KinematicCutVariable::~KinematicCutVariable() {
}

KinematicCutVariable* KinematicCutVariable::clone() {
    return new KinematicCutVariable(*this);
}

std::string KinematicCutVariable::toString() const {

    switch (m_type) {

    case KinematicCutVariable::Xb: {
        return "xB";
    }

    case KinematicCutVariable::T: {
        return "t";
    }

    case KinematicCutVariable::mT: {
        return "-t";
    }

    case KinematicCutVariable::Q2: {
        return "Q2";
    }

    case KinematicCutVariable::E: {
        return "E";
    }

    case KinematicCutVariable::Phi: {
        return "phi";
    }

    case KinematicCutVariable::ToverQ2: {
        return "t/Q2";
    }

    case KinematicCutVariable::mToverQ2: {
        return "-t/Q2";
    }

    default: {
        return "UNDEFINED";
    }
    }
}

double KinematicCutVariable::evaluate(
        const DVCSObservableKinematic& kinematics) const {

    switch (m_type) {

    case KinematicCutVariable::Xb: {
        return kinematics.getXB().getValue();
    }

    case KinematicCutVariable::T: {
        return kinematics.getT().getValue();
    }

    case KinematicCutVariable::mT: {
        return -1. * kinematics.getT().getValue();
    }

    case KinematicCutVariable::Q2: {
        return kinematics.getQ2().getValue();
    }

    case KinematicCutVariable::E: {
        return kinematics.getE().getValue();
    }

    case KinematicCutVariable::Phi: {
        return kinematics.getPhi().getValue();
    }

    case KinematicCutVariable::ToverQ2: {
        return kinematics.getT().getValue() / kinematics.getQ2().getValue();
    }

    case KinematicCutVariable::mToverQ2: {
        return -1. * kinematics.getT().getValue() / kinematics.getQ2().getValue();
    }

    default: {

        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Unable to evaluate for "
                        << toString());

        return 0.;
    }
    }
}
