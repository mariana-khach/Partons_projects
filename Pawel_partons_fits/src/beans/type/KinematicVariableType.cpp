/*
 * KinematicVariableType.cpp
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#include <ElementaryUtils/string_utils/Formatter.h>
#include <ElementaryUtils/string_utils/StringUtils.h>
#include "../../../include/beans/type/KinematicVariableType.h"

KinematicVariableType::KinematicVariableType() :
        BaseObject("KinematicVariableType"), m_type(
                KinematicVariableType::UNDEFINED) {
}

KinematicVariableType::KinematicVariableType(KinematicVariableType::Type type) :
        BaseObject("KinematicVariableType"), m_type(type) {
}

KinematicVariableType::KinematicVariableType(
        const std::string& fitStatusTypeString) :
        BaseObject("KinematicVariableType"), m_type(UNDEFINED) {

    if (ElemUtils::StringUtils::equals(fitStatusTypeString, "xi")) {
        m_type = KinematicVariableType::xi;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "xB")) {
        m_type = KinematicVariableType::xB;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "t")) {
        m_type = KinematicVariableType::t;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "Q2")) {
        m_type = KinematicVariableType::Q2;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "muF2")) {
        m_type = KinematicVariableType::muF2;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "muR2")) {
        m_type = KinematicVariableType::muR2;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "phi")) {
        m_type = KinematicVariableType::phi;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "E")) {
        m_type = KinematicVariableType::E;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString,
            "UNDEFINED")) {
        m_type = KinematicVariableType::UNDEFINED;
    } else {
        warn(__func__,
                ElemUtils::Formatter() << "Cannot find type for string = "
                        << fitStatusTypeString);
    }
}

const std::string KinematicVariableType::toString() {

    switch (m_type) {

    case xi:
        return "xi";
        break;
    case xB:
        return "xB";
        break;
    case t:
        return "t";
        break;
    case Q2:
        return "Q2";
        break;
    case muF2:
        return "muF2";
        break;
    case muR2:
        return "muR2";
        break;
    case phi:
        return "phi";
        break;
    case E:
        return "E";
        break;
    default:
        return "UNDEFINED";
    }
}

KinematicVariableType::Type KinematicVariableType::getType() const {
    return m_type;
}

void KinematicVariableType::setType(KinematicVariableType::Type type) {
    m_type = type;
}
