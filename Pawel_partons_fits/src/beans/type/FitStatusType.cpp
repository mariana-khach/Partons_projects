/*
 * FitStatusType.cpp
 *
 *  Created on: Nov 3, 2016
 *      Author: debian
 */

#include "../../../include/beans/type/FitStatusType.h"

#include <ElementaryUtils/string_utils/Formatter.h>
#include <ElementaryUtils/string_utils/StringUtils.h>

FitStatusType::FitStatusType() :
        BaseObject("FitStatusType"), m_type(FitStatusType::UNDEFINED) {
}

FitStatusType::FitStatusType(FitStatusType::Type type) :
        BaseObject("FitStatusType"), m_type(type) {
}

FitStatusType::FitStatusType(const std::string& fitStatusTypeString) :
        BaseObject("FitStatusType"), m_type(UNDEFINED) {

    if (ElemUtils::StringUtils::equals(fitStatusTypeString, "SUCCESSFUL")) {
        m_type = FitStatusType::SUCCESSFUL;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString,
            "PROBLEMS")) {
        m_type = FitStatusType::PROBLEMS;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString, "FAILED")) {
        m_type = FitStatusType::FAILED;
    } else if (ElemUtils::StringUtils::equals(fitStatusTypeString,
            "UNDEFINED")) {
        m_type = FitStatusType::UNDEFINED;
    } else {
        warn(__func__,
                ElemUtils::Formatter()
                        << "Cannot find type for string = "
                        << fitStatusTypeString);
    }
}

const std::string FitStatusType::toString() {

    switch (m_type) {

    case SUCCESSFUL:
        return "SUCCESSFUL";
        break;
    case PROBLEMS:
        return "PROBLEMS";
        break;
    case FAILED:
        return "FAILED";
        break;
    default:
        return "UNDEFINED";
    }
}

FitStatusType::Type FitStatusType::getType() const {
    return m_type;
}

void FitStatusType::setType(FitStatusType::Type type) {
    m_type = type;
}
