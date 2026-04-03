/*
 * ROOTMinimizerPackageType.cpp
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#include "../../../include/modules/minimizer_model/ROOTMinimizerPackageType.h"

#include <ElementaryUtils/string_utils/Formatter.h>
#include <ElementaryUtils/string_utils/StringUtils.h>

ROOTMinimizerPackageType::ROOTMinimizerPackageType() :
        BaseObject("ROOTMinimizerPackageType"), m_type(
                ROOTMinimizerPackageType::UNDEFINED) {
}

ROOTMinimizerPackageType::ROOTMinimizerPackageType(
        ROOTMinimizerPackageType::Type type) :
        BaseObject("ROOTMinimizerPackageType"), m_type(type) {
}

ROOTMinimizerPackageType::ROOTMinimizerPackageType(
        const std::string& rootMinimizerPackageTypeString) :
        BaseObject("ROOTMinimizerPackageType"), m_type(UNDEFINED) {

    if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "UNDEFINED")) {
        m_type = ROOTMinimizerPackageType::UNDEFINED;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "Default")) {
        m_type = ROOTMinimizerPackageType::Default;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "Minuit")) {
        m_type = ROOTMinimizerPackageType::Minuit;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "Minuit2")) {
        m_type = ROOTMinimizerPackageType::Minuit2;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "Fumili")) {
        m_type = ROOTMinimizerPackageType::Fumili;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "GSLMultiMin")) {
        m_type = ROOTMinimizerPackageType::GSLMultiMin;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "GSLMultiFit")) {
        m_type = ROOTMinimizerPackageType::GSLMultiFit;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "GSLSimAn")) {
        m_type = ROOTMinimizerPackageType::GSLSimAn;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerPackageTypeString,
            "Genetic")) {
        m_type = ROOTMinimizerPackageType::Genetic;
    } else {
        warn(__func__,
                ElemUtils::Formatter() << "Cannot find type for string = "
                        << rootMinimizerPackageTypeString);
    }
}

const std::string ROOTMinimizerPackageType::toString() {

    switch (m_type) {

    case Default:
        return "Default";
        break;
    case Minuit:
        return "Minuit";
        break;
    case Minuit2:
        return "Minuit2";
        break;
    case Fumili:
        return "Fumili";
        break;
    case GSLMultiMin:
        return "GSLMultiMin";
        break;
    case GSLMultiFit:
        return "GSLMultiFit";
        break;
    case GSLSimAn:
        return "GSLSimAn";
        break;
    case Genetic:
        return "Genetic";
        break;
    default:
        return "UNDEFINED";
    }
}

ROOTMinimizerPackageType::Type ROOTMinimizerPackageType::getType() const {
    return m_type;
}

void ROOTMinimizerPackageType::setType(ROOTMinimizerPackageType::Type type) {
    m_type = type;
}
