/*
 * ROOTMinimizerAlgorithmType.cpp
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#include "../../../include/modules/minimizer_model/ROOTMinimizerAlgorithmType.h"

#include <ElementaryUtils/string_utils/Formatter.h>
#include <ElementaryUtils/string_utils/StringUtils.h>

ROOTMinimizerAlgorithmType::ROOTMinimizerAlgorithmType() :
        BaseObject("ROOTMinimizerAlgorithmType"), m_type(
                ROOTMinimizerAlgorithmType::UNDEFINED) {
}

ROOTMinimizerAlgorithmType::ROOTMinimizerAlgorithmType(
        ROOTMinimizerAlgorithmType::Type type) :
        BaseObject("ROOTMinimizerAlgorithmType"), m_type(type) {
}

ROOTMinimizerAlgorithmType::ROOTMinimizerAlgorithmType(
        const std::string& rootMinimizerAlgorithmTypeString) :
        BaseObject("ROOTMinimizerAlgorithmType"), m_type(UNDEFINED) {

    if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "UNDEFINED")) {
        m_type = ROOTMinimizerAlgorithmType::UNDEFINED;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "Default")) {
        m_type = ROOTMinimizerAlgorithmType::Default;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "Migrad")) {
        m_type = ROOTMinimizerAlgorithmType::Migrad;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "Simplex")) {
        m_type = ROOTMinimizerAlgorithmType::Simplex;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "Combined")) {
        m_type = ROOTMinimizerAlgorithmType::Combined;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "Scan")) {
        m_type = ROOTMinimizerAlgorithmType::Scan;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "Fumili2")) {
        m_type = ROOTMinimizerAlgorithmType::Fumili2;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "ConjugateFR")) {
        m_type = ROOTMinimizerAlgorithmType::ConjugateFR;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "ConjugatePR")) {
        m_type = ROOTMinimizerAlgorithmType::ConjugatePR;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "BFGS")) {
        m_type = ROOTMinimizerAlgorithmType::BFGS;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "BFGS2")) {
        m_type = ROOTMinimizerAlgorithmType::BFGS2;
    } else if (ElemUtils::StringUtils::equals(rootMinimizerAlgorithmTypeString,
            "SteepestDescent")) {
        m_type = ROOTMinimizerAlgorithmType::SteepestDescent;
    } else {
        warn(__func__,
                ElemUtils::Formatter() << "Cannot find type for string = "
                        << rootMinimizerAlgorithmTypeString);
    }
}

const std::string ROOTMinimizerAlgorithmType::toString() {

    switch (m_type) {

    case Default:
        return "Default";
        break;
    case Migrad:
        return "Migrad";
        break;
    case Simplex:
        return "Simplex";
        break;
    case Combined:
        return "Combined";
        break;
    case Scan:
        return "Scan";
        break;
    case Fumili2:
        return "Fumili2";
        break;
    case ConjugateFR:
        return "ConjugateFR";
        break;
    case ConjugatePR:
        return "ConjugatePR";
        break;
    case BFGS:
        return "BFGS";
        break;
    case BFGS2:
        return "BFGS2";
        break;
    case SteepestDescent:
        return "SteepestDescent";
        break;
    default:
        return "UNDEFINED";
    }
}

ROOTMinimizerAlgorithmType::Type ROOTMinimizerAlgorithmType::getType() const {
    return m_type;
}

void ROOTMinimizerAlgorithmType::setType(
        ROOTMinimizerAlgorithmType::Type type) {
    m_type = type;
}
