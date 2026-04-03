/*
 * KinematicCut.cpp
 *
 *  Created on: Jan 22, 2017
 *      Author: Pawel Sznajder
 */

#include "../../../include/beans/kinematic_cut/KinematicCut.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <iterator>
#include <sstream>
#include <vector>

using namespace PARTONS;

KinematicCut::KinematicCut() :
        BaseObject("KinematicCut") {

    m_variable = KinematicCutVariable::UNDEFINED;

    m_lActive = false;
    m_lOperator = KinematicCutArithmeticOperator::UNDEFINED;
    m_lValue = 0.;

    m_rActive = false;
    m_rOperator = KinematicCutArithmeticOperator::UNDEFINED;
    m_rValue = 0.;
}

KinematicCut::KinematicCut(const std::string& data) :
        BaseObject("KinematicCut") {

    //reset
    m_variable = KinematicCutVariable::UNDEFINED;

    m_lActive = false;
    m_lOperator = KinematicCutArithmeticOperator::UNDEFINED;
    m_lValue = 0.;

    m_rActive = false;
    m_rOperator = KinematicCutArithmeticOperator::UNDEFINED;
    m_rValue = 0.;

    //trim
    std::string dataTrimmed;

    for (size_t i = 0; i < data.length(); i++) {

        if (i != data.length() - 1) {
            if (std::isspace(data.at(i)) && std::isspace(data.at(i + 1)))
                continue;
        }

        dataTrimmed.push_back(data.at(i));
    }

    dataTrimmed.erase(dataTrimmed.begin(),
            std::find_if(dataTrimmed.begin(), dataTrimmed.end(),
                    std::not1(std::ptr_fun<int, int>(std::isspace))));

    dataTrimmed.erase(
            std::find_if(dataTrimmed.rbegin(), dataTrimmed.rend(),
                    std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
            dataTrimmed.end());

    //get tokens
    std::vector<std::string> tokens;

    std::size_t start = 0;
    std::size_t end = 0;

    while ((end = dataTrimmed.find(' ', start)) != std::string::npos) {
        tokens.push_back(dataTrimmed.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(dataTrimmed.substr(start));

    //check number of tokens
    if (!(tokens.size() == 3 || tokens.size() == 5)) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Invalid number of tokens, "
                        << tokens.size() << ", for string " << data);
    }

    //if it is 5
    if (tokens.size() == 5) {

        m_variable = parseCutVariable(tokens.at(2));

        if (m_variable == KinematicCutVariable::UNDEFINED) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to find variable in "
                            << data);
        }

        m_lActive = true;
        m_lOperator = parseCutArithmeticOperator(tokens.at(1));
        m_lValue = parseCutValue(tokens.at(0));

        m_rActive = true;
        m_rOperator = parseCutArithmeticOperator(tokens.at(3));
        m_rValue = parseCutValue(tokens.at(4));
    }

    //if it is 3
    if (tokens.size() == 3) {

        KinematicCutVariable::Type lVariable = parseCutVariable(tokens.at(0));
        KinematicCutVariable::Type rVariable = parseCutVariable(tokens.at(2));

        if (lVariable != KinematicCutVariable::UNDEFINED
                && rVariable == KinematicCutVariable::UNDEFINED) {

            m_variable = lVariable;
            m_rActive = true;
            m_rOperator = parseCutArithmeticOperator(tokens.at(1));
            m_rValue = parseCutValue(tokens.at(2));
        }

        else if (lVariable == KinematicCutVariable::UNDEFINED
                && rVariable != KinematicCutVariable::UNDEFINED) {

            m_variable = rVariable;
            m_lActive = true;
            m_lOperator = parseCutArithmeticOperator(tokens.at(1));
            m_lValue = parseCutValue(tokens.at(0));
        }

        else{
            throw ElemUtils::CustomException(getClassName(), __func__,
                            ElemUtils::Formatter() << "Unable to determine variable in "
                                    << data);
        }
    }

    //check
    if (m_lActive && m_lOperator == KinematicCutArithmeticOperator::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Unable to find left operator in "
                        << data);
    }

    if (m_rActive && m_rOperator == KinematicCutArithmeticOperator::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Unable to find right operator in "
                        << data);
    }
}

KinematicCut::KinematicCut(const KinematicCut& other) :
        BaseObject(other) {

    m_variable = other.m_variable;

    m_lActive = other.m_lActive;
    m_lOperator = other.m_lOperator;
    m_lValue = other.m_lValue;

    m_rActive = other.m_rActive;
    m_rOperator = other.m_rOperator;
    m_rValue = other.m_rValue;
}

KinematicCut::~KinematicCut() {
}

KinematicCut* KinematicCut::clone() {
    return new KinematicCut(*this);
}

std::string KinematicCut::toString() const {

    std::stringstream stream;

    if (m_lActive) {
        stream << m_lValue << " "
                << KinematicCutArithmeticOperator(m_lOperator).toString()
                << " ";
    }

    stream << KinematicCutVariable(m_variable).toString();

    if (m_rActive) {
        stream << " " << KinematicCutArithmeticOperator(m_rOperator).toString()
                << " " << m_rValue;
    }

    return stream.str();
}

bool KinematicCut::evaluate(const DVCSObservableKinematic& kinematics) const {

    //get variable value
    double variableValue = KinematicCutVariable(m_variable).evaluate(
            kinematics);

    //lhs
    bool lhs = true;

    if (m_lActive)
        lhs = KinematicCutArithmeticOperator(m_lOperator).evaluate(m_lValue,
                variableValue);

    //rhs
    bool rhs = true;

    if (m_rActive)
        rhs = KinematicCutArithmeticOperator(m_rOperator).evaluate(variableValue,
                m_rValue);

    //return
    return (lhs & rhs);
}

KinematicCutVariable::Type KinematicCut::parseCutVariable(
        const std::string& data) const {

    for (size_t i = size_t(KinematicCutVariable::UNDEFINED) + 1;
            i < size_t(KinematicCutVariable::LAST); i++) {

        KinematicCutVariable::Type type = KinematicCutVariable::Type(i);
        if (KinematicCutVariable(type).toString() == data)
            return type;
    }

    return KinematicCutVariable::UNDEFINED;
}

KinematicCutArithmeticOperator::Type KinematicCut::parseCutArithmeticOperator(
        const std::string& data) const {

    for (size_t i = size_t(KinematicCutArithmeticOperator::UNDEFINED) + 1;
            i < size_t(KinematicCutArithmeticOperator::LAST); i++) {

        KinematicCutArithmeticOperator::Type type =
                KinematicCutArithmeticOperator::Type(i);
        if (KinematicCutArithmeticOperator(type).toString() == data)
            return type;
    }

    return KinematicCutArithmeticOperator::UNDEFINED;
}

double KinematicCut::parseCutValue(const std::string& data) const {

    double value;
    std::stringstream convert(data);

    convert >> value;

    if (!convert.eof() || convert.fail()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Unable to convert into number for "
                        << data);
    }

    return value;
}
