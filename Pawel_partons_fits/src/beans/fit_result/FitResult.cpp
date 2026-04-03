/*
 * FitResult.cpp
 *
 *  Created on: Nov 10, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#include "../../../include/beans/fit_result/FitResult.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <iterator>
#include <sstream>

#include "../../../include/modules/minimizer_model/FitsMinimizerModule.h"

using namespace PARTONS;

FitResult::FitResult() :
        BaseObject("FitResult"), m_nData(0), m_fitStatusType(
                FitStatusType::UNDEFINED), m_fcnValue(0.), m_nCalls(0) {
}

FitResult::FitResult(const FitResult& other) :
        BaseObject(other), m_nData(other.m_nData), m_fitStatusType(
                other.m_fitStatusType), m_fcnValue(other.m_fcnValue), m_nCalls(
                other.m_nCalls) {

    m_variableNames = other.m_variableNames;
    m_variableStartValues = other.m_variableStartValues;
    m_variableFixedValues = other.m_variableFixedValues;
    m_variableSteps = other.m_variableSteps;
    m_variableLimitValues = other.m_variableLimitValues;
    m_variableFitValues = other.m_variableFitValues;
    m_variableFitErrors = other.m_variableFitErrors;
    m_correlationMatrix = other.m_correlationMatrix;
}

FitResult::~FitResult() {
}

size_t FitResult::getNData() const {
    return m_nData;
}

FitStatusType::Type FitResult::getFitStatusType() const {
    return m_fitStatusType;
}

double FitResult::getFcnValue() const {
    return m_fcnValue;
}

size_t FitResult::getNCalls() const {
    return m_nCalls;
}

const std::vector<std::string>& FitResult::getVariableNames() const {
    return m_variableNames;
}

const std::map<std::string, double>& FitResult::getVariableStartValues() const {
    return m_variableStartValues;
}

const std::map<std::string, bool>& FitResult::getVariableFixedValues() const {
    return m_variableFixedValues;
}

const std::map<std::string, double>& FitResult::getVariableSteps() const {
    return m_variableSteps;
}

const std::map<std::string, std::pair<double, double> >& FitResult::getVariableLimitValues() const {
    return m_variableLimitValues;
}

const std::map<std::string, double>& FitResult::getVariableFitValues() const {
    return m_variableFitValues;
}

const std::map<std::string, double>& FitResult::getVariableFitErrors() const {
    return m_variableFitErrors;
}

const NumA::MatrixD& FitResult::getCovarianceMatrix() const {
    return m_covarianceMatrix;
}

const NumA::MatrixD& FitResult::getCorrelationMatrix() const {
    return m_correlationMatrix;
}

void FitResult::gatherInformationFromDataFitList(
        const List<ObservableDataFit>& dataFitList) {
    m_nData = dataFitList.size();
}

void FitResult::gatherInformationFromMinimizerModule(
        const FitsMinimizerModule* minimizerModule) {

    m_fitStatusType = minimizerModule->getFitStatus();
    m_fcnValue = minimizerModule->getFCNValue();
    m_nCalls = minimizerModule->getNCalls();
    m_variableNames = minimizerModule->getVariableNames();
    m_variableStartValues = minimizerModule->getVariableStartValues();
    m_variableFixedValues = minimizerModule->getVariableFixedValues();
    m_variableSteps = minimizerModule->getVariableSteps();
    m_variableLimitValues = minimizerModule->getVariableLimitValues();

    std::vector<std::string>::const_iterator it;

    for (it = minimizerModule->getVariableNames().begin();
            it != minimizerModule->getVariableNames().end(); it++) {
        m_variableFitValues.insert(
                std::make_pair(*it, minimizerModule->getVariableValue(*it)));
        m_variableFitErrors.insert(
                std::make_pair(*it, minimizerModule->getVariableError(*it)));
    }

    m_covarianceMatrix = minimizerModule->getCovarianceMatrix();
    m_correlationMatrix = minimizerModule->getCorrelationMatrix();
}

void FitResult::writeToLogger() const {

    std::vector<std::string>::const_iterator itNames;
    std::map<std::string, bool>::const_iterator itFixedValues;

    size_t nFreeParameters = 0;

    for (itFixedValues = m_variableFixedValues.begin();
            itFixedValues != m_variableFixedValues.end(); itFixedValues++)
        nFreeParameters += size_t(!itFixedValues->second);

    info(__func__,
            ElemUtils::Formatter() << "Fit status: "
                    << FitStatusType(m_fitStatusType).toString());
    info(__func__, ElemUtils::Formatter() << "Number of calls: " << m_nCalls);
    info(__func__, ElemUtils::Formatter() << "FCN value: " << m_fcnValue);
    info(__func__,
            ElemUtils::Formatter()
                    << "Number of data points/all parameters/free parameters: "
                    << m_nData << "/" << m_variableNames.size() << "/"
                    << nFreeParameters);

    for (itNames = m_variableNames.begin(); itNames != m_variableNames.end();
            itNames++) {

        info(__func__,
                ElemUtils::Formatter() << "Parameter name/is fixed: "
                        << *itNames << "/"
                        << m_variableFixedValues.find(*itNames)->second);

        if (m_variableFixedValues.find(*itNames)->second) {
            info(__func__,
                    ElemUtils::Formatter() << "          value: "
                            << m_variableStartValues.find(*itNames)->second);
        } else {

            info(__func__,
                    ElemUtils::Formatter()
                            << "          limit min/initial value/initial step/limit max: "
                            << (m_variableLimitValues.find(*itNames)->second).first
                            << "/"
                            << m_variableStartValues.find(*itNames)->second
                            << "/" << m_variableSteps.find(*itNames)->second
                            << "/"
                            << (m_variableLimitValues.find(*itNames)->second).second);
            info(__func__,
                    ElemUtils::Formatter()
                            << "          fitted value/uncertainty: "
                            << m_variableFitValues.find(*itNames)->second << "/"
                            << m_variableFitErrors.find(*itNames)->second);
        }
    }

    info(__func__, ElemUtils::Formatter() << "Covariance/correlation matrix:");
    for (size_t i = 0; i < m_correlationMatrix.rows(); i++) {
        for (size_t j = 0; j < m_correlationMatrix.cols(); j++) {
            info(__func__,
                    ElemUtils::Formatter() << "          i: " << i << " j: "
                            << j << "\t" << m_covarianceMatrix.at(i, j) << "\t"
                            << m_correlationMatrix.at(i, j));
        }
    }
}
