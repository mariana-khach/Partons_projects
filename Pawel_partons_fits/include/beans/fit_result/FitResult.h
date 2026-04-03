/*
 * FitResult.h
 *
 *  Created on: Nov 10, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef FITRESULT_H_
#define FITRESULT_H_

#include <NumA/linear_algebra/matrix/MatrixD.h>
#include <partons/beans/List.h>
#include <stddef.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../observable_data_fit/ObservableDataFit.h"
#include "../type/FitStatusType.h"

class FitsMinimizerModule;

class FitResult: public PARTONS::BaseObject {

public:

    FitResult();
    FitResult(const FitResult& other);
    virtual ~FitResult();

    size_t getNData() const;
    FitStatusType::Type getFitStatusType() const;
    double getFcnValue() const;
    size_t getNCalls() const;
    const std::vector<std::string>& getVariableNames() const;
    const std::map<std::string, double>& getVariableStartValues() const;
    const std::map<std::string, bool>& getVariableFixedValues() const;
    const std::map<std::string, double>& getVariableSteps() const;
    const std::map<std::string, std::pair<double, double> >& getVariableLimitValues() const;
    const std::map<std::string, double>& getVariableFitValues() const;
    const std::map<std::string, double>& getVariableFitErrors() const;
    const NumA::MatrixD& getCovarianceMatrix() const;
    const NumA::MatrixD& getCorrelationMatrix() const;

    void gatherInformationFromDataFitList(const PARTONS::List<ObservableDataFit>& dataFitList);
    void gatherInformationFromMinimizerModule(const FitsMinimizerModule* minimizerModule);

    void writeToLogger() const;

private:

    size_t m_nData;
    FitStatusType::Type m_fitStatusType;
    double m_fcnValue;
    size_t m_nCalls;
    std::vector<std::string> m_variableNames;
    std::map<std::string, double> m_variableStartValues;
    std::map<std::string, bool> m_variableFixedValues;
    std::map<std::string, double> m_variableSteps;
    std::map<std::string, std::pair<double, double> > m_variableLimitValues;
    std::map<std::string, double> m_variableFitValues;
    std::map<std::string, double> m_variableFitErrors;
    NumA::MatrixD m_covarianceMatrix;
    NumA::MatrixD m_correlationMatrix;
};

#endif /* FITRESULT_H_ */
