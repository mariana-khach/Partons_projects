/*
 * FitsMinimizerModule.h
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef FITSMINIMIZERMODULE_H_
#define FITSMINIMIZERMODULE_H_

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/channel/ChannelType.h>
#include <partons/ModuleObject.h>
#include <stddef.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../../beans/type/FitStatusType.h"
#include "../../services/FitsService.h"

namespace NumA {
class MatrixD;
} /* namespace NumA */

class FitsMinimizerModule: public PARTONS::ModuleObject {

public:

    /** Constructor
     @param className Class name
     */
    FitsMinimizerModule(const std::string &className);

    /** Destructor
     */
    virtual ~FitsMinimizerModule();

    virtual void resolveObjectDependencies();
    virtual void configure(const ElemUtils::Parameters &variables);
    virtual FitsMinimizerModule* clone() const = 0;

    void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    /**
     * Set variable
     * @param varName Name
     * @param initialValue Initial value
     * @param step Initial step
     * @return Registered index
     */
    virtual size_t SetVariable(const std::string& varName, double initialValue,
            double step);

    /**
     * Set limited variable
     * @param varName Name
     * @param initialValue Initial value
     * @param step Initial step
     * @param limits Limits (min, max)
     * @return Registered index
     */
    virtual size_t SetLimitedVariable(const std::string& varName,
            double initialValue, double step,
            const std::pair<double, double>& limits);

    /**
     * Set fixed variable
     * @param varName Name
     * @param initialValue Initial value
     * @return Registered index
     */
    virtual size_t SetFixedVariable(const std::string& varName,
            double initialValue);

    /**
     * Get number of variables
     */
    size_t getNVariables() const;

    /** Check if variable is registered by name
     * @param varName Name
     */
    bool isVariableRegistered(const std::string& varName) const;

    /** Check if variable is registered by index
     * @param i Index
     */
    bool isVariableRegistered(size_t i) const;

    /**
     * Get name of variable by index
     * @param i Index
     */
    std::string getVariableName(size_t i) const;

    /**
     * Get variable index by name
     * @param i Index
     */
    size_t getVariableIndex(const std::string& varName) const;

    /**
     * Get variable start value by index
     * @param i Index
     */
    double getVariableStartValue(size_t i) const;

    /**
     * Get variable start value by name
     * @param varName Name
     */
    double getVariableStartValue(const std::string& varName) const;

    /**
     * Get variable step by index
     * @param i Index
     */
    double getVariableStep(size_t i) const;

    /**
     * Get variable step by name
     * @param varName Name
     */
    double getVariableStep(const std::string& varName) const;

    /**
     * Check if variable fixed by index
     * @param i Index
     */
    bool isVariableFixed(size_t i) const;

    /**
     * Check if variable fixed by name
     * @param varName Name
     */
    bool isVariableFixed(const std::string& varName) const;

    /**
     * Get variable limits by index
     * @param i Index
     */
    std::pair<double, double> getVariableLimits(size_t i) const;

    /**
     * Get variable limits by name
     * @param varName Name
     */
    std::pair<double, double> getVariableLimits(
            const std::string& varName) const;

    /**
     * Get value of variable by index
     * @param i Index
     */
    virtual double getVariableValue(size_t i) const = 0;

    /**
     * Get value of variable by name
     * @param varName Name
     */
    virtual double getVariableValue(const std::string& varName) const = 0;

    /**
     * Get error of variable by index
     * @param i Index
     */
    virtual double getVariableError(size_t i) const = 0;

    /**
     * Get error of variable by name
     * @param varName Name
     */
    virtual double getVariableError(const std::string& varName) const = 0;

    /**
     * Get fcn value
     */
    virtual double getFCNValue() const = 0;

    /**
     * Get fit status
     */
    virtual FitStatusType::Type getFitStatus() const = 0;

    /**
     * Get number of calls
     */
    virtual size_t getNCalls() const = 0;

    /**
     * Run minimization
     */
    virtual void runMinimization() = 0;

    /**
     * Definition of pointer to FCN function
     */
    typedef double (FitsService::*fcnType)(const double* params);

    /** Set FCN
     * @param c Reference to class
     * @param fcn FCN function
     * @param n Number of variables
     */
    virtual void setFCN(FitsService* c, fcnType fcn, size_t nPar) = 0;

    /**
     * Get variable names
     */
    const std::vector<std::string>& getVariableNames() const;

    /**
     * Get variable start values
     */
    const std::map<std::string, double>& getVariableStartValues() const;

    /**
     * Get variable steps
     */
    const std::map<std::string, double>& getVariableSteps() const;

    /**
     * Get variable fixed values
     */
    const std::map<std::string, bool>& getVariableFixedValues() const;

    /**
     * Get variable limit values
     */
    const std::map<std::string, std::pair<double, double> >& getVariableLimitValues() const;

    /**
     * Get covariance matrix
     */
    virtual NumA::MatrixD getCovarianceMatrix() const = 0;

    /**
      * Get correlation matrix
      */
     virtual NumA::MatrixD getCorrelationMatrix() const = 0;

protected:

    /** Copy constructor
     * @param other Object to be copied
     */
    FitsMinimizerModule(const FitsMinimizerModule &other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    std::vector<std::string> m_variableNames;
    std::map<std::string, double> m_variableStartValues;
    std::map<std::string, double> m_variableSteps;
    std::map<std::string, bool> m_variableFixedValues;
    std::map<std::string, std::pair<double, double> > m_variableLimitValues;
};

#endif /* FITSMINIMIZERMODULE_H_ */
