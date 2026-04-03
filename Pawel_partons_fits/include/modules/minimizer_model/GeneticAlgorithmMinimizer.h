/*
 * GeneticAlgorithmMinimizer.h
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef GENETICALGORITHMMINIMIZER_H_
#define GENETICALGORITHMMINIMIZER_H_

#include <ElementaryUtils/parameters/Parameters.h>
#include <stddef.h>
#include <string>
#include <utility>

#include "../../beans/type/FitStatusType.h"
#include "FitsMinimizerModule.h"

class GeneticAlgorithm;

class GeneticAlgorithmMinimizer: public FitsMinimizerModule {

public:

    static const std::string PARAMETER_NAME_POPULATION_SIZE_MIN;
    static const std::string PARAMETER_NAME_POPULATION_SIZE_MAX;
    static const std::string PARAMETER_NAME_POPULATION_SIZE_SLOPE;
    static const std::string PARAMETER_NAME_BEST_FRACTION;
    static const std::string PARAMETER_NAME_CROSS_SLOPE;
    static const std::string PARAMETER_NAME_MUTATION_A_PROB_MAX;
    static const std::string PARAMETER_NAME_MUTATION_B_PROB_MAX;
    static const std::string PARAMETER_NAME_MUTATION_A_SIZE;
    static const std::string PARAMETER_NAME_MUTATION_A_RESISTANCE;
    static const std::string PARAMETER_NAME_MUTATION_B_RESISTANCE;
    static const std::string PARAMETER_NAME_MUTATION_A_SLOPE;
    static const std::string PARAMETER_NAME_MUTATION_B_SLOPE;
    static const std::string PARAMETER_NAME_STOP_CONDITION_A;
    static const std::string PARAMETER_NAME_STOP_CONDITION_B1;
    static const std::string PARAMETER_NAME_STOP_CONDITION_B2;
    static const std::string PARAMETER_NAME_START_FROM_DUMP_FILE;
    static const std::string PARAMETER_NAME_DUMP_BEST_TO_FILE_EVERY;
    static const std::string PARAMETER_NAME_DUMP_BEST_TO_FILE_PATH;
    static const std::string PARAMETER_NAME_DUMP_POPULATION_TO_FILE_EVERY;
    static const std::string PARAMETER_NAME_DUMP_POPULATION_TO_FILE_PATH;

    static const size_t classId;

    GeneticAlgorithmMinimizer(const std::string& className);
    virtual ~GeneticAlgorithmMinimizer();

    virtual void resolveObjectDependencies();
    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual GeneticAlgorithmMinimizer* clone() const;

    virtual size_t SetVariable(const std::string& parName, double initialValue,
            double initialStep);
    virtual size_t SetLimitedVariable(const std::string& parName,
            double initialValue, double initialStep,
            const std::pair<double, double>& limits);
    virtual size_t SetFixedVariable(const std::string& parName,
            double initialValue);
    virtual double getVariableValue(size_t i) const;
    virtual double getVariableValue(const std::string& varName) const;
    virtual double getVariableError(size_t i) const;
    virtual double getVariableError(const std::string& varName) const;
    virtual double getFCNValue() const;
    virtual FitStatusType::Type getFitStatus() const;
    virtual size_t getNCalls() const;
    virtual void runMinimization();
    virtual void setFCN(FitsService* c, fcnType fcn, size_t nPar);
    virtual NumA::MatrixD getCovarianceMatrix() const;
    virtual NumA::MatrixD getCorrelationMatrix() const;

protected:

    GeneticAlgorithmMinimizer(const GeneticAlgorithmMinimizer &other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    GeneticAlgorithm* m_geneticAlgorithm;
};

#endif /* GENETICALGORITHMMINIMIZER_H_ */
