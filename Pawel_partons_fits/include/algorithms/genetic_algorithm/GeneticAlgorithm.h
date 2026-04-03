/*
 * GeneticAlgorithm.h
 *
 *  Created on: Dec 22, 2015
 *      Author: Pawel Sznajder
 */

#ifndef GENETICALGORITHM_H_
#define GENETICALGORITHM_H_

#include <stddef.h>
#include <string>
#include <utility>
#include <vector>

#include "../../beans/type/FitStatusType.h"
#include "../../services/FitsService.h"
#include "GeneticAlgorithmCandidate.h"

class GeneticAlgorithm: public PARTONS::BaseObject {

public:

    GeneticAlgorithm();
    GeneticAlgorithm(const GeneticAlgorithm& other);
    ~GeneticAlgorithm();

    void run();

    size_t addUnlimitedVariable(const std::string& name, double initialValue,
            size_t nSteps);
    size_t addLimitedVariable(const std::string& name, double initialValue,
            const std::pair<double, double>& limit, double step);
    size_t addFixedVariable(const std::string& name, double value);

    typedef double (FitsService::*fcnType)(const double* params);
    void setFCN(FitsService* fitsService, fcnType fcn);

    double getBestFraction() const;
    void setBestFraction(double bestFraction);
    double getCrossSlope() const;
    void setCrossSlope(double crossSlope);
    double getMutationAProbMax() const;
    void setMutationAProbMax(double mutationAProbMax);
    double getMutationAResistance() const;
    void setMutationAResistance(double mutationAResistance);
    double getMutationASize() const;
    void setMutationASize(double mutationASize);
    double getMutationASlope() const;
    void setMutationASlope(double mutationASlope);
    double getMutationBProbMax() const;
    void setMutationBProbMax(double mutationBProbMax);
    double getMutationBResistance() const;
    void setMutationBResistance(double mutationBResistance);
    double getMutationBSlope() const;
    void setMutationBSlope(double mutationBSlope);
    size_t getPopulationSizeMax() const;
    void setPopulationSizeMax(size_t populationSizeMax);
    size_t getPopulationSizeMin() const;
    void setPopulationSizeMin(size_t populationSizeMin);
    double getPopulationSizeSlope() const;
    void setPopulationSizeSlope(double populationSizeSlope);
    size_t getStopConditionA() const;
    void setStopConditionA(size_t stopConditionA);
    size_t getStopConditionB1() const;
    void setStopConditionB1(size_t stopConditionB1);
    double getStopConditionB2() const;
    void setStopConditionB2(double stopConditionB2);
    double getBestFcn() const;
    FitStatusType::Type getFitStatus() const;
    size_t getNCalls() const;
    const std::vector<double>& getBestParameters() const;
    size_t getDumpBestToFileEvery() const;
    void setDumpBestToFileEvery(size_t dumpBestToFileEvery);
    const std::string& getDumpBestToFilePath() const;
    void setDumpBestToFilePath(const std::string& dumpBestToFilePath);
    size_t getDumpPopulationToFileEvery() const;
    void setDumpPopulationToFileEvery(size_t dumpPopulationToFileEvery);
    const std::string& getDumpPopulationToFilePath() const;
    void setDumpPopulationToFilePath(
            const std::string& dumpPopulationToFilePath);

    void dumpPopulationToFile(const std::string& fileName) const;
    void readPopulationFromFile(const std::string& fileName);
    void dumpBestToFile(const std::string& fileName) const;

private:

    void buildPopulation();
    void evaluate();
    void segregate();
    void cross();
    void mutate();

    size_t m_genotypeSize;
    std::vector<std::string> m_gensName;
    std::vector<double> m_gensInitialValue;
    std::vector<bool> m_gensFix;
    std::vector<std::pair<double, double> > m_gensLimit;
    std::vector<double> m_gensStep;

    size_t m_populationSizeMin;   ///< Minimal size of population
    size_t m_populationSizeMax;   ///< Maximal size of population
    double m_populationSizeSlope; ///< Slope between transition between minimal and maximal size
    double m_bestFraction;  ///< Fraction of candidates kept for next population
    double m_crossSlope; ///< Cross exponential factor
    double m_mutationAProbMax; ///< Probability of mutation type A (small change of gen value)
    double m_mutationBProbMax; ///< Probability of mutation type B (new gen value)
    double m_mutationASize; ///< Width of normal distribution related to size of mutation type A (fraction of gen range)
    double m_mutationAResistance; ///< Fraction of candidates resistant to mutation type A
    double m_mutationBResistance; ///< Fraction of candidates resistant to mutation type B
    double m_mutationASlope; ///< Slope allowing to make mutation A time dependent
    double m_mutationBSlope; ///< Slope allowing to make mutation B time dependent

    size_t m_nIterationsActual; ///< Actual number of iterations
    size_t m_populationSizeActual; ///< Actual size of population
    double m_mutationAProbActual; ///< Actual probability of mutation type A (small change of gen value)
    double m_mutationBProbActual; ///< Actual probability of mutation type B (new gen value)

    std::vector<GeneticAlgorithmCandidate*> m_population; ///< population of candidates

    FitsService* m_fitsService;
    fcnType m_fcn;

    size_t m_stopConditionA;
    size_t m_stopConditionB1;
    double m_stopConditionB2;

    size_t m_dumpBestToFileEvery;
    std::string m_dumpBestToFilePath;
    size_t m_dumpPopulationToFileEvery;
    std::string m_dumpPopulationToFilePath;

    FitStatusType::Type m_fitStatus;
    double m_bestFCN;
    size_t m_nCalls;
    std::vector<double> m_bestParameters;
};

#endif /* GENETICALGORITHM_H_ */
