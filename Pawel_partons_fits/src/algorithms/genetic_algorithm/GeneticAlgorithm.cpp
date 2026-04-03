/*
 * GeneticAlgorithm.cpp
 *
 *  Created on: Dec 22, 2015
 *      Author: Pawel Sznajder
 */

#include "../../../include/algorithms/genetic_algorithm/GeneticAlgorithm.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>

#include "../../../include/algorithms/random_generator/RandomGenerator.h"

GeneticAlgorithm::GeneticAlgorithm() :
        BaseObject("GeneticAlgorithm") {

    m_genotypeSize = 0;
    m_gensName.clear();
    m_gensLimit.clear();
    m_gensStep.clear();

    m_populationSizeMin = 20;
    m_populationSizeMax = 200;
    m_populationSizeSlope = 0.1;
    m_bestFraction = 0.3;
    m_crossSlope = -1.;
    m_mutationAProbMax = 0.05;
    m_mutationBProbMax = 0.01;
    m_mutationASize = 0.01;
    m_mutationAResistance = 0.1;
    m_mutationBResistance = 0.2;
    m_mutationASlope = 0.;
    m_mutationBSlope = 0.;

    m_nIterationsActual = 1;
    m_populationSizeActual = 0;
    m_mutationAProbActual = 0.;
    m_mutationBProbActual = 0.;

    m_population.clear();

    m_fcn = 0;
    m_fitsService = 0;

    m_stopConditionA = 1E9;
    m_stopConditionB1 = 20;
    m_stopConditionB2 = 1.E-4;

    m_dumpBestToFileEvery = 0;
    m_dumpBestToFilePath = "";
    m_dumpPopulationToFileEvery = 0;
    m_dumpPopulationToFilePath = "";

    m_fitStatus = FitStatusType::UNDEFINED;
    m_bestFCN = 0.;
    m_nCalls = 0;
    m_bestParameters.clear();
}

GeneticAlgorithm::GeneticAlgorithm(const GeneticAlgorithm& other) :
        BaseObject(other) {

    m_genotypeSize = other.m_genotypeSize;
    m_gensName = other.m_gensName;
    m_gensLimit = other.m_gensLimit;
    m_gensStep = other.m_gensStep;

    m_populationSizeMin = other.m_populationSizeMin;
    m_populationSizeMax = other.m_populationSizeMax;
    m_populationSizeSlope = other.m_populationSizeSlope;
    m_bestFraction = other.m_bestFraction;
    m_crossSlope = other.m_crossSlope;
    m_mutationAProbMax = other.m_mutationAProbMax;
    m_mutationBProbMax = other.m_mutationBProbMax;
    m_mutationASize = other.m_mutationASize;
    m_mutationAResistance = other.m_mutationAResistance;
    m_mutationBResistance = other.m_mutationBResistance;
    m_mutationASlope = other.m_mutationASlope;
    m_mutationBSlope = other.m_mutationBSlope;

    m_nIterationsActual = other.m_nIterationsActual;
    m_populationSizeActual = other.m_populationSizeActual;
    m_mutationAProbActual = other.m_mutationAProbActual;
    m_mutationBProbActual = other.m_mutationBProbActual;

    m_population.clear();

    m_fcn = other.m_fcn;
    m_fitsService = other.m_fitsService;

    m_stopConditionA = other.m_stopConditionA;
    m_stopConditionB1 = other.m_stopConditionB1;
    m_stopConditionB2 = other.m_stopConditionB2;

    m_dumpBestToFileEvery = other.m_dumpBestToFileEvery;
    m_dumpBestToFilePath = other.m_dumpBestToFilePath;
    m_dumpPopulationToFileEvery = other.m_dumpPopulationToFileEvery;
    m_dumpPopulationToFilePath = other.m_dumpPopulationToFilePath;

    m_fitStatus = other.m_fitStatus;
    m_bestFCN = other.m_bestFCN;
    m_nCalls = other.m_nCalls;
    m_bestParameters = other.m_bestParameters;
}

GeneticAlgorithm::~GeneticAlgorithm() {

    //delete population
    std::vector<GeneticAlgorithmCandidate*>::iterator it;

    for (it = m_population.begin(); it != m_population.end(); it++) {

        if (*it != 0) {
            delete *it;
        }
    }

    m_population.clear();
}

size_t GeneticAlgorithm::addUnlimitedVariable(const std::string& name,
        double initialValue, size_t nSteps) {
    throw ElemUtils::CustomException(getClassName(), __func__,
            "Unable to set unlimited variable for genetic algorithm");
}

size_t GeneticAlgorithm::addLimitedVariable(const std::string& name,
        double initialValue, const std::pair<double, double>& limit,
        double step) {

    //check min and max
    if (limit.first >= limit.second) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Provided minimum is grater or equal provided maximum");
    }

    //check initial value
    if (initialValue < limit.first || initialValue > limit.second) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Initial value not in limits");
    }

    //check number of steps
    if (step <= 0.) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Step smaller or equal zero");
    }

    //add parameter
    m_gensName.push_back(name);
    m_gensInitialValue.push_back(initialValue);
    m_gensFix.push_back(false);
    m_gensLimit.push_back(limit);
    m_gensStep.push_back(step);

    //return
    return m_genotypeSize++;
}

size_t GeneticAlgorithm::addFixedVariable(const std::string& name,
        double value) {

    //add parameter
    m_gensName.push_back(name);
    m_gensInitialValue.push_back(value);
    m_gensFix.push_back(true);
    m_gensLimit.push_back(std::make_pair(value, value));
    m_gensStep.push_back(0.);

    //return
    return m_genotypeSize++;
}

void GeneticAlgorithm::setFCN(FitsService* fitsService, fcnType fcn) {

    m_fitsService = fitsService;
    m_fcn = fcn;
}

void GeneticAlgorithm::buildPopulation() {

    //check if different
    if (m_population.size() == m_populationSizeActual)
        return;

    //check if larger
    if (m_population.size() > m_populationSizeActual) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Population size if larger than that actual");
    }

    //rebuld
    for (size_t i = m_population.size(); i < m_populationSizeActual; i++) {
        m_population.push_back(new GeneticAlgorithmCandidate(m_gensLimit));
    }
}

void GeneticAlgorithm::run() {

    //filos
    std::queue<double> fifo;

    //iterations
    for (;;) {

        //info
        info(__func__,
                ElemUtils::Formatter() << "Start iteration "
                        << m_nIterationsActual);

        //time
        size_t timeActual = m_nIterationsActual - 1;

        //size of population
        m_populationSizeActual =
                size_t(
                        std::ceil(
                                m_populationSizeMax * m_populationSizeMin
                                        / (m_populationSizeMin
                                                + (m_populationSizeMax
                                                        - m_populationSizeMin)
                                                        * exp(
                                                                -1
                                                                        * m_populationSizeSlope
                                                                        * timeActual))));

        info(__func__,
                ElemUtils::Formatter() << "Actual population size is "
                        << m_populationSizeActual);

        //rebuild population if needed
        buildPopulation();

        //evaluate
        evaluate();

        //segregate
        segregate();

        //best fcn
        m_bestFCN = (*(m_population.begin()))->getFitness();

        //status
        info(__func__, ElemUtils::Formatter() << "fcn value: " << m_bestFCN);

        //dump
        if (m_dumpBestToFileEvery > 0) {
            if (m_nIterationsActual % m_dumpBestToFileEvery == 0)
                dumpBestToFile(m_dumpBestToFilePath);
        }

        if (m_dumpPopulationToFileEvery > 0) {
            if (m_nIterationsActual % m_dumpPopulationToFileEvery == 0)
                dumpPopulationToFile(m_dumpPopulationToFilePath);
        }

        //fifos
        fifo.push(m_bestFCN);
        if (m_nIterationsActual >= m_stopConditionB1)
            fifo.pop();

        //stop condition
        //A
        if (m_nIterationsActual == m_stopConditionA) {

            info(__func__, "Maximum number of iterations reached");
            m_fitStatus = FitStatusType::SUCCESSFUL;
            m_nCalls = m_nIterationsActual;
            m_bestParameters = m_population.at(0)->getGenotype();
            break;
        }

        //B
        if (m_nIterationsActual >= m_stopConditionB1) {

            if ((fifo.front() - fifo.back()) / fifo.front()
                    < m_stopConditionB2) {

                info(__func__, "Saturation reached");
                m_fitStatus = FitStatusType::SUCCESSFUL;
                m_nCalls = m_nIterationsActual;
                m_bestParameters = m_population.at(0)->getGenotype();
                break;
            }
        }

        //cross
        cross();

        //mutate
        m_mutationAProbActual = m_mutationAProbMax
                * exp(m_mutationASlope * timeActual);
        m_mutationBProbActual = m_mutationBProbMax
                * exp(m_mutationBSlope * timeActual);

        info(__func__,
                ElemUtils::Formatter() << "Actual mutation probability is "
                        << m_mutationAProbActual << " (A) and "
                        << m_mutationBProbActual << " (B)");

        mutate();

        //number of iterations
        m_nIterationsActual++;
    }
}

void GeneticAlgorithm::readPopulationFromFile(const std::string& fileName) {

    //open file
    std::fstream file(fileName.c_str(), std::ios::in | std::ios::binary);

    //check if open properly
    if (!file.is_open()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Can not open drop file for reading");
    }

    //status
    bool status = true;

    //genotype size
    size_t genptypeSize;

    //read
    status &= bool(file.read((char*) &genptypeSize, sizeof(size_t)));

    status &= bool(file.read((char*) &m_nIterationsActual, sizeof(size_t)));
    status &= bool(file.read((char*) &m_populationSizeActual, sizeof(size_t)));
    status &= bool(file.read((char*) &m_mutationAProbActual, sizeof(double)));
    status &= bool(file.read((char*) &m_mutationBProbActual, sizeof(double)));

    for (size_t i = 0; i < m_populationSizeActual; i++) {

        m_population.push_back(
                new GeneticAlgorithmCandidate(genptypeSize, file));
    }

    //close file
    file.close();

    if (!status) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Problem in reading from "
                        << fileName);
    }

    //status
    info(__func__,
            ElemUtils::Formatter() << "Population read from " << fileName);
}

void GeneticAlgorithm::dumpPopulationToFile(const std::string& fileName) const {

    //open file
    std::fstream file(fileName.c_str(),
            std::ios::out | std::ios::binary | std::ios::trunc);

    //check if open properly
    if (!file.is_open()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Can not open drop file for writing");
    }

    //status
    bool status = true;

    //write
    status &= bool(file.write((char*) &m_genotypeSize, sizeof(size_t)));

    status &= bool(file.write((char*) &m_nIterationsActual, sizeof(size_t)));
    status &= bool(file.write((char*) &m_populationSizeActual, sizeof(size_t)));
    status &= bool(file.write((char*) &m_mutationAProbActual, sizeof(double)));
    status &= bool(file.write((char*) &m_mutationBProbActual, sizeof(double)));

    std::vector<GeneticAlgorithmCandidate*>::const_iterator it;

    for (it = m_population.begin(); it != m_population.end(); it++) {
        status &= (*it)->writeToFile(file);
    }

    //close file
    file.close();

    //status
    if (!status) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Problem in dumping in " << fileName);
    }

    //status
    info(__func__,
            ElemUtils::Formatter() << "Population dumped to " << fileName);
}

void GeneticAlgorithm::dumpBestToFile(const std::string& fileName) const {

    //open
    std::fstream file(fileName.c_str(),
            std::ios::out | std::ios::binary | std::ios::app);

    //check if open properly
    if (!(file.is_open())) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Can not open best candidate drop file for writing");
    }

    //status
    bool status = true;

    //write
    status &= bool(file.write((char*) &m_genotypeSize, sizeof(size_t)));
    status &= (*(m_population.begin()))->writeToFile(file);

    //close file
    file.close();

    //status
    if (!status) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Problem in dumping in " << fileName);
    }

    //status
    info(__func__,
            ElemUtils::Formatter() << "Best candidate dumped to " << fileName);
}

void GeneticAlgorithm::evaluate() {

    //loop over population size
    for (int i = 0; i < m_populationSizeActual; i++) {

        //check if needs evaluation
        if (m_population.at(i)->getIsFitnessUpToDate())
            continue;

        //get genotype
        m_population.at(i)->setFitness(
                (m_fitsService->*m_fcn)(
                        &(m_population.at(i)->getGenotype()[0])));

        //mark as up to date
        m_population.at(i)->setIsFitnessUpToDate(true);
    }
}

void GeneticAlgorithm::segregate() {

    //loop over population size
    for (size_t i = 0; i < size_t(m_bestFraction * m_populationSizeActual);
            i++) {
        for (size_t j = i + 1; j < m_populationSizeActual; j++) {
            if (m_population.at(i)->getFitness()
                    > m_population.at(j)->getFitness())
                std::swap(m_population[i], m_population[j]);
        }
    }
}

void GeneticAlgorithm::cross() {

    //loop over population size
    for (size_t i = size_t(m_bestFraction * m_populationSizeActual);
            i < m_populationSizeActual; i++) {

        //dice parents
        size_t p1 = -1;
        size_t p2 = -1;

        while (p1 == p2) {

            if (m_crossSlope == 0.) {

                p1 = size_t(
                        RandomGenerator::getInstance()->diceFlat(0.,
                                size_t(
                                        m_bestFraction
                                                * m_populationSizeActual)));

                p2 = size_t(
                        RandomGenerator::getInstance()->diceFlat(0.,
                                size_t(
                                        m_bestFraction
                                                * m_populationSizeActual)));
            } else {

                p1 = size_t(
                        RandomGenerator::getInstance()->diceExp(0.,
                                size_t(m_bestFraction * m_populationSizeActual),
                                -1 * m_crossSlope / m_populationSizeActual));

                p2 = size_t(
                        RandomGenerator::getInstance()->diceExp(0.,
                                size_t(m_bestFraction * m_populationSizeActual),
                                -1 * m_crossSlope / m_populationSizeActual));
            }
        }

        //mix and copy genotypes of parents
        for (size_t j = 0; j < m_genotypeSize; j++) {

            if (m_gensFix.at(j))
                continue;

            m_population.at(i)->modifyGen(j,
                    (RandomGenerator::getInstance()->diceFlat() < 0.5) ?
                            (m_population.at(p1)->getGenotype()[j]) :
                            (m_population.at(p2)->getGenotype()[j]));
        }

        //set chi2 actual
        m_population.at(i)->setIsFitnessUpToDate(false);
    }
}

void GeneticAlgorithm::mutate() {

    //loop over population size
    for (size_t i = 0; i < m_populationSizeActual; i++) {

        //mutation type A
        if (i > size_t(m_mutationAResistance * m_populationSizeActual)) {

            //loop over genes
            for (size_t j = 0; j < m_genotypeSize; j++) {

                //check if not fixed
                if (m_gensFix.at(j))
                    continue;

                //mutate
                if (RandomGenerator::getInstance()->diceFlat()
                        < m_mutationAProbActual) {
                    m_population.at(i)->modifyGen(j,
                            RandomGenerator::getInstance()->diceNormal(
                                    (m_population.at(i)->getGenotype()).at(j),
                                    m_mutationASize
                                            * (m_gensLimit.at(j).second
                                                    - m_gensLimit.at(j).first)));
                    m_population.at(i)->setIsFitnessUpToDate(false);
                }
            }
        }

        //mutation type B
        if (i > size_t(m_mutationBResistance * m_populationSizeActual)) {

            //loop over genes
            for (size_t j = 0; j < m_genotypeSize; j++) {

                //check if not fixed
                if (m_gensFix.at(j))
                    continue;

                //mutate
                if (RandomGenerator::getInstance()->diceFlat()
                        < m_mutationBProbActual) {
                    m_population.at(i)->modifyGen(j,
                            RandomGenerator::getInstance()->diceFlat(
                                    m_gensLimit.at(j).first,
                                    m_gensLimit.at(j).second));
                    m_population.at(i)->setIsFitnessUpToDate(false);
                }
            }
        }
    }
}

double GeneticAlgorithm::getBestFraction() const {
    return m_bestFraction;
}

void GeneticAlgorithm::setBestFraction(double bestFraction) {
    m_bestFraction = bestFraction;
}

double GeneticAlgorithm::getCrossSlope() const {
    return m_crossSlope;
}

void GeneticAlgorithm::setCrossSlope(double crossSlope) {
    m_crossSlope = crossSlope;
}

double GeneticAlgorithm::getMutationAProbMax() const {
    return m_mutationAProbMax;
}

void GeneticAlgorithm::setMutationAProbMax(double mutationAProbMax) {
    m_mutationAProbMax = mutationAProbMax;
}

double GeneticAlgorithm::getMutationAResistance() const {
    return m_mutationAResistance;
}

void GeneticAlgorithm::setMutationAResistance(double mutationAResistance) {
    m_mutationAResistance = mutationAResistance;
}

double GeneticAlgorithm::getMutationASize() const {
    return m_mutationASize;
}

void GeneticAlgorithm::setMutationASize(double mutationASize) {
    m_mutationASize = mutationASize;
}

double GeneticAlgorithm::getMutationASlope() const {
    return m_mutationASlope;
}

void GeneticAlgorithm::setMutationASlope(double mutationASlope) {
    m_mutationASlope = mutationASlope;
}

double GeneticAlgorithm::getMutationBProbMax() const {
    return m_mutationBProbMax;
}

void GeneticAlgorithm::setMutationBProbMax(double mutationBProbMax) {
    m_mutationBProbMax = mutationBProbMax;
}

double GeneticAlgorithm::getMutationBResistance() const {
    return m_mutationBResistance;
}

void GeneticAlgorithm::setMutationBResistance(double mutationBResistance) {
    m_mutationBResistance = mutationBResistance;
}

double GeneticAlgorithm::getMutationBSlope() const {
    return m_mutationBSlope;
}

void GeneticAlgorithm::setMutationBSlope(double mutationBSlope) {
    m_mutationBSlope = mutationBSlope;
}

size_t GeneticAlgorithm::getPopulationSizeMax() const {
    return m_populationSizeMax;
}

void GeneticAlgorithm::setPopulationSizeMax(size_t populationSizeMax) {
    m_populationSizeMax = populationSizeMax;
}

size_t GeneticAlgorithm::getPopulationSizeMin() const {
    return m_populationSizeMin;
}

void GeneticAlgorithm::setPopulationSizeMin(size_t populationSizeMin) {
    m_populationSizeMin = populationSizeMin;
}

double GeneticAlgorithm::getPopulationSizeSlope() const {
    return m_populationSizeSlope;
}

void GeneticAlgorithm::setPopulationSizeSlope(double populationSizeSlope) {
    m_populationSizeSlope = populationSizeSlope;
}

size_t GeneticAlgorithm::getStopConditionA() const {
    return m_stopConditionA;
}

void GeneticAlgorithm::setStopConditionA(size_t stopConditionA) {
    m_stopConditionA = stopConditionA;
}

size_t GeneticAlgorithm::getStopConditionB1() const {
    return m_stopConditionB1;
}

void GeneticAlgorithm::setStopConditionB1(size_t stopConditionB1) {
    m_stopConditionB1 = stopConditionB1;
}

double GeneticAlgorithm::getStopConditionB2() const {
    return m_stopConditionB2;
}

void GeneticAlgorithm::setStopConditionB2(double stopConditionB2) {
    m_stopConditionB2 = stopConditionB2;
}

double GeneticAlgorithm::getBestFcn() const {
    return m_bestFCN;
}

FitStatusType::Type GeneticAlgorithm::getFitStatus() const {
    return m_fitStatus;
}

size_t GeneticAlgorithm::getNCalls() const {
    return m_nCalls;
}

const std::vector<double>& GeneticAlgorithm::getBestParameters() const {
    return m_bestParameters;
}

size_t GeneticAlgorithm::getDumpBestToFileEvery() const {
    return m_dumpBestToFileEvery;
}

void GeneticAlgorithm::setDumpBestToFileEvery(size_t dumpBestToFileEvery) {
    m_dumpBestToFileEvery = dumpBestToFileEvery;
}

const std::string& GeneticAlgorithm::getDumpBestToFilePath() const {
    return m_dumpBestToFilePath;
}

void GeneticAlgorithm::setDumpBestToFilePath(
        const std::string& dumpBestToFilePath) {
    m_dumpBestToFilePath = dumpBestToFilePath;
}

size_t GeneticAlgorithm::getDumpPopulationToFileEvery() const {
    return m_dumpPopulationToFileEvery;
}

void GeneticAlgorithm::setDumpPopulationToFileEvery(
        size_t dumpPopulationToFileEvery) {
    m_dumpPopulationToFileEvery = dumpPopulationToFileEvery;
}

const std::string& GeneticAlgorithm::getDumpPopulationToFilePath() const {
    return m_dumpPopulationToFilePath;
}

void GeneticAlgorithm::setDumpPopulationToFilePath(
        const std::string& dumpPopulationToFilePath) {
    m_dumpPopulationToFilePath = dumpPopulationToFilePath;
}
