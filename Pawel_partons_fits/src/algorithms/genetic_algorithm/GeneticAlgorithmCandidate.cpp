/*
 * GeneticAlgorithmCandidate.cpp
 *
 *  Created on: Dec 22, 2015
 *      Author: Pawel Sznajder
 */

#include "../../../include/algorithms/genetic_algorithm/GeneticAlgorithmCandidate.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <iterator>

#include "../../../include/algorithms/random_generator/RandomGenerator.h"

GeneticAlgorithmCandidate::GeneticAlgorithmCandidate() :
        BaseObject("GeneticAlgorithmCandidate") {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Constructor not supporter");

    m_fitness = -1.;
    m_fitnessIsUpToDate = false;
    m_genotype.clear();
    m_genotypeLimits.clear();
}

GeneticAlgorithmCandidate::GeneticAlgorithmCandidate(
        const GeneticAlgorithmCandidate& other) :
        BaseObject(other) {

    m_fitness = other.m_fitness;
    m_fitnessIsUpToDate = other.m_fitnessIsUpToDate;
    m_genotype = other.m_genotype;
    m_genotypeLimits = other.m_genotypeLimits;
}

GeneticAlgorithmCandidate::GeneticAlgorithmCandidate(
        const std::vector<std::pair<double, double> >& limits) :
        BaseObject("GeneticAlgorithmCandidate") {

    m_fitness = -1.;
    m_fitnessIsUpToDate = false;

    std::vector<std::pair<double, double> >::const_iterator it;

    for (it = limits.begin(); it != limits.end(); it++) {

        if (it->first == it->second) {

            m_genotype.push_back(it->first);

        } else {

            m_genotype.push_back(
                    RandomGenerator::getInstance()->diceFlat(it->first,
                            it->second));
        }
    }

    m_genotypeLimits = limits;
}

GeneticAlgorithmCandidate::GeneticAlgorithmCandidate(size_t genotypeSize,
        std::fstream& file) :
        BaseObject("GeneticAlgorithmCandidate") {

    if (!file.is_open()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "File not open");
    }

    bool status = true;

    status &= bool(file.read((char*) &m_fitness, sizeof(double)));
    status &= bool(file.read((char*) &m_fitnessIsUpToDate, sizeof(bool)));

    m_genotype.clear();

    for (size_t i = 0; i < genotypeSize; i++) {

        double valueDouble;

        status &= bool(file.read((char*) &valueDouble, sizeof(double)));
        m_genotype.push_back(valueDouble);
    }

    for (size_t i = 0; i < genotypeSize; i++) {

        double valueDouble1, valueDouble2;

        status &= bool(file.read((char*) &valueDouble1, sizeof(double)));
        status &= bool(file.read((char*) &valueDouble2, sizeof(double)));
        m_genotypeLimits.push_back(std::make_pair(valueDouble1, valueDouble2));
    }
}

GeneticAlgorithmCandidate::~GeneticAlgorithmCandidate() {
}

double GeneticAlgorithmCandidate::getFitness() const {
    return m_fitness;
}

void GeneticAlgorithmCandidate::setFitness(double fitness) {
    m_fitness = fitness;
}

bool GeneticAlgorithmCandidate::getIsFitnessUpToDate() const {
    return m_fitnessIsUpToDate;
}

void GeneticAlgorithmCandidate::setIsFitnessUpToDate(bool upToDate) {
    m_fitnessIsUpToDate = upToDate;
}

const std::vector<double>& GeneticAlgorithmCandidate::getGenotype() const {
    return m_genotype;
}

const std::vector<std::pair<double, double> >& GeneticAlgorithmCandidate::getGenotypeLimits() const {
    return m_genotypeLimits;
}

void GeneticAlgorithmCandidate::modifyGen(size_t gen, double value) {

    if (gen >= m_genotype.size()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Illegal gen id");
    }

    if (value < m_genotypeLimits.at(gen).first) {
        warn(__func__,
                ElemUtils::Formatter() << "New value " << value
                        << " smaller than limit "
                        << m_genotypeLimits.at(gen).first);
        return;
    }

    if (value > m_genotypeLimits.at(gen).second) {
        warn(__func__,
                ElemUtils::Formatter() << "New value " << value
                        << " greater than limit "
                        << m_genotypeLimits.at(gen).second);
        return;
    }

    m_genotype.at(gen) = value;
}

bool GeneticAlgorithmCandidate::writeToFile(std::fstream& file) const {

    if (!file.is_open()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "File not open");
    }

    bool status = true;

    status &= bool(file.write((char*) &m_fitness, sizeof(double)));
    status &= bool(file.write((char*) &m_fitnessIsUpToDate, sizeof(bool)));

    std::vector<double>::const_iterator itGen;

    for (itGen = m_genotype.begin(); itGen != m_genotype.end(); itGen++) {

        double valueDouble;

        valueDouble = *itGen;
        status &= bool(file.write((char*) &valueDouble, sizeof(double)));
    }

    std::vector<std::pair<double, double> >::const_iterator itGenLimit;

    for (itGenLimit = m_genotypeLimits.begin();
            itGenLimit != m_genotypeLimits.end(); itGenLimit++) {

        double valueDouble1, valueDouble2;

        valueDouble1 = itGenLimit->first;
        valueDouble2 = itGenLimit->second;

        status &= bool(file.write((char*) &valueDouble1, sizeof(double)));
        status &= bool(file.write((char*) &valueDouble2, sizeof(double)));
    }

    return status;
}
