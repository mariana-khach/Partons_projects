/*
 * GeneticAlgorithmCandidate.h
 *
 *  Created on: Dec 22, 2015
 *      Author: Pawel Sznajder
 */

#ifndef GENETICALGORITHMCANDIDATE_H_
#define GENETICALGORITHMCANDIDATE_H_

#include <partons/BaseObject.h>
#include <stddef.h>
#include <fstream>
#include <utility>
#include <vector>

class GeneticAlgorithmCandidate: public PARTONS::BaseObject {

public:

    GeneticAlgorithmCandidate();
    GeneticAlgorithmCandidate(const GeneticAlgorithmCandidate& other);
    GeneticAlgorithmCandidate(
            const std::vector<std::pair<double, double> >& limits);
    GeneticAlgorithmCandidate(size_t genotypeSize, std::fstream& file);
    ~GeneticAlgorithmCandidate();

    double getFitness() const;
    void setFitness(double fitness);

    bool getIsFitnessUpToDate() const;
    void setIsFitnessUpToDate(bool upToDate);

    const std::vector<double>& getGenotype() const;
    const std::vector<std::pair<double, double> >& getGenotypeLimits() const;

    void modifyGen(size_t gen, double value);
    bool writeToFile(std::fstream& file) const;

private:

    double m_fitness;   ///< Value of fitness function
    bool m_fitnessIsUpToDate; ///< True if assign value of fitness function is actual for genotype
    std::vector<double> m_genotype;   ///< Genotype
    std::vector<std::pair<double, double> > m_genotypeLimits;   ///< Genotype limits
};

#endif /* GENETICALGORITHMCANDIDATE_H_ */
