/*
 * RandomGenerator.h
 *
 *  Created on: Aug 1, 2016
 *      Author: Pawel Sznajder
 */

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <stddef.h>

class RandomGenerator {

public:

    /**
     * Destructor
     */
    virtual ~RandomGenerator();

    /**
     * Share a unique pointer of this class
     */
    static RandomGenerator* getInstance();

    /**
     * Set new seed
     * @param seed '0' for time-based seed or fixed seed
     * @return Value of new seed
     */
    size_t setSeed(size_t seed);

    /**
     * Get seed
     */
    size_t getSeed() const;

    /**
     * Dice double precision number from flat distribution in range [0, 1)
     */
    double diceFlat() const;

    /**
     * Dice double precision number from flat distribution in given range
     * @param min Min
     * @param max Max
     */
    double diceFlat(double min, double max) const;

    /**
     * Dice double precision number from normal distribution with mean = 0 and sigma = 1
     */
    double diceNormal() const;

    /**
     * Dice double precision number from normal distribution with given mean and sigma
     * @param mean Mean
     * @param sigma Sigma
     */
    double diceNormal(double mean, double sigma) const;

    /**
     * Dice double precision number from exponential distribution with lambda (negative slope) = 1
     */
    double diceExp() const;

    /**
     * Dice double precision number from exponential distribution with given lambda (negative slope)
     */
    double diceExp(double lambda) const;

    /**
     * Dice double precision number from exponential distribution with given lambda (negative slope) in given range
     */
    double diceExp(double min, double max, double lambda) const;

private:

    /**
     * Private pointer of this class for a unique instance
     */
    static RandomGenerator* m_pInstance;

    /**
     * Default constructor
     */
    RandomGenerator();

    /**
     * Seed
     */
    size_t m_seed;
};

#endif /* RANDOMGENERATOR_H_ */
