/*
 * RandomGenerator.cpp
 *
 *  Created on: Aug 1, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../include/algorithms/random_generator/RandomGenerator.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/logger/LoggerManager.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

RandomGenerator* RandomGenerator::m_pInstance = 0;

RandomGenerator::RandomGenerator() {

    //set
    m_seed = time(0);

    //establish
    srand(m_seed);

    //info
    ElemUtils::LoggerManager::getInstance()->info("RandomGenerator", __func__,
            ElemUtils::Formatter() << "Initial random seed set to " << m_seed);
}

RandomGenerator::~RandomGenerator() {

    if (m_pInstance) {
        delete m_pInstance;
        m_pInstance = 0;
    }
}

RandomGenerator* RandomGenerator::getInstance() {

    if (!m_pInstance) {
        m_pInstance = new RandomGenerator();
    }

    return m_pInstance;
}

size_t RandomGenerator::setSeed(size_t newSeed) {

    //set
    if (newSeed == 0) {
        m_seed = time(0);
    } else {
        m_seed = newSeed;
    }

    //establish
    srand(m_seed);

    //info
    ElemUtils::LoggerManager::getInstance()->info("RandomGenerator", __func__,
            ElemUtils::Formatter() << "Random seed set to " << m_seed);

    //return
    return m_seed;
}

size_t RandomGenerator::getSeed() const {
    return m_seed;
}

double RandomGenerator::diceFlat() const {
    return rand() / double(RAND_MAX);
}

double RandomGenerator::diceFlat(double min, double max) const {

    if (min >= max) {
        throw ElemUtils::CustomException("RandomGenerator", __func__,
                ElemUtils::Formatter() << "Illegal values of min = " << min
                        << " and max = " << max);
    }

    return min + (max - min) * diceFlat();
}

double RandomGenerator::diceExp() const {
    return diceExp(1.);
}

double RandomGenerator::diceNormal() const {

    //from http://www.design.caltech.edu/erik/Misc/Gaussian.html

    double x1, x2, w;

    do {
        x1 = 2 * diceFlat() - 1.;
        x2 = 2 * diceFlat() - 1.;
        w = x1 * x1 + x2 * x2;
    } while (w >= 1.);

    w = sqrt((-2. * log(w)) / w);

    return x1 * w;
}

double RandomGenerator::diceNormal(double mean, double sigma) const {

    if (sigma <= 0.) {
        throw ElemUtils::CustomException("RandomGenerator", __func__,
                ElemUtils::Formatter() << "Illegal value of sigma = " << sigma);
    }

    return mean + sigma * diceNormal();
}

double RandomGenerator::diceExp(double lambda) const {
    return -1 * log(1. - diceFlat()) / lambda;
}

double RandomGenerator::diceExp(double min, double max, double lambda) const {

    double minFlat = 1. - exp(-1 * lambda * min);
    double maxFlat = 1. - exp(-1 * lambda * max);

    return -1 * log(1. - diceFlat(minFlat, maxFlat)) / lambda;
}
