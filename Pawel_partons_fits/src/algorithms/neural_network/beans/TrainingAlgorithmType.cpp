/*
 * TrainingType.cpp
 *
 *  Created on: Jul 28, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/beans/TrainingAlgorithmType.h"

TrainingAlgorithmType::TrainingAlgorithmType() :
        BaseObject("TrainingAlgorithmType") {
    m_type = TrainingAlgorithmType::UNDEFINED;
}

TrainingAlgorithmType::TrainingAlgorithmType(TrainingAlgorithmType::Type type) :
        BaseObject("TrainingAlgorithmType") {
    m_type = type;
}

TrainingAlgorithmType::TrainingAlgorithmType(
        const TrainingAlgorithmType& other) :
        BaseObject(other) {
    m_type = other.m_type;
}

TrainingAlgorithmType::~TrainingAlgorithmType() {
}

TrainingAlgorithmType* TrainingAlgorithmType::clone() const {
    return new TrainingAlgorithmType(*this);
}

std::string TrainingAlgorithmType::toString() const {

    switch (m_type) {

    case TrainingAlgorithmType::UNDEFINED: {
        return "UNDEFINED";
    }

        break;

    case TrainingAlgorithmType::BackPropagation: {
        return "BackPropagation";
    }

        break;

    case TrainingAlgorithmType::GeneticAlgorithm: {
        return "GeneticAlgorithm";
    }

        break;

    default: {
        return "UNDEFINED";
    }

        break;
    }
}
