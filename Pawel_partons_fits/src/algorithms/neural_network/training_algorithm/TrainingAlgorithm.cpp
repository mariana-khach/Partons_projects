/*
 * TrainingAlgorithm.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../../include/algorithms/neural_network/training_algorithm/TrainingAlgorithm.h"

#include <ElementaryUtils/logger/CustomException.h>

TrainingAlgorithm::TrainingAlgorithm() :
        BaseObject("TrainingAlgorithm") {
}

TrainingAlgorithm::TrainingAlgorithm(const std::string& name) :
        BaseObject(name) {
}

TrainingAlgorithm::TrainingAlgorithm(const TrainingAlgorithm& other) :
        BaseObject(other) {
}

TrainingAlgorithm::~TrainingAlgorithm() {
}

TrainingAlgorithm* TrainingAlgorithm::clone() const {
    return new TrainingAlgorithm(*this);
}

void TrainingAlgorithm::train(const NeuralNetwork& nn, const Data& input,
        const Data& output, TrainingFunctionType::Type trainingFunctionType,
        const std::vector<double>& parameters) {

    throw ElemUtils::CustomException(getClassName(), __func__,
            "Action for mother class undefined");
}
