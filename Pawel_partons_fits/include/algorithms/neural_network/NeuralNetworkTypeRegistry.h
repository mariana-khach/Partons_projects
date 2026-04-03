/*
 * NeuralNetworkTypeRegistry.h
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#ifndef NEURAL_NETWORK_TYPE_REGISTRY_H_
#define NEURAL_NETWORK_TYPE_REGISTRY_H_

#include <map>
#include <string>

#include "beans/ActivationFunctionType.h"
#include "beans/CombinationFunctionType.h"
#include "beans/ScalingFunctionType.h"
#include "beans/TrainingFunctionType.h"

namespace PARTONS {
class BaseObjectFactory;
} /* namespace PARTONS */
class TrainingFunction;
class ScalingFunction;
class CombinationFunction;
class ActivationFunction;

class NeuralNetworkTypeRegistry: public PARTONS::BaseObject {

public:

    ~NeuralNetworkTypeRegistry();

    static NeuralNetworkTypeRegistry* getInstance();

    unsigned int registerActivationFunction(
            const ActivationFunctionType::Type type,
            ActivationFunction* const object);
    unsigned int registerCombinationFunction(
            const CombinationFunctionType::Type type,
            CombinationFunction* const object);
    unsigned int registerScalingFunction(const ScalingFunctionType::Type type,
            ScalingFunction* const object);
    unsigned int registerTrainingFunction(const TrainingFunctionType::Type type,
            TrainingFunction* const object);

    ActivationFunction* getActivationFunction(
            const ActivationFunctionType::Type type) const;
    CombinationFunction* getCombinationFunction(
            const CombinationFunctionType::Type type) const;
    ScalingFunction* getScalingFunction(
            const ScalingFunctionType::Type type) const;
    TrainingFunction* getTrainingFunction(
            const TrainingFunctionType::Type type) const;

private:

    static NeuralNetworkTypeRegistry* p_instance;

    NeuralNetworkTypeRegistry();

    PARTONS::BaseObjectFactory* m_pBaseObjectFactory;

    std::map<ActivationFunctionType::Type, std::string> m_activationFunctions;
    std::map<CombinationFunctionType::Type, std::string> m_combinationFunctions;
    std::map<ScalingFunctionType::Type, std::string> m_scalingFunctions;
    std::map<TrainingFunctionType::Type, std::string> m_trainingFunctions;
};

#endif /* NEURAL_NETWORK_TYPE_REGISTRY_H_ */
