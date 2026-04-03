/*
 * NeuralNetworkTypeRegistry.cpp
 *
 *  Created on: Apr 23, 2016
 *      Author: Pawel Sznajder
 */

#include "../../../include/algorithms/neural_network/NeuralNetworkTypeRegistry.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObjectFactory.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/Partons.h>
#include <utility>

#include "../../../include/algorithms/neural_network/activation_function/ActivationFunction.h"
#include "../../../include/algorithms/neural_network/combination_function/CombinationFunction.h"
#include "../../../include/algorithms/neural_network/scaling_function/ScalingFunction.h"
#include "../../../include/algorithms/neural_network/training_function/TrainingFunction.h"

using namespace PARTONS;

NeuralNetworkTypeRegistry::NeuralNetworkTypeRegistry() :
        BaseObject("NeuralNetworkTypeRegistry") {

    m_pBaseObjectFactory = Partons::getInstance()->getBaseObjectFactory();

    m_activationFunctions.clear();
    m_combinationFunctions.clear();
    m_scalingFunctions.clear();
    m_trainingFunctions.clear();
}

NeuralNetworkTypeRegistry::~NeuralNetworkTypeRegistry() {

    if (p_instance) {
        delete p_instance;
        p_instance = 0;
    }
}

NeuralNetworkTypeRegistry* NeuralNetworkTypeRegistry::p_instance = 0;

NeuralNetworkTypeRegistry* NeuralNetworkTypeRegistry::getInstance() {

    if (p_instance == 0) {
        p_instance = new NeuralNetworkTypeRegistry();
    }

    return p_instance;
}

unsigned int NeuralNetworkTypeRegistry::registerActivationFunction(
        const ActivationFunctionType::Type type,
        ActivationFunction* const object) {

    if (type == ActivationFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    if (m_activationFunctions.find(type) != m_activationFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Object already registered");
    }

    m_activationFunctions.insert(std::make_pair(type, object->getClassName()));

    return BaseObjectRegistry::getInstance()->registerBaseObject(object);
}

unsigned int NeuralNetworkTypeRegistry::registerCombinationFunction(
        const CombinationFunctionType::Type type,
        CombinationFunction* const object) {

    if (type == CombinationFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    if (m_combinationFunctions.find(type) != m_combinationFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Object already registered");
    }

    m_combinationFunctions.insert(std::make_pair(type, object->getClassName()));

    return BaseObjectRegistry::getInstance()->registerBaseObject(object);
}

unsigned int NeuralNetworkTypeRegistry::registerScalingFunction(
        const ScalingFunctionType::Type type, ScalingFunction* const object) {

    if (type == ScalingFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    if (m_scalingFunctions.find(type) != m_scalingFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Object already registered");
    }

    m_scalingFunctions.insert(std::make_pair(type, object->getClassName()));

    return BaseObjectRegistry::getInstance()->registerBaseObject(object);
}

unsigned int NeuralNetworkTypeRegistry::registerTrainingFunction(
        const TrainingFunctionType::Type type, TrainingFunction* const object) {

    if (type == TrainingFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    if (m_trainingFunctions.find(type) != m_trainingFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Object already registered");
    }

    m_trainingFunctions.insert(std::make_pair(type, object->getClassName()));

    return BaseObjectRegistry::getInstance()->registerBaseObject(object);
}

ActivationFunction* NeuralNetworkTypeRegistry::getActivationFunction(
        const ActivationFunctionType::Type type) const {

    if (type == ActivationFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    std::map<ActivationFunctionType::Type, std::string>::const_iterator it =
            m_activationFunctions.find(type);

    if (it == m_activationFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Object not registered, type "
                        << ActivationFunctionType(type).toString());
    }

    return static_cast<ActivationFunction*>(m_pBaseObjectFactory->newBaseObject(
            it->second));
}

CombinationFunction* NeuralNetworkTypeRegistry::getCombinationFunction(
        const CombinationFunctionType::Type type) const {

    if (type == CombinationFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    std::map<CombinationFunctionType::Type, std::string>::const_iterator it =
            m_combinationFunctions.find(type);

    if (it == m_combinationFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Object not registered, type "
                        << CombinationFunctionType(type).toString());
    }

    return static_cast<CombinationFunction*>(m_pBaseObjectFactory->newBaseObject(
            it->second));
}

ScalingFunction* NeuralNetworkTypeRegistry::getScalingFunction(
        const ScalingFunctionType::Type type) const {

    if (type == ScalingFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    std::map<ScalingFunctionType::Type, std::string>::const_iterator it =
            m_scalingFunctions.find(type);

    if (it == m_scalingFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Object not registered, type "
                        << ScalingFunctionType(type).toString());
    }

    return static_cast<ScalingFunction*>(m_pBaseObjectFactory->newBaseObject(
            it->second));
}

TrainingFunction* NeuralNetworkTypeRegistry::getTrainingFunction(
        const TrainingFunctionType::Type type) const {

    if (type == TrainingFunctionType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined type");
    }

    std::map<TrainingFunctionType::Type, std::string>::const_iterator it =
            m_trainingFunctions.find(type);

    if (it == m_trainingFunctions.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Object not registered, type "
                        << TrainingFunctionType(type).toString());
    }

    return static_cast<TrainingFunction*>(m_pBaseObjectFactory->newBaseObject(
            it->second));
}
