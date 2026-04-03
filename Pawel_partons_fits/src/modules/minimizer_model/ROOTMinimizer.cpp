#include "../../../include/modules/minimizer_model/ROOTMinimizer.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/GenericType.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>
#include <NumA/linear_algebra/matrix/MatrixD.h>
#include <partons/BaseObjectRegistry.h>

using namespace PARTONS;

const std::string ROOTMinimizer::PARAMETER_NAME_PACKAGE_NAME =
        "root_minimizer_package_name";
const std::string ROOTMinimizer::PARAMETER_NAME_ALGORITHM_NAME =
        "root_minimizer_algorithm_name";
const std::string ROOTMinimizer::PARAMETER_NAME_PRINT_LEVEL =
        "root_minimizer_print_level";
const std::string ROOTMinimizer::PARAMETER_NAME_TOLERANCE =
        "root_minimizer_fit_tolerance";
const std::string ROOTMinimizer::PARAMETER_NAME_MAX_ITERATIONS =
        "root_minimizer_max_iterations";
const std::string ROOTMinimizer::PARAMETER_NAME_MAX_FCN_CALLS =
        "root_minimizer_max_fcn_calls";

const size_t ROOTMinimizer::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new ROOTMinimizer("ROOTMinimizer"));

ROOTMinimizer::ROOTMinimizer(const std::string& className) :
        FitsMinimizerModule(className), m_pMinimizer(0), m_fcnFunctor(), m_packageType(
                ROOTMinimizerPackageType::Default), m_algorithmType(
                ROOTMinimizerAlgorithmType::Default) {
    setNewMinimizer();
}

ROOTMinimizer::ROOTMinimizer(const ROOTMinimizer& other) :
        FitsMinimizerModule(other), m_pMinimizer(0) {

    m_fcnFunctor = other.m_fcnFunctor;

    m_packageType = other.m_packageType;
    m_algorithmType = other.m_algorithmType;

    setNewMinimizer();
}

ROOTMinimizer::~ROOTMinimizer() {

    if (m_pMinimizer) {
        delete m_pMinimizer;
        m_pMinimizer = 0;
    }
}

void ROOTMinimizer::resolveObjectDependencies() {
    FitsMinimizerModule::resolveObjectDependencies();
}

void ROOTMinimizer::configure(const ElemUtils::Parameters& parameters) {

    FitsMinimizerModule::configure(parameters);

    bool newMinimizerRequired = false;

    if (parameters.isAvailable(ROOTMinimizer::PARAMETER_NAME_PACKAGE_NAME)) {

        m_packageType = ROOTMinimizerPackageType(
                parameters.getLastAvailable().getString()).getType();

        info(__func__,
                ElemUtils::Formatter() << "Package changed to: "
                        << parameters.getLastAvailable().getString());

        newMinimizerRequired = true;
    }

    if (parameters.isAvailable(ROOTMinimizer::PARAMETER_NAME_ALGORITHM_NAME)) {

        m_algorithmType = ROOTMinimizerAlgorithmType(
                parameters.getLastAvailable().getString()).getType();

        info(__func__,
                ElemUtils::Formatter() << "Algorithm changed to: "
                        << parameters.getLastAvailable().getString());

        newMinimizerRequired = true;
    }

    if (newMinimizerRequired)
        setNewMinimizer();

    if (parameters.isAvailable(ROOTMinimizer::PARAMETER_NAME_PRINT_LEVEL)) {

        m_pMinimizer->SetPrintLevel(parameters.getLastAvailable().toInt());

        info(__func__,
                ElemUtils::Formatter() << "Print level changed to: "
                        << parameters.getLastAvailable().toInt());
    }

    if (parameters.isAvailable(ROOTMinimizer::PARAMETER_NAME_TOLERANCE)) {

        m_pMinimizer->SetTolerance(parameters.getLastAvailable().toDouble());

        info(__func__,
                ElemUtils::Formatter() << "Fit tolerance changed to: "
                        << parameters.getLastAvailable().toDouble());
    }

    if (parameters.isAvailable(ROOTMinimizer::PARAMETER_NAME_MAX_ITERATIONS)) {

        m_pMinimizer->SetMaxIterations(parameters.getLastAvailable().toInt());

        info(__func__,
                ElemUtils::Formatter()
                        << "Max number of iterations changed to: "
                        << parameters.getLastAvailable().toInt());
    }

    if (parameters.isAvailable(ROOTMinimizer::PARAMETER_NAME_MAX_FCN_CALLS)) {

        m_pMinimizer->SetMaxFunctionCalls(
                parameters.getLastAvailable().toInt());

        info(__func__,
                ElemUtils::Formatter() << "Max number of FCN calls changed to: "
                        << parameters.getLastAvailable().toInt());
    }
}

ROOTMinimizer* ROOTMinimizer::clone() const {
    return new ROOTMinimizer(*this);
}

//TODO not called anywhere
void ROOTMinimizer::initModule() {
    FitsMinimizerModule::initModule();
}

//TODO not called anywhere
void ROOTMinimizer::isModuleWellConfigured() {
    FitsMinimizerModule::isModuleWellConfigured();
}

size_t ROOTMinimizer::SetVariable(const std::string& parName,
        double initialValue, double initialStep) {

    size_t i = FitsMinimizerModule::SetVariable(parName, initialValue,
            initialStep);
    m_pMinimizer->SetVariable(i, parName, initialValue, initialStep);
    return i;
}

size_t ROOTMinimizer::SetLimitedVariable(const std::string& parName,
        double initialValue, double initialStep,
        const std::pair<double, double>& limits) {

    size_t i = FitsMinimizerModule::SetLimitedVariable(parName, initialValue,
            initialStep, limits);
    m_pMinimizer->SetLimitedVariable(i, parName, initialValue, initialStep,
            limits.first, limits.second);
    return i;
}

size_t ROOTMinimizer::SetFixedVariable(const std::string& parName,
        double initialValue) {

    size_t i = FitsMinimizerModule::SetFixedVariable(parName, initialValue);
    m_pMinimizer->SetFixedVariable(i, parName, initialValue);
    return i;
}

double ROOTMinimizer::getVariableValue(size_t i) const {

    if (i >= getNVariables()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Illegal index");
    }

    return m_pMinimizer->X()[i];
}

double ROOTMinimizer::getVariableValue(const std::string& varName) const {
    return getVariableValue(getVariableIndex(varName));
}

double ROOTMinimizer::getVariableError(size_t i) const {

    if (i >= getNVariables()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Illegal index");
    }

    return m_pMinimizer->Errors()[i];
}

double ROOTMinimizer::getVariableError(const std::string& varName) const {
    getVariableError(getVariableIndex(varName));
}

double ROOTMinimizer::getFCNValue() const {
    return m_pMinimizer->MinValue();
}

FitStatusType::Type ROOTMinimizer::getFitStatus() const {

    int statusCode = m_pMinimizer->Status();

    switch (statusCode) {

    case 0:
        return FitStatusType::SUCCESSFUL;
        break;

    default:
        return FitStatusType::FAILED;
    }
}

size_t ROOTMinimizer::getNCalls() const {
    return m_pMinimizer->NCalls();
}

void ROOTMinimizer::runMinimization() {
    m_pMinimizer->Minimize();
}

void ROOTMinimizer::setFCN(FitsService* c, fcnType fcn, size_t nPar) {

    m_fcnFunctor = ROOT::Math::Functor(c, fcn, nPar);
    m_pMinimizer->SetFunction(m_fcnFunctor);
}

void ROOTMinimizer::setNewMinimizer() {

    if (m_pMinimizer) {
        delete m_pMinimizer;
        m_pMinimizer = 0;

        warn(__FUNCTION__, "Old minimizer destroyed");
    }

    if (m_packageType == ROOTMinimizerPackageType::UNDEFINED
            || m_algorithmType == ROOTMinimizerAlgorithmType::UNDEFINED) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Undefined package or algorithm type");
    }

    std::string packageName =
            (m_packageType == ROOTMinimizerPackageType::Default) ?
                    ("") : (ROOTMinimizerPackageType(m_packageType).toString());
    std::string algorithmName =
            (m_algorithmType == ROOTMinimizerAlgorithmType::Default) ?
                    ("") :
                    (ROOTMinimizerAlgorithmType(m_algorithmType).toString());

    m_pMinimizer = ROOT::Math::Factory::CreateMinimizer(packageName,
            algorithmName);
}

NumA::MatrixD ROOTMinimizer::getCovarianceMatrix() const {

    if (getNVariables() == 0) {
        warn(__func__,
                "Unable to return covariance matrix as number of parameters is zero");
        return NumA::MatrixD(0, 0);
    }

    NumA::MatrixD m(getNVariables(), getNVariables());

    for (size_t i = 0; i < getNVariables(); i++) {
        for (size_t j = 0; j < getNVariables(); j++) {
            m.at(i, j) = m_pMinimizer->CovMatrix(i, j);
        }
    }

    return m;
}

NumA::MatrixD ROOTMinimizer::getCorrelationMatrix() const {

    if (getNVariables() == 0) {
        warn(__func__,
                "Unable to return correlation matrix as number of parameters is zero");
        return NumA::MatrixD(0, 0);
    }

    NumA::MatrixD m(getNVariables(), getNVariables());

    for (size_t i = 0; i < getNVariables(); i++) {
        for (size_t j = 0; j < getNVariables(); j++) {
            m.at(i, j) = m_pMinimizer->Correlation(i, j);
        }
    }

    return m;
}

