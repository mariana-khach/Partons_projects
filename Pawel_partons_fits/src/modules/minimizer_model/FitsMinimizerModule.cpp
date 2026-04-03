#include "../../../include/modules/minimizer_model/FitsMinimizerModule.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <ElementaryUtils/string_utils/StringUtils.h>
#include <iterator>
#include <limits>

using namespace PARTONS;

FitsMinimizerModule::FitsMinimizerModule(const std::string& className) :
        ModuleObject(className, PARTONS::ChannelType::UNDEFINED) {
}

FitsMinimizerModule::FitsMinimizerModule(const FitsMinimizerModule& other) :
        ModuleObject(other) {
}

FitsMinimizerModule::~FitsMinimizerModule() {
}

void FitsMinimizerModule::resolveObjectDependencies() {
    ModuleObject::resolveObjectDependencies();
}

void FitsMinimizerModule::configure(const ElemUtils::Parameters& parameters) {
    ModuleObject::configure(parameters);
}

void FitsMinimizerModule::initModule() {
}

void FitsMinimizerModule::isModuleWellConfigured() {
}

void FitsMinimizerModule::prepareSubModules(
        const std::map<std::string, BaseObjectData>& subModulesData) {
    throw ElemUtils::CustomException(getClassName(), __func__,
            "TODO : implement");
}

size_t FitsMinimizerModule::SetVariable(const std::string& varName,
        double initialValue, double step) {

    if (isVariableRegistered(varName)) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Variable name " << varName
                        << " already registered");
    }

    m_variableNames.push_back(varName);
    m_variableStartValues.insert(std::make_pair(varName, initialValue));
    m_variableSteps.insert(std::make_pair(varName, step));
    m_variableFixedValues.insert(std::make_pair(varName, false));
    m_variableLimitValues.insert(
            std::make_pair(varName,
                    std::make_pair(std::numeric_limits<int>::min(),
                            std::numeric_limits<int>::max())));

    return m_variableNames.size() - 1;
}

size_t FitsMinimizerModule::SetLimitedVariable(const std::string& varName,
        double initialValue, double step,
        const std::pair<double, double>& limits) {

    if (isVariableRegistered(varName)) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Variable name " << varName
                        << " already registered");
    }

    m_variableNames.push_back(varName);
    m_variableStartValues.insert(std::make_pair(varName, initialValue));
    m_variableSteps.insert(std::make_pair(varName, step));
    m_variableFixedValues.insert(std::make_pair(varName, false));
    m_variableLimitValues.insert(std::make_pair(varName, limits));

    return m_variableNames.size() - 1;
}

size_t FitsMinimizerModule::SetFixedVariable(const std::string& varName,
        double initialValue) {

    if (isVariableRegistered(varName)) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Variable name " << varName
                        << " already registered");
    }

    m_variableNames.push_back(varName);
    m_variableStartValues.insert(std::make_pair(varName, initialValue));
    m_variableSteps.insert(std::make_pair(varName, 0.));
    m_variableFixedValues.insert(std::make_pair(varName, true));
    m_variableLimitValues.insert(std::make_pair(varName, std::make_pair(0., 0.)));

    return m_variableNames.size() - 1;
}

size_t FitsMinimizerModule::getNVariables() const {
    return m_variableNames.size();
}

bool FitsMinimizerModule::isVariableRegistered(
        const std::string& varName) const {

    std::vector<std::string>::const_iterator it;

    for (it = m_variableNames.begin(); it != m_variableNames.end(); it++) {
        if (ElemUtils::StringUtils::equals(varName, *it)) {
            return true;
        }
    }

    return false;
}

bool FitsMinimizerModule::isVariableRegistered(size_t i) const {
    return i < m_variableNames.size();
}

std::string FitsMinimizerModule::getVariableName(size_t i) const {

    if (!isVariableRegistered(i)) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Illegal index");
    }

    return m_variableNames.at(i);
}

size_t FitsMinimizerModule::getVariableIndex(
        const std::string& varName) const {

    std::vector<std::string>::const_iterator it;

    for (it = m_variableNames.begin(); it != m_variableNames.end(); it++) {
        if (ElemUtils::StringUtils::equals(varName, *it)) {
            break;
        }
    }

    if (it == m_variableNames.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Variable name unknown");
    }

    return it - m_variableNames.begin();
}

double FitsMinimizerModule::getVariableStartValue(size_t i) const {

    std::map<std::string, double>::const_iterator it = m_variableStartValues.find(
            getVariableName(i));

    return it->second;
}

double FitsMinimizerModule::getVariableStartValue(
        const std::string& varName) const {

    std::map<std::string, double>::const_iterator it = m_variableStartValues.find(
            varName);

    if (it == m_variableStartValues.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Variable name unknown");
    }

    return it->second;
}

double FitsMinimizerModule::getVariableStep(size_t i) const {

    std::map<std::string, double>::const_iterator it = m_variableSteps.find(
            getVariableName(i));

    return it->second;
}

double FitsMinimizerModule::getVariableStep(const std::string& varName) const {

    std::map<std::string, double>::const_iterator it = m_variableSteps.find(varName);

    if (it == m_variableSteps.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Variable name unknown");
    }

    return it->second;
}

bool FitsMinimizerModule::isVariableFixed(size_t i) const {

    std::map<std::string, bool>::const_iterator it = m_variableFixedValues.find(
            getVariableName(i));

    return it->second;
}

bool FitsMinimizerModule::isVariableFixed(const std::string& varName) const {

    std::map<std::string, bool>::const_iterator it = m_variableFixedValues.find(
            varName);

    if (it == m_variableFixedValues.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Variable name unknown");
    }

    return it->second;
}

std::pair<double, double> FitsMinimizerModule::getVariableLimits(
        size_t i) const {

    std::map<std::string, std::pair<double, double> >::const_iterator it =
            m_variableLimitValues.find(getVariableName(i));

    return it->second;
}

std::pair<double, double> FitsMinimizerModule::getVariableLimits(
        const std::string& varName) const {

    std::map<std::string, std::pair<double, double> >::const_iterator it =
            m_variableLimitValues.find(varName);

    if (it == m_variableLimitValues.end()) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Variable name unknown");
    }

    return it->second;

}

const std::map<std::string, bool>& FitsMinimizerModule::getVariableFixedValues() const {
    return m_variableFixedValues;
}

const std::map<std::string, std::pair<double, double> >& FitsMinimizerModule::getVariableLimitValues() const {
    return m_variableLimitValues;
}

const std::vector<std::string>& FitsMinimizerModule::getVariableNames() const {
    return m_variableNames;
}

const std::map<std::string, double>& FitsMinimizerModule::getVariableStartValues() const {
    return m_variableStartValues;
}

const std::map<std::string, double>& FitsMinimizerModule::getVariableSteps() const {
    return m_variableSteps;
}
