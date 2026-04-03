#include "../../../include/modules/fits_model/FitsModelModule.h"

using namespace PARTONS;

FitsModelModule::FitsModelModule(const std::string &className) :
        ModuleObject(className, PARTONS::ChannelType::UNDEFINED) {
}

FitsModelModule::FitsModelModule(const FitsModelModule &other) :
        ModuleObject(other) {
    m_parameters = other.m_parameters;
}

FitsModelModule::~FitsModelModule() {
}

void FitsModelModule::resolveObjectDependencies() {
    ModuleObject::resolveObjectDependencies();
}

void FitsModelModule::configure(const ElemUtils::Parameters &parameters) {
    ModuleObject::configure(parameters);
}

void FitsModelModule::initModule() {
}

void FitsModelModule::isModuleWellConfigured() {
}

const ElemUtils::Parameters& FitsModelModule::getParameters() const {
    return m_parameters;
}

void FitsModelModule::setParameters(const ElemUtils::Parameters& parameters) {
    m_parameters = parameters;
}

void FitsModelModule::prepareSubModules(
        const std::map<std::string, BaseObjectData>& subModulesData) {
    // Nothing to do.
}
