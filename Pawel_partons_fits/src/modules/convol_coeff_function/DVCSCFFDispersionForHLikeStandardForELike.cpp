#include "../../../include/modules/convol_coeff_function/DVCSCFFDispersionForHLikeStandardForELike.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSCFFDispersionRelation.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSCFFStandard.h>
#include <partons/modules/gpd_subtraction_constant/GPDSubtractionConstantModule.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/beans/gpd/GPDSubtractionConstantKinematic.h>
#include <partons/beans/gpd/GPDSubtractionConstantResult.h>
#include <partons/services/DVCSConvolCoeffFunctionService.h>
#include <partons/ServiceObjectRegistry.h>
#include <iostream>
#include <utility>

using namespace PARTONS;

const unsigned int DVCSCFFDispersionForHLikeStandardForELike::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new DVCSCFFDispersionForHLikeStandardForELike(
                        "DVCSCFFDispersionForHLikeStandardForELike"));

DVCSCFFDispersionForHLikeStandardForELike::DVCSCFFDispersionForHLikeStandardForELike(
        const std::string &className) :
        DVCSConvolCoeffFunctionModule(className) {

    m_pDVCSCFFDispersionRelationModel = 0;
    m_pDVCSCFFStandardModel = 0;
    m_pSubtractionConstantModule = 0;

    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::H,
                    &DVCSConvolCoeffFunctionModule::computeCFF));
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::E,
                    &DVCSConvolCoeffFunctionModule::computeCFF));
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::Ht,
                    &DVCSConvolCoeffFunctionModule::computeCFF));
    m_listOfCFFComputeFunctionAvailable.insert(
            std::make_pair(GPDType::Et,
                    &DVCSConvolCoeffFunctionModule::computeCFF));
}

DVCSCFFDispersionForHLikeStandardForELike::DVCSCFFDispersionForHLikeStandardForELike(
        const DVCSCFFDispersionForHLikeStandardForELike &other) :
        DVCSConvolCoeffFunctionModule(other) {

    if (other.m_pDVCSCFFDispersionRelationModel) {
        m_pDVCSCFFDispersionRelationModel = 0;
        setDVCSCFFDispersionRelationModel(
                other.m_pDVCSCFFDispersionRelationModel->clone());
    } else {
        m_pDVCSCFFDispersionRelationModel = 0;
    }

    if (other.m_pDVCSCFFStandardModel) {
        m_pDVCSCFFStandardModel = 0;
        setDVCSCFFStandardModel(other.m_pDVCSCFFStandardModel->clone());
    } else {
        m_pDVCSCFFStandardModel = 0;
    }

    if (other.m_pSubtractionConstantModule) {
        m_pSubtractionConstantModule = 0;
        setSubtractionConstantModule(
                other.m_pSubtractionConstantModule->clone());
    } else {
        m_pSubtractionConstantModule = 0;
    }
}

DVCSCFFDispersionForHLikeStandardForELike* DVCSCFFDispersionForHLikeStandardForELike::clone() const {
    return new DVCSCFFDispersionForHLikeStandardForELike(*this);
}

DVCSCFFDispersionForHLikeStandardForELike::~DVCSCFFDispersionForHLikeStandardForELike() {

    setDVCSCFFDispersionRelationModel(0);
    m_pDVCSCFFDispersionRelationModel = 0;

    setDVCSCFFStandardModel(0);
    m_pDVCSCFFStandardModel = 0;

    setSubtractionConstantModule(0);
    m_pSubtractionConstantModule = 0;
}

void DVCSCFFDispersionForHLikeStandardForELike::configure(
        const ElemUtils::Parameters &parameters) {

    DVCSConvolCoeffFunctionModule::configure(parameters);

    m_pDVCSCFFDispersionRelationModel->configure(parameters);
    m_pDVCSCFFStandardModel->configure(parameters);
}

void DVCSCFFDispersionForHLikeStandardForELike::resolveObjectDependencies() {

    DVCSConvolCoeffFunctionModule::resolveObjectDependencies();

    m_pDVCSCFFStandardModel =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFStandard::classId);

    m_pDVCSCFFDispersionRelationModel =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFDispersionRelation::classId);
}

void DVCSCFFDispersionForHLikeStandardForELike::prepareSubModules(
        const std::map<std::string, BaseObjectData>& subModulesData) {

    //mother
    DVCSConvolCoeffFunctionModule::prepareSubModules(subModulesData);

    //search
    std::map<std::string, BaseObjectData>::const_iterator it;

    //GPD subtraction constant module
    it =
            subModulesData.find(
                    GPDSubtractionConstantModule::GPD_SUBTRACTION_CONSTANT_MODULE_CLASS_NAME);

    //check if there
    if (it != subModulesData.end()) {

        if (m_pSubtractionConstantModule != 0) {
            setSubtractionConstantModule(0);
            m_pSubtractionConstantModule = 0;
        }

        if (m_pSubtractionConstantModule == 0) {

            m_pSubtractionConstantModule =
                    Partons::getInstance()->getModuleObjectFactory()->newGPDSubtractionConstantModule(
                            (it->second).getModuleClassName());

            info(__func__,
                    ElemUtils::Formatter()
                            << "Configured with GPDSubtractionConstantModule = "
                            << m_pSubtractionConstantModule->getClassName());

            m_pSubtractionConstantModule->configure(
                    (it->second).getParameters());
            m_pSubtractionConstantModule->prepareSubModules(
                    (it->second).getSubModules());

            //GPD model
            m_pDVCSCFFDispersionRelationModel->setGPDModule(m_pGPDModule);
            m_pDVCSCFFStandardModel->setGPDModule(m_pGPDModule);

            //Subtraction constant
            static_cast<DVCSCFFDispersionRelation*>(m_pDVCSCFFDispersionRelationModel)->setSubtractionConstantModule(
                    m_pSubtractionConstantModule);
        }
    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << getClassName()
                        << " is GPDSubtractionConstantModule dependent and you have not provided one");
    }
}

PARTONS::DVCSConvolCoeffFunctionModule* DVCSCFFDispersionForHLikeStandardForELike::getDVCSCFFDispersionRelationModel() const {
    return m_pDVCSCFFDispersionRelationModel;
}

void DVCSCFFDispersionForHLikeStandardForELike::setDVCSCFFDispersionRelationModel(
        PARTONS::DVCSConvolCoeffFunctionModule* dvcsCFFDispersionRelationModel) {

    m_pModuleObjectFactory->updateModulePointerReference(
            m_pDVCSCFFDispersionRelationModel, dvcsCFFDispersionRelationModel);
    m_pDVCSCFFDispersionRelationModel = dvcsCFFDispersionRelationModel;
}

PARTONS::DVCSConvolCoeffFunctionModule* DVCSCFFDispersionForHLikeStandardForELike::getDVCSCFFStandardModel() const {
    return m_pDVCSCFFStandardModel;
}

void DVCSCFFDispersionForHLikeStandardForELike::setDVCSCFFStandardModel(
        PARTONS::DVCSConvolCoeffFunctionModule* dvcsCFFStandardModel) {

    m_pModuleObjectFactory->updateModulePointerReference(
            m_pDVCSCFFStandardModel, dvcsCFFStandardModel);
    m_pDVCSCFFStandardModel = dvcsCFFStandardModel;
}

GPDSubtractionConstantModule* DVCSCFFDispersionForHLikeStandardForELike::getSubtractionConstantModule() const {
    return m_pSubtractionConstantModule;
}

void DVCSCFFDispersionForHLikeStandardForELike::setSubtractionConstantModule(
        GPDSubtractionConstantModule* subtractionConstantModule) {

    m_pModuleObjectFactory->updateModulePointerReference(
            m_pSubtractionConstantModule, subtractionConstantModule);
    m_pSubtractionConstantModule = subtractionConstantModule;
}

void DVCSCFFDispersionForHLikeStandardForELike::initModule() {
    DVCSConvolCoeffFunctionModule::initModule();
}

void DVCSCFFDispersionForHLikeStandardForELike::isModuleWellConfigured() {
    DVCSConvolCoeffFunctionModule::isModuleWellConfigured();
}

std::complex<double> DVCSCFFDispersionForHLikeStandardForELike::computeCFF() {

    switch (m_currentGPDComputeType) {

    case GPDType::H: {
        return computeCFFHHt();
    }
        break;

    case GPDType::E: {
        return computeCFFEEt();
    }
        break;

    case GPDType::Ht: {
        return computeCFFHHt();
    }
        break;

    case GPDType::Et: {
        return computeCFFEEt();
    }
        break;

    default: {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Calculation for GPD type "
                        << GPDType(m_currentGPDComputeType).toString()
                        << " unsupported");
        return std::complex<double>(0., 0.);
    }

    }
}

std::complex<double> DVCSCFFDispersionForHLikeStandardForELike::computeCFFHHt() {

    // Run
    List<GPDType> list;
    list.add(m_currentGPDComputeType);

    return Partons::getInstance()->getServiceObjectRegistry()->getDVCSConvolCoeffFunctionService()->computeSingleKinematic(
            DVCSConvolCoeffFunctionKinematic(m_xi, m_t, m_Q2, m_MuF2, m_MuR2),
            m_pDVCSCFFDispersionRelationModel, list).getResult(
            m_currentGPDComputeType);
}

std::complex<double> DVCSCFFDispersionForHLikeStandardForELike::computeCFFEEt() {

    //Run
    List<GPDType> list;
    list.add(m_currentGPDComputeType);

    return Partons::getInstance()->getServiceObjectRegistry()->getDVCSConvolCoeffFunctionService()->computeSingleKinematic(
            DVCSConvolCoeffFunctionKinematic(m_xi, m_t, m_Q2, m_MuF2, m_MuR2),
            m_pDVCSCFFStandardModel, list).getResult(m_currentGPDComputeType)
            + std::complex<double>(
                    m_pSubtractionConstantModule->compute(
                            GPDSubtractionConstantKinematic(m_t, m_MuF2, m_MuR2),
                            m_currentGPDComputeType).getValue().getValue(),
                    0.);
}
