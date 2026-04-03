#include "../../../include/beans/observable_data_fit/ObservableDataFit.h"

#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/modules/observable/DVCS/DVCSObservable.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/ServiceObjectRegistry.h>

#include "../../../include/services/FitsService.h"

using namespace PARTONS;

ObservableDataFit::ObservableDataFit(const DVCSObservableResult &observableResult) :
        BaseObject("ObservableDataFit"), m_pFitsService(
                static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                        "FitsService"))), m_pObservableService(
                Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService()), m_pObservable(
                0) {

    m_pObservable =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    observableResult.getComputationModuleName());

    m_cffKinematic = DVCSConvolCoeffFunctionKinematic();
    m_cffKinematicsIndex = 0;

    m_observableKinematic = observableResult.getKinematic();
    m_referenceObservableResult = observableResult;
    m_fittedObservableResult = DVCSObservableResult();
}

ObservableDataFit::~ObservableDataFit() {
    //TODO how to handle memory without double free ??? ; Related to cross configuration mechanism

//    if (m_pObservable) {
//        delete m_pObservable;
//        m_pObservable = 0;
//    }
}

void ObservableDataFit::resolveObjectDependencies() {
    m_pObservable->setProcessModule(
            static_cast<DVCSProcessModule*>(m_pFitsService->getDVCSProcessModule()));
}

std::string ObservableDataFit::toString() const {

    ElemUtils::Formatter formatter;

    formatter << "ObservableKinematic: " << m_observableKinematic.toString()
            << " ReferenceObservableResult: "
            << m_referenceObservableResult.toString()
            << " FittedObservableResult: "
            << m_fittedObservableResult.toString();

    return formatter.str();
}

void ObservableDataFit::compute(
        const PARTONS::DVCSConvolCoeffFunctionResult& dvcsConvolCoeffFunctionResult) {

    debug(__func__,
            ElemUtils::Formatter() << "Processing ... "
                    << m_pObservable->toString());

    static_cast<DVCSProcessModule*>(m_pFitsService->getDVCSProcessModule())->setConvolCoeffFunction(
            m_cffKinematic,
            dvcsConvolCoeffFunctionResult);

    m_fittedObservableResult = m_pObservableService->computeSingleKinematic(
            m_observableKinematic, m_pObservable);
}

const DVCSObservableResult& ObservableDataFit::getFittedObservableResult() const {
    return m_fittedObservableResult;
}

const DVCSObservableResult& ObservableDataFit::getReferenceObservableResult() const {
    return m_referenceObservableResult;
}

const DVCSObservableKinematic& ObservableDataFit::getObservableKinematic() const {
    return m_observableKinematic;
}

DVCSObservable* ObservableDataFit::getObservable() const {
    return m_pObservable;
}

const PARTONS::DVCSConvolCoeffFunctionKinematic& ObservableDataFit::getCffKinematic() const {
    return m_cffKinematic;
}

void ObservableDataFit::setCffKinematic(
        const PARTONS::DVCSConvolCoeffFunctionKinematic& cffKinematic) {
    m_cffKinematic = cffKinematic;
}

size_t ObservableDataFit::getCffKinematicsIndex() const {
    return m_cffKinematicsIndex;
}

void ObservableDataFit::setCffKinematicsIndex(size_t cffKinematicsIndex) {
    m_cffKinematicsIndex = cffKinematicsIndex;
}
