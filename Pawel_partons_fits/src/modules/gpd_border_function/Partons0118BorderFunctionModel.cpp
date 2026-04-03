/*
 * Partons0118BorderFunctionModel.cpp
 *
 *  Created on: Jan 17, 2016
 *      Author: Pawel Sznajder (NCBJ)
 */

#include "../../../include/modules/gpd_border_function/Partons0118BorderFunctionModel.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/GenericType.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/parton_distribution/GluonDistribution.h>
#include <partons/beans/parton_distribution/QuarkDistribution.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/gpd/GPDGK16Numerical.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/services/GPDService.h>
#include <partons/ServiceObjectRegistry.h>
#include <cmath>
#include <utility>

#include "../../../include/modules/pdf/PdfParameterization.h"

using namespace PARTONS;

const unsigned int Partons0118BorderFunctionModel::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new Partons0118BorderFunctionModel(
                        "Partons0118BorderFunctionModel"));

const std::string Partons0118BorderFunctionModel::PARAMETER_NAME_PARTONS0118BORDERFUNCTIONMODEL_PDF_REPLICA =
        "partons_0118_bf_pdf_replica";
const std::string Partons0118BorderFunctionModel::PARAMETER_NAME_PARTONS0118BORDERFUNCTIONMODEL_PDF_REPLICA_POL =
        "partons_0118_bf_pdf_replica_pol";

Partons0118BorderFunctionModel::Partons0118BorderFunctionModel(
        const std::string& className) :
        GPDModule(className) {

    m_listGPDComputeTypeAvailable.insert(
            std::make_pair(GPDType::H, &GPDModule::computeH));

    m_listGPDComputeTypeAvailable.insert(
            std::make_pair(GPDType::Ht, &GPDModule::computeHt));

    m_listGPDComputeTypeAvailable.insert(
            std::make_pair(GPDType::E, &GPDModule::computeE));

    m_listGPDComputeTypeAvailable.insert(
            std::make_pair(GPDType::Et, &GPDModule::computeEt));

    m_pPdf = new PdfParameterization();

    m_pdfReplica = 0;
    m_pdfReplicaPol = 0;

    m_pGPDModule = 0;
    m_pGPDService = 0;

    m_par_H_skew_uVal.clear();
    m_par_H_skew_uSea.clear();

    m_par_H_skew_dVal.clear();
    m_par_H_skew_dSea.clear();

    m_par_H_skew_sSea.clear();

    m_par_Ht_skew_uVal.clear();
    m_par_Ht_skew_uSea.clear();

    m_par_Ht_skew_dVal.clear();
    m_par_Ht_skew_dSea.clear();

    m_par_Ht_skew_sSea.clear();

    m_par_H_tDep_uVal.clear();
    m_par_H_tDep_uSea.clear();

    m_par_H_tDep_dVal.clear();
    m_par_H_tDep_dSea.clear();

    m_par_H_tDep_sSea.clear();

    m_par_Ht_tDep_uVal.clear();
    m_par_Ht_tDep_uSea.clear();

    m_par_Ht_tDep_dVal.clear();
    m_par_Ht_tDep_dSea.clear();

    m_par_Ht_tDep_sSea.clear();

    m_par_E_N = 0.;
    m_par_Et_N = 0.;

    m_MuF2_ref = 2.;
}

Partons0118BorderFunctionModel::~Partons0118BorderFunctionModel() {

    if (m_pPdf) {
        delete m_pPdf;
        m_pPdf = 0;
    }

    if (m_pGPDModule) {
        setGPDModule(0);
        m_pGPDModule = 0;
    }
}

Partons0118BorderFunctionModel::Partons0118BorderFunctionModel(
        const Partons0118BorderFunctionModel& other) :
        GPDModule(other) {

    if (other.m_pPdf) {
        m_pPdf = other.m_pPdf->clone();
    } else {
        m_pPdf = 0;
    }

    m_pdfReplica = other.m_pdfReplica;
    m_pdfReplicaPol = other.m_pdfReplicaPol;

    if (other.m_pGPDModule) {
        m_pGPDModule = 0;
        setGPDModule(other.m_pGPDModule->clone());
    } else {
        m_pGPDModule = 0;
    }

    m_pGPDService = other.m_pGPDService;

    m_par_H_skew_uVal = other.m_par_H_skew_uVal;
    m_par_H_skew_uSea = other.m_par_H_skew_uSea;

    m_par_H_skew_dVal = other.m_par_H_skew_dVal;
    m_par_H_skew_dSea = other.m_par_H_skew_dSea;

    m_par_H_skew_sSea = other.m_par_H_skew_sSea;

    m_par_Ht_skew_uVal = other.m_par_Ht_skew_uVal;
    m_par_Ht_skew_uSea = other.m_par_Ht_skew_uSea;

    m_par_Ht_skew_dVal = other.m_par_Ht_skew_dVal;
    m_par_Ht_skew_dSea = other.m_par_Ht_skew_dSea;

    m_par_Ht_skew_sSea = other.m_par_Ht_skew_sSea;

    m_par_H_tDep_uVal = other.m_par_H_tDep_uVal;
    m_par_H_tDep_uSea = other.m_par_H_tDep_uSea;

    m_par_H_tDep_dVal = other.m_par_H_tDep_dVal;
    m_par_H_tDep_dSea = other.m_par_H_tDep_dSea;

    m_par_H_tDep_sSea = other.m_par_H_tDep_sSea;

    m_par_Ht_tDep_uVal = other.m_par_Ht_tDep_uVal;
    m_par_Ht_tDep_uSea = other.m_par_Ht_tDep_uSea;

    m_par_Ht_tDep_dVal = other.m_par_Ht_tDep_dVal;
    m_par_Ht_tDep_dSea = other.m_par_Ht_tDep_dSea;

    m_par_Ht_tDep_sSea = other.m_par_Ht_tDep_sSea;

    m_par_E_N = other.m_par_E_N;
    m_par_Et_N = other.m_par_Et_N;
}

Partons0118BorderFunctionModel* Partons0118BorderFunctionModel::clone() const {
    return new Partons0118BorderFunctionModel(*this);
}

void Partons0118BorderFunctionModel::resolveObjectDependencies() {

    m_pGPDModule =
            Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    GPDGK16Numerical::classId);

    m_pGPDService =
            Partons::getInstance()->getServiceObjectRegistry()->getGPDService();
}

void Partons0118BorderFunctionModel::configure(
        const ElemUtils::Parameters& parameters) {

    GPDModule::configure(parameters);

    if (parameters.isAvailable(
            Partons0118BorderFunctionModel::PARAMETER_NAME_PARTONS0118BORDERFUNCTIONMODEL_PDF_REPLICA)) {

        m_pdfReplica = parameters.getLastAvailable().toInt();

        info(__func__,
                ElemUtils::Formatter()
                        << "Unpolarized PDF replica changed to : "
                        << m_pdfReplica);
    }

    if (parameters.isAvailable(
            Partons0118BorderFunctionModel::PARAMETER_NAME_PARTONS0118BORDERFUNCTIONMODEL_PDF_REPLICA_POL)) {

        m_pdfReplicaPol = parameters.getLastAvailable().toInt();

        info(__func__,
                ElemUtils::Formatter() << "Polarized PDF replica changed to : "
                        << m_pdfReplicaPol);
    }
}

std::string Partons0118BorderFunctionModel::toString() const {
    GPDModule::toString();
}

PartonDistribution Partons0118BorderFunctionModel::computeH() {

    //variables
    double HuVal, HuSea;
    double HdVal, HdSea;
    double HsSea;

    //calculate
    double pdfuval = m_pPdf->pdf(QuarkFlavor::UP, false, false, m_pdfReplica,
            m_xi, m_MuF2);
    double pdfusea = m_pPdf->pdf(QuarkFlavor::UP, true, false, m_pdfReplica,
            m_xi, m_MuF2);
    double pdfdval = m_pPdf->pdf(QuarkFlavor::DOWN, false, false, m_pdfReplica,
            m_xi, m_MuF2);
    double pdfdsea = m_pPdf->pdf(QuarkFlavor::DOWN, true, false, m_pdfReplica,
            m_xi, m_MuF2);
    double pdfssea = m_pPdf->pdf(QuarkFlavor::STRANGE, true, false,
            m_pdfReplica, m_xi, m_MuF2);

    HuVal = pdfuval * tDependance(m_par_H_tDep_uVal)
            * skewnessFunction(m_par_H_skew_uVal);
    HuSea = pdfusea * tDependance(m_par_H_tDep_uSea)
            * skewnessFunction(m_par_H_skew_uSea);
    HdVal = pdfdval * tDependance(m_par_H_tDep_dVal)
            * skewnessFunction(m_par_H_skew_dVal);
    HdSea = pdfdsea * tDependance(m_par_H_tDep_dSea)
            * skewnessFunction(m_par_H_skew_dSea);
    HsSea = pdfssea * tDependance(m_par_H_tDep_sSea)
            * skewnessFunction(m_par_H_skew_sSea);

    //get results
    QuarkDistribution quarkDistribution_u(QuarkFlavor::UP);
    QuarkDistribution quarkDistribution_d(QuarkFlavor::DOWN);
    QuarkDistribution quarkDistribution_s(QuarkFlavor::STRANGE);

    quarkDistribution_u.setQuarkDistribution(HuVal + HuSea);
    quarkDistribution_d.setQuarkDistribution(HdVal + HdSea);
    quarkDistribution_s.setQuarkDistribution(HsSea);

    quarkDistribution_u.setQuarkDistributionPlus(HuVal + 2 * HuSea);
    quarkDistribution_d.setQuarkDistributionPlus(HdVal + 2 * HdSea);
    quarkDistribution_s.setQuarkDistributionPlus(2 * HsSea);

    GluonDistribution gluonDistribution(0.);

    //store results
    PartonDistribution partonDistribution;

    partonDistribution.setGluonDistribution(gluonDistribution);
    partonDistribution.addQuarkDistribution(quarkDistribution_u);
    partonDistribution.addQuarkDistribution(quarkDistribution_d);
    partonDistribution.addQuarkDistribution(quarkDistribution_s);

    //return
    return partonDistribution;
}

PartonDistribution Partons0118BorderFunctionModel::computeHt() {

    //variables
    double HtuVal, HtuSea;
    double HtdVal, HtdSea;
    double HtsSea;

    //calculate
    double pdfuval = m_pPdf->pdf(QuarkFlavor::UP, false, true, m_pdfReplicaPol,
            m_xi, m_MuF2);
    double pdfusea = m_pPdf->pdf(QuarkFlavor::UP, true, true, m_pdfReplicaPol,
            m_xi, m_MuF2);
    double pdfdval = m_pPdf->pdf(QuarkFlavor::DOWN, false, true,
            m_pdfReplicaPol, m_xi, m_MuF2);
    double pdfdsea = m_pPdf->pdf(QuarkFlavor::DOWN, true, true, m_pdfReplicaPol,
            m_xi, m_MuF2);
    double pdfssea = m_pPdf->pdf(QuarkFlavor::STRANGE, true, true,
            m_pdfReplicaPol, m_xi, m_MuF2);

    HtuVal = pdfuval * tDependance(m_par_Ht_tDep_uVal)
            * skewnessFunction(m_par_Ht_skew_uVal);
    HtuSea = pdfusea * tDependance(m_par_Ht_tDep_uSea)
            * skewnessFunction(m_par_Ht_skew_uSea);
    HtdVal = pdfdval * tDependance(m_par_Ht_tDep_dVal)
            * skewnessFunction(m_par_Ht_skew_dVal);
    HtdSea = pdfdsea * tDependance(m_par_Ht_tDep_dSea)
            * skewnessFunction(m_par_Ht_skew_dSea);
    HtsSea = pdfssea * tDependance(m_par_Ht_tDep_sSea)
            * skewnessFunction(m_par_Ht_skew_sSea);

    //get results
    QuarkDistribution quarkDistribution_u(QuarkFlavor::UP);
    QuarkDistribution quarkDistribution_d(QuarkFlavor::DOWN);
    QuarkDistribution quarkDistribution_s(QuarkFlavor::STRANGE);

    quarkDistribution_u.setQuarkDistribution(HtuVal + HtuSea);
    quarkDistribution_d.setQuarkDistribution(HtdVal + HtdSea);
    quarkDistribution_s.setQuarkDistribution(HtsSea);

    quarkDistribution_u.setQuarkDistributionPlus(HtuVal + 2 * HtuSea);
    quarkDistribution_d.setQuarkDistributionPlus(HtdVal + 2 * HtdSea);
    quarkDistribution_s.setQuarkDistributionPlus(2 * HtsSea);

    GluonDistribution gluonDistribution(0.);

    //store results
    PartonDistribution partonDistribution;

    partonDistribution.setGluonDistribution(gluonDistribution);
    partonDistribution.addQuarkDistribution(quarkDistribution_u);
    partonDistribution.addQuarkDistribution(quarkDistribution_d);
    partonDistribution.addQuarkDistribution(quarkDistribution_s);

    //return
    return partonDistribution;
}

PartonDistribution Partons0118BorderFunctionModel::computeE() {

    List<GPDType> list;
    list.add(GPDType::E);

    PartonDistribution result =
            m_pGPDService->computeSingleKinematic(
                    GPDKinematic(m_x, m_xi, m_t, m_MuF2, m_MuR2), m_pGPDModule,
                    list).getPartonDistribution(GPDType::E);

    PartonDistribution partonDistribution;

    QuarkDistribution quarkDistribution_u(QuarkFlavor::UP);
    QuarkDistribution quarkDistribution_d(QuarkFlavor::DOWN);
    QuarkDistribution quarkDistribution_s(QuarkFlavor::STRANGE);

    GluonDistribution gluonDistribution;

    quarkDistribution_u.setQuarkDistributionPlus(
            m_par_E_N
                    * result.getQuarkDistribution(QuarkFlavor::UP).getQuarkDistributionPlus());
    quarkDistribution_d.setQuarkDistributionPlus(
            m_par_E_N
                    * result.getQuarkDistribution(QuarkFlavor::DOWN).getQuarkDistributionPlus());
    quarkDistribution_s.setQuarkDistributionPlus(
            m_par_E_N
                    * result.getQuarkDistribution(QuarkFlavor::STRANGE).getQuarkDistributionPlus());

    partonDistribution.addQuarkDistribution(quarkDistribution_u);
    partonDistribution.addQuarkDistribution(quarkDistribution_d);
    partonDistribution.addQuarkDistribution(quarkDistribution_s);
    partonDistribution.setGluonDistribution(gluonDistribution);

    return partonDistribution;
}

PartonDistribution Partons0118BorderFunctionModel::computeEt() {

    List<GPDType> list;
    list.add(GPDType::Et);

    PartonDistribution result =
            m_pGPDService->computeSingleKinematic(
                    GPDKinematic(m_x, m_xi, m_t, m_MuF2, m_MuR2), m_pGPDModule,
                    list).getPartonDistribution(GPDType::Et);

    PartonDistribution partonDistribution;

    QuarkDistribution quarkDistribution_u(QuarkFlavor::UP);
    QuarkDistribution quarkDistribution_d(QuarkFlavor::DOWN);
    QuarkDistribution quarkDistribution_s(QuarkFlavor::STRANGE);

    GluonDistribution gluonDistribution;

    quarkDistribution_u.setQuarkDistributionPlus(
            m_par_Et_N
                    * result.getQuarkDistribution(QuarkFlavor::UP).getQuarkDistributionPlus());
    quarkDistribution_d.setQuarkDistributionPlus(
            m_par_Et_N
                    * result.getQuarkDistribution(QuarkFlavor::DOWN).getQuarkDistributionPlus());
    quarkDistribution_s.setQuarkDistributionPlus(
            m_par_Et_N
                    * result.getQuarkDistribution(QuarkFlavor::STRANGE).getQuarkDistributionPlus());

    partonDistribution.addQuarkDistribution(quarkDistribution_u);
    partonDistribution.addQuarkDistribution(quarkDistribution_d);
    partonDistribution.addQuarkDistribution(quarkDistribution_s);
    partonDistribution.setGluonDistribution(gluonDistribution);

    return partonDistribution;
}

double Partons0118BorderFunctionModel::skewnessFunction(
        const std::vector<double>& parameters) const {

    if (parameters.size() != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Wrong number of parameters, is "
                        << parameters.size() << " should be 1");
    }

    return parameters.at(0) / pow(1. - pow(m_xi, 2), 2);
}

double Partons0118BorderFunctionModel::tDependance(
        const std::vector<double>& parameters) const {

    if (parameters.size() != 3) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Wrong number of parameters, is "
                        << parameters.size() << " should be 3");
    }

    double profile = parameters.at(0) * pow(1. - m_xi, 3) * log(1 / m_xi)
            + parameters.at(1) * pow(1. - m_xi, 3)
            + parameters.at(2) * m_xi * pow(1. - m_xi, 2);

    return exp(profile * m_t);
}

PARTONS::GPDModule* Partons0118BorderFunctionModel::getGPDModule() const {
    return m_pGPDModule;
}

void Partons0118BorderFunctionModel::setGPDModule(
        PARTONS::GPDModule* gpdModule) {

    m_pModuleObjectFactory->updateModulePointerReference(m_pGPDModule,
            gpdModule);
    m_pGPDModule = gpdModule;
}

double Partons0118BorderFunctionModel::getParEN() const {
    return m_par_E_N;
}

void Partons0118BorderFunctionModel::setParEN(double parEN) {
    m_par_E_N = parEN;
}

double Partons0118BorderFunctionModel::getParEtN() const {
    return m_par_Et_N;
}

void Partons0118BorderFunctionModel::setParEtN(double parEtN) {
    m_par_Et_N = parEtN;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHSkewDSea() const {
    return m_par_H_skew_dSea;
}

void Partons0118BorderFunctionModel::setParHSkewDSea(
        const std::vector<double>& parHSkewDSea) {
    m_par_H_skew_dSea = parHSkewDSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHSkewDVal() const {
    return m_par_H_skew_dVal;
}

void Partons0118BorderFunctionModel::setParHSkewDVal(
        const std::vector<double>& parHSkewDVal) {
    m_par_H_skew_dVal = parHSkewDVal;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHSkewSSea() const {
    return m_par_H_skew_sSea;
}

void Partons0118BorderFunctionModel::setParHSkewSSea(
        const std::vector<double>& parHSkewSSea) {
    m_par_H_skew_sSea = parHSkewSSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHSkewUSea() const {
    return m_par_H_skew_uSea;
}

void Partons0118BorderFunctionModel::setParHSkewUSea(
        const std::vector<double>& parHSkewUSea) {
    m_par_H_skew_uSea = parHSkewUSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHSkewUVal() const {
    return m_par_H_skew_uVal;
}

void Partons0118BorderFunctionModel::setParHSkewUVal(
        const std::vector<double>& parHSkewUVal) {
    m_par_H_skew_uVal = parHSkewUVal;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHTDepDSea() const {
    return m_par_H_tDep_dSea;
}

void Partons0118BorderFunctionModel::setParHTDepDSea(
        const std::vector<double>& parHTDepDSea) {
    m_par_H_tDep_dSea = parHTDepDSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHTDepDVal() const {
    return m_par_H_tDep_dVal;
}

void Partons0118BorderFunctionModel::setParHTDepDVal(
        const std::vector<double>& parHTDepDVal) {
    m_par_H_tDep_dVal = parHTDepDVal;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHTDepSSea() const {
    return m_par_H_tDep_sSea;
}

void Partons0118BorderFunctionModel::setParHTDepSSea(
        const std::vector<double>& parHTDepSSea) {
    m_par_H_tDep_sSea = parHTDepSSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHTDepUSea() const {
    return m_par_H_tDep_uSea;
}

void Partons0118BorderFunctionModel::setParHTDepUSea(
        const std::vector<double>& parHTDepUSea) {
    m_par_H_tDep_uSea = parHTDepUSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHTDepUVal() const {
    return m_par_H_tDep_uVal;
}

void Partons0118BorderFunctionModel::setParHTDepUVal(
        const std::vector<double>& parHTDepUVal) {
    m_par_H_tDep_uVal = parHTDepUVal;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtSkewDSea() const {
    return m_par_Ht_skew_dSea;
}

void Partons0118BorderFunctionModel::setParHtSkewDSea(
        const std::vector<double>& parHtSkewDSea) {
    m_par_Ht_skew_dSea = parHtSkewDSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtSkewDVal() const {
    return m_par_Ht_skew_dVal;
}

void Partons0118BorderFunctionModel::setParHtSkewDVal(
        const std::vector<double>& parHtSkewDVal) {
    m_par_Ht_skew_dVal = parHtSkewDVal;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtSkewSSea() const {
    return m_par_Ht_skew_sSea;
}

void Partons0118BorderFunctionModel::setParHtSkewSSea(
        const std::vector<double>& parHtSkewSSea) {
    m_par_Ht_skew_sSea = parHtSkewSSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtSkewUSea() const {
    return m_par_Ht_skew_uSea;
}

void Partons0118BorderFunctionModel::setParHtSkewUSea(
        const std::vector<double>& parHtSkewUSea) {
    m_par_Ht_skew_uSea = parHtSkewUSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtSkewUVal() const {
    return m_par_Ht_skew_uVal;
}

void Partons0118BorderFunctionModel::setParHtSkewUVal(
        const std::vector<double>& parHtSkewUVal) {
    m_par_Ht_skew_uVal = parHtSkewUVal;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtTDepDSea() const {
    return m_par_Ht_tDep_dSea;
}

void Partons0118BorderFunctionModel::setParHtTDepDSea(
        const std::vector<double>& parHtTDepDSea) {
    m_par_Ht_tDep_dSea = parHtTDepDSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtTDepDVal() const {
    return m_par_Ht_tDep_dVal;
}

void Partons0118BorderFunctionModel::setParHtTDepDVal(
        const std::vector<double>& parHtTDepDVal) {
    m_par_Ht_tDep_dVal = parHtTDepDVal;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtTDepSSea() const {
    return m_par_Ht_tDep_sSea;
}

void Partons0118BorderFunctionModel::setParHtTDepSSea(
        const std::vector<double>& parHtTDepSSea) {
    m_par_Ht_tDep_sSea = parHtTDepSSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtTDepUSea() const {
    return m_par_Ht_tDep_uSea;
}

void Partons0118BorderFunctionModel::setParHtTDepUSea(
        const std::vector<double>& parHtTDepUSea) {
    m_par_Ht_tDep_uSea = parHtTDepUSea;
}

const std::vector<double>& Partons0118BorderFunctionModel::getParHtTDepUVal() const {
    return m_par_Ht_tDep_uVal;
}

void Partons0118BorderFunctionModel::setParHtTDepUVal(
        const std::vector<double>& parHtTDepUVal) {
    m_par_Ht_tDep_uVal = parHtTDepUVal;
}

