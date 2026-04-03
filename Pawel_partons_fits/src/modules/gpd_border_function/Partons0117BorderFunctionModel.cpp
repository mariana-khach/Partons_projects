/*
 * Partons0117BorderFunctionModel.cpp
 *
 *  Created on: Jan 17, 2016
 *      Author: Pawel Sznajder (NCBJ)
 */

#include "../../../include/modules/gpd_border_function/Partons0117BorderFunctionModel.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/Parameter.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <LHAPDF/LHAPDF.h>
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

using namespace PARTONS;

const unsigned int Partons0117BorderFunctionModel::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new Partons0117BorderFunctionModel(
                        "Partons0117BorderFunctionModel"));

const std::string Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_SET =
        "partons_0117_bf_pdf_set";
const std::string Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_MEMBER =
        "partons_0117_bf_pdf_member";
const std::string Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_SET_POL =
        "partons_0117_bf_pdf_set_pol";
const std::string Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_MEMBER_POL =
        "partons_0117_bf_pdf_member_pol";

Partons0117BorderFunctionModel::Partons0117BorderFunctionModel(
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

    m_pPdf = 0;
    m_pPdfPol = 0;

    m_pdfSet = "NNPDF30_lo_as_0118";
    m_pdfMember = 0;
    m_pdfSetPol = "NNPDFpol11_100";
    m_pdfMemberPol = 0;

    setPDF();
    setPDFPol();

    m_pGPDModule = 0;
    m_pGPDService = 0;

    m_par_H_skew_uVal.clear();
    m_par_H_skew_uSea.clear();

    m_par_H_skew_dVal.clear();
    m_par_H_skew_dSea.clear();

    m_par_Ht_skew_uVal.clear();
    m_par_Ht_skew_uSea.clear();

    m_par_Ht_skew_dVal.clear();
    m_par_Ht_skew_dSea.clear();

    m_par_H_tDep_uVal.clear();
    m_par_H_tDep_uSea.clear();

    m_par_H_tDep_dVal.clear();
    m_par_H_tDep_dSea.clear();

    m_par_Ht_tDep_uVal.clear();
    m_par_Ht_tDep_uSea.clear();

    m_par_Ht_tDep_dVal.clear();
    m_par_Ht_tDep_dSea.clear();

    m_par_E_N = 0.;
    m_par_Et_N = 0.;

    m_MuF2_ref = 2.;
}

Partons0117BorderFunctionModel::~Partons0117BorderFunctionModel() {

    if (m_pPdf) {
        delete m_pPdf;
        m_pPdf = 0;
    }

    if (m_pPdfPol) {
        delete m_pPdfPol;
        m_pPdfPol = 0;
    }

    if (m_pGPDModule) {
        setGPDModule(0);
        m_pGPDModule = 0;
    }
}

Partons0117BorderFunctionModel::Partons0117BorderFunctionModel(
        const Partons0117BorderFunctionModel& other) :
        GPDModule(other) {

    m_pPdf = 0;
    m_pPdfPol = 0;

    m_pdfSet = other.m_pdfSet;
    m_pdfMember = other.m_pdfMember;
    m_pdfSetPol = other.m_pdfSetPol;
    m_pdfMemberPol = other.m_pdfMemberPol;

    setPDF();
    setPDFPol();

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

    m_par_Ht_skew_uVal = other.m_par_Ht_skew_uVal;
    m_par_Ht_skew_uSea = other.m_par_Ht_skew_uSea;

    m_par_Ht_skew_dVal = other.m_par_Ht_skew_dVal;
    m_par_Ht_skew_dSea = other.m_par_Ht_skew_dSea;

    m_par_H_tDep_uVal = other.m_par_H_tDep_uVal;
    m_par_H_tDep_uSea = other.m_par_H_tDep_uSea;

    m_par_H_tDep_dVal = other.m_par_H_tDep_dVal;
    m_par_H_tDep_dSea = other.m_par_H_tDep_dSea;

    m_par_Ht_tDep_uVal = other.m_par_Ht_tDep_uVal;
    m_par_Ht_tDep_uSea = other.m_par_Ht_tDep_uSea;

    m_par_Ht_tDep_dVal = other.m_par_Ht_tDep_dVal;
    m_par_Ht_tDep_dSea = other.m_par_Ht_tDep_dSea;

    m_par_E_N = other.m_par_E_N;
    m_par_Et_N = other.m_par_Et_N;
}

Partons0117BorderFunctionModel* Partons0117BorderFunctionModel::clone() const {
    return new Partons0117BorderFunctionModel(*this);
}

void Partons0117BorderFunctionModel::resolveObjectDependencies() {

    m_pGPDModule =
            Partons::getInstance()->getModuleObjectFactory()->newGPDModule(
                    GPDGK16Numerical::classId);

    m_pGPDService =
            Partons::getInstance()->getServiceObjectRegistry()->getGPDService();
}

void Partons0117BorderFunctionModel::configure(
        const ElemUtils::Parameters& parameters) {

    GPDModule::configure(parameters);

    bool pdfChanged = false;

    if (parameters.isAvailable(
            Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_SET)) {

        m_pdfSet = parameters.getLastAvailable().toString();

        info(__func__,
                ElemUtils::Formatter() << "Unpolarized PDF set changed to : "
                        << m_pdfSet);

        pdfChanged = true;
    }

    if (parameters.isAvailable(
            Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_MEMBER)) {

        m_pdfMember = parameters.getLastAvailable().toInt();

        info(__func__,
                ElemUtils::Formatter() << "Unpolarized PDF member changed to : "
                        << m_pdfMember);

        pdfChanged = true;
    }

    bool pdfChangedPol = false;

    if (parameters.isAvailable(
            Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_SET_POL)) {

        m_pdfSetPol = parameters.getLastAvailable().toString();

        info(__func__,
                ElemUtils::Formatter() << "Polarized PDF set changed to : "
                        << m_pdfSetPol);

        pdfChangedPol = true;
    }

    if (parameters.isAvailable(
            Partons0117BorderFunctionModel::PARAMETER_NAME_PARTONS0117BORDERFUNCTIONMODEL_PDF_MEMBER_POL)) {

        m_pdfMemberPol = parameters.getLastAvailable().toInt();

        info(__func__,
                ElemUtils::Formatter() << "Polarized PDF member changed to : "
                        << m_pdfMemberPol);

        pdfChangedPol = true;
    }

    if (pdfChanged)
        setPDF();
    if (pdfChangedPol)
        setPDFPol();
}

std::string Partons0117BorderFunctionModel::toString() const {
    GPDModule::toString();
}

PartonDistribution Partons0117BorderFunctionModel::computeH() {

    //variables
    double HuVal, HuSea;
    double HdVal, HdSea;

    //calculate
    double pdfu = m_pPdf->xfxQ2(2, m_xi, m_MuF2) / m_xi;
    double pdfubar = m_pPdf->xfxQ2(-2, m_xi, m_MuF2) / m_xi;
    double pdfd = m_pPdf->xfxQ2(1, m_xi, m_MuF2) / m_xi;
    double pdfdbar = m_pPdf->xfxQ2(-1, m_xi, m_MuF2) / m_xi;

    HuVal = (pdfu - pdfubar) * tDependance(m_par_H_tDep_uVal)
            * skewnessFunction(m_par_H_skew_uVal);
    HuSea = pdfubar * tDependance(m_par_H_tDep_uSea)
            * skewnessFunction(m_par_H_skew_uSea);
    HdVal = (pdfd - pdfdbar) * tDependance(m_par_H_tDep_dVal)
            * skewnessFunction(m_par_H_skew_dVal);
    HdSea = pdfdbar * tDependance(m_par_H_tDep_dSea)
            * skewnessFunction(m_par_H_skew_dSea);

    //get results
    QuarkDistribution quarkDistribution_u(QuarkFlavor::UP);
    QuarkDistribution quarkDistribution_d(QuarkFlavor::DOWN);
    QuarkDistribution quarkDistribution_s(QuarkFlavor::STRANGE);

    quarkDistribution_u.setQuarkDistribution(HuVal + HuSea);
    quarkDistribution_d.setQuarkDistribution(HdVal + HdSea);
    quarkDistribution_s.setQuarkDistribution(0.);

    quarkDistribution_u.setQuarkDistributionPlus(HuVal + 2 * HuSea);
    quarkDistribution_d.setQuarkDistributionPlus(HdVal + 2 * HdSea);
    quarkDistribution_s.setQuarkDistributionPlus(0.);

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

PartonDistribution Partons0117BorderFunctionModel::computeHt() {

    //variables
    double HtuVal, HtuSea;
    double HtdVal, HtdSea;

    //calculate
    double pdfu = m_pPdfPol->xfxQ2(2, m_xi, m_MuF2) / m_xi;
    double pdfubar = m_pPdfPol->xfxQ2(-2, m_xi, m_MuF2) / m_xi;
    double pdfd = m_pPdfPol->xfxQ2(1, m_xi, m_MuF2) / m_xi;
    double pdfdbar = m_pPdfPol->xfxQ2(-1, m_xi, m_MuF2) / m_xi;

    HtuVal = (pdfu - pdfubar) * tDependance(m_par_Ht_tDep_uVal)
            * skewnessFunction(m_par_Ht_skew_uVal);
    HtuSea = pdfubar * tDependance(m_par_Ht_tDep_uSea)
            * skewnessFunction(m_par_Ht_skew_uSea);
    HtdVal = (pdfd - pdfdbar) * tDependance(m_par_Ht_tDep_dVal)
            * skewnessFunction(m_par_Ht_skew_dVal);
    HtdSea = pdfdbar * tDependance(m_par_Ht_tDep_dSea)
            * skewnessFunction(m_par_Ht_skew_dSea);

    //get results
    QuarkDistribution quarkDistribution_u(QuarkFlavor::UP);
    QuarkDistribution quarkDistribution_d(QuarkFlavor::DOWN);
    QuarkDistribution quarkDistribution_s(QuarkFlavor::STRANGE);

    quarkDistribution_u.setQuarkDistribution(HtuVal + HtuSea);
    quarkDistribution_d.setQuarkDistribution(HtdVal + HtdSea);
    quarkDistribution_s.setQuarkDistribution(0.);

    quarkDistribution_u.setQuarkDistributionPlus(HtuVal + 2 * HtuSea);
    quarkDistribution_d.setQuarkDistributionPlus(HtdVal + 2 * HtdSea);
    quarkDistribution_s.setQuarkDistributionPlus(0.);

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

PartonDistribution Partons0117BorderFunctionModel::computeE() {

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

PartonDistribution Partons0117BorderFunctionModel::computeEt() {

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

double Partons0117BorderFunctionModel::skewnessFunction(
        const std::vector<double>& parameters) const {

    if (parameters.size() != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Wrong number of parameters, is "
                        << parameters.size() << " should be 1");
    }

    return parameters.at(0) / pow(1. - pow(m_xi, 2), 2);
}

double Partons0117BorderFunctionModel::tDependance(
        const std::vector<double>& parameters) const {

    if (parameters.size() != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Wrong number of parameters, is "
                        << parameters.size() << " should be 1");
    }

    return pow(m_xi, -1 * parameters.at(0) * m_t);
}

void Partons0117BorderFunctionModel::setPDF() {

    if (m_pPdf) {
        delete m_pPdf;
        m_pPdf = 0;
    }

    m_pPdf = LHAPDF::mkPDF(m_pdfSet, m_pdfMember);

    info(__func__,
            ElemUtils::Formatter() << "Unpolarized PDF: set: " << m_pdfSet
                    << " member: " << m_pdfMember);
}

void Partons0117BorderFunctionModel::setPDFPol() {

    if (m_pPdfPol) {
        delete m_pPdfPol;
        m_pPdfPol = 0;
    }

    m_pPdfPol = LHAPDF::mkPDF(m_pdfSetPol, m_pdfMemberPol);

    info(__func__,
            ElemUtils::Formatter() << "Polarized PDF: set: " << m_pdfSetPol
                    << " member: " << m_pdfMemberPol);
}

double Partons0117BorderFunctionModel::getParEN() const {
    return m_par_E_N;
}

void Partons0117BorderFunctionModel::setParEN(double parEN) {
    m_par_E_N = parEN;
}

double Partons0117BorderFunctionModel::getParEtN() const {
    return m_par_Et_N;
}

void Partons0117BorderFunctionModel::setParEtN(double parEtN) {
    m_par_Et_N = parEtN;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHSkewDSea() const {
    return m_par_H_skew_dSea;
}

void Partons0117BorderFunctionModel::setParHSkewDSea(
        const std::vector<double>& parHSkewDSea) {
    m_par_H_skew_dSea = parHSkewDSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHSkewDVal() const {
    return m_par_H_skew_dVal;
}

void Partons0117BorderFunctionModel::setParHSkewDVal(
        const std::vector<double>& parHSkewDVal) {
    m_par_H_skew_dVal = parHSkewDVal;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHSkewUSea() const {
    return m_par_H_skew_uSea;
}

void Partons0117BorderFunctionModel::setParHSkewUSea(
        const std::vector<double>& parHSkewUSea) {
    m_par_H_skew_uSea = parHSkewUSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHSkewUVal() const {
    return m_par_H_skew_uVal;
}

void Partons0117BorderFunctionModel::setParHSkewUVal(
        const std::vector<double>& parHSkewUVal) {
    m_par_H_skew_uVal = parHSkewUVal;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHTDepDSea() const {
    return m_par_H_tDep_dSea;
}

void Partons0117BorderFunctionModel::setParHTDepDSea(
        const std::vector<double>& parHTDepDSea) {
    m_par_H_tDep_dSea = parHTDepDSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHTDepDVal() const {
    return m_par_H_tDep_dVal;
}

void Partons0117BorderFunctionModel::setParHTDepDVal(
        const std::vector<double>& parHTDepDVal) {
    m_par_H_tDep_dVal = parHTDepDVal;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHTDepUSea() const {
    return m_par_H_tDep_uSea;
}

void Partons0117BorderFunctionModel::setParHTDepUSea(
        const std::vector<double>& parHTDepUSea) {
    m_par_H_tDep_uSea = parHTDepUSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHTDepUVal() const {
    return m_par_H_tDep_uVal;
}

void Partons0117BorderFunctionModel::setParHTDepUVal(
        const std::vector<double>& parHTDepUVal) {
    m_par_H_tDep_uVal = parHTDepUVal;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtSkewDSea() const {
    return m_par_Ht_skew_dSea;
}

void Partons0117BorderFunctionModel::setParHtSkewDSea(
        const std::vector<double>& parHtSkewDSea) {
    m_par_Ht_skew_dSea = parHtSkewDSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtSkewDVal() const {
    return m_par_Ht_skew_dVal;
}

void Partons0117BorderFunctionModel::setParHtSkewDVal(
        const std::vector<double>& parHtSkewDVal) {
    m_par_Ht_skew_dVal = parHtSkewDVal;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtSkewUSea() const {
    return m_par_Ht_skew_uSea;
}

void Partons0117BorderFunctionModel::setParHtSkewUSea(
        const std::vector<double>& parHtSkewUSea) {
    m_par_Ht_skew_uSea = parHtSkewUSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtSkewUVal() const {
    return m_par_Ht_skew_uVal;
}

void Partons0117BorderFunctionModel::setParHtSkewUVal(
        const std::vector<double>& parHtSkewUVal) {
    m_par_Ht_skew_uVal = parHtSkewUVal;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtTDepDSea() const {
    return m_par_Ht_tDep_dSea;
}

void Partons0117BorderFunctionModel::setParHtTDepDSea(
        const std::vector<double>& parHtTDepDSea) {
    m_par_Ht_tDep_dSea = parHtTDepDSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtTDepDVal() const {
    return m_par_Ht_tDep_dVal;
}

void Partons0117BorderFunctionModel::setParHtTDepDVal(
        const std::vector<double>& parHtTDepDVal) {
    m_par_Ht_tDep_dVal = parHtTDepDVal;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtTDepUSea() const {
    return m_par_Ht_tDep_uSea;
}

void Partons0117BorderFunctionModel::setParHtTDepUSea(
        const std::vector<double>& parHtTDepUSea) {
    m_par_Ht_tDep_uSea = parHtTDepUSea;
}

const std::vector<double>& Partons0117BorderFunctionModel::getParHtTDepUVal() const {
    return m_par_Ht_tDep_uVal;
}

void Partons0117BorderFunctionModel::setParHtTDepUVal(
        const std::vector<double>& parHtTDepUVal) {
    m_par_Ht_tDep_uVal = parHtTDepUVal;
}

double Partons0117BorderFunctionModel::getDelta(double value, double x) const {
    return log(fabs(value)) / log(1 / x);
}

double Partons0117BorderFunctionModel::getDeltaUVal() const {

    const double x = 1.E-6;
    const double Q2 = 2.;

    double pdfu = m_pPdf->xfxQ2(2, x, Q2) / x;
    double pdfubar = m_pPdf->xfxQ2(-2, x, Q2) / x;

    return getDelta(pdfu - pdfubar, x);
}

double Partons0117BorderFunctionModel::getDeltaUSea() const {

    const double x = 1.E-6;
    const double Q2 = 2.;

    double pdfubar = m_pPdf->xfxQ2(-2, x, Q2) / x;

    return getDelta(pdfubar, x);
}

double Partons0117BorderFunctionModel::getDeltaDVal() const {

    const double x = 1.E-6;
    const double Q2 = 2.;

    double pdfd = m_pPdf->xfxQ2(1, x, Q2) / x;
    double pdfdbar = m_pPdf->xfxQ2(-1, x, Q2) / x;

    return getDelta(pdfd - pdfdbar, x);
}

double Partons0117BorderFunctionModel::getDeltaDSea() const {

    const double x = 1.E-6;
    const double Q2 = 2.;

    double pdfdbar = m_pPdf->xfxQ2(-1, x, Q2) / x;

    return getDelta(pdfdbar, x);
}

double Partons0117BorderFunctionModel::getDeltaUValPol() const {

    const double x = 1.E-5;
    const double Q2 = 2.;

    double pdfu = m_pPdfPol->xfxQ2(2, x, Q2) / x;
    double pdfubar = m_pPdfPol->xfxQ2(-2, x, Q2) / x;

    return getDelta(pdfu - pdfubar, x);
}

double Partons0117BorderFunctionModel::getDeltaUSeaPol() const {

    const double x = 1.E-5;
    const double Q2 = 2.;

    double pdfubar = m_pPdfPol->xfxQ2(-2, x, Q2) / x;

    return getDelta(pdfubar, x);
}

double Partons0117BorderFunctionModel::getDeltaDValPol() const {

    const double x = 1.E-5;
    const double Q2 = 2.;

    double pdfd = m_pPdfPol->xfxQ2(1, x, Q2) / x;
    double pdfdbar = m_pPdfPol->xfxQ2(-1, x, Q2) / x;

    return getDelta(pdfd - pdfdbar, x);
}

double Partons0117BorderFunctionModel::getDeltaDSeaPol() const {

    const double x = 1.E-5;
    const double Q2 = 2.;

    double pdfdbar = m_pPdfPol->xfxQ2(-1, x, Q2) / x;

    return getDelta(pdfdbar, x);
}

PARTONS::GPDModule* Partons0117BorderFunctionModel::getGPDModule() const{
    return m_pGPDModule;
}
void Partons0117BorderFunctionModel::setGPDModule(PARTONS::GPDModule* gpdModule){

    m_pModuleObjectFactory->updateModulePointerReference(
            m_pGPDModule, gpdModule);
    m_pGPDModule = gpdModule;
}
