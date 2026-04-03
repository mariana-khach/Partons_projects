#include "../../../include/modules/gpd_subtraction_constant/Partons0118SubtractionConstantModel.h"

#include <ElementaryUtils/string_utils/Formatter.h>
#include <NumA/functor/one_dimension/Functor1D.h>
#include <NumA/integration/one_dimension/Integrator1D.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <cmath>

#include "../../../include/modules/pdf/PdfParameterization.h"

using namespace PARTONS;

const unsigned int Partons0118SubtractionConstantModel::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new Partons0118SubtractionConstantModel(
                        "Partons0118SubtractionConstantModel"));

const std::string Partons0118SubtractionConstantModel::PARAMETER_NAME_PARTONS0118SUBTRACTIONCONSTANTMODEL_PDF_REPLICA =
        "partons_0118_sc_pdf_replica";

Partons0118SubtractionConstantModel::Partons0118SubtractionConstantModel(
        const std::string& className) :
        GPDSubtractionConstantModule(className), MathIntegratorModule() {

    m_pPdf = new PdfParameterization();

    m_pdfReplica = 0;

    m_tmpEffBeta = 0.;
    m_tmpQuark = QuarkFlavor::UNDEFINED;
    m_tmpIsSea = false;

    m_tmpPar_H_skew.clear();
    m_tmpPar_H_tDep.clear();

    m_tmpD0 = 0.;
    m_tmpD1 = 0.;

    m_par_H_skew_uVal.clear();
    m_par_H_skew_uSea.clear();

    m_par_H_skew_dVal.clear();
    m_par_H_skew_dSea.clear();

    m_par_H_skew_sSea.clear();

    m_par_H_tDep_uVal.clear();
    m_par_H_tDep_uSea.clear();

    m_par_H_tDep_dVal.clear();
    m_par_H_tDep_dSea.clear();

    m_par_H_tDep_sSea.clear();

    initFunctorsForIntegrations();
}

Partons0118SubtractionConstantModel::~Partons0118SubtractionConstantModel() {

    if (m_pPdf) {
        delete m_pPdf;
        m_pPdf = 0;
    }

    if (m_pint_IntegralFunction) {
        delete m_pint_IntegralFunction;
        m_pint_IntegralFunction = 0;
    }
}

Partons0118SubtractionConstantModel* Partons0118SubtractionConstantModel::clone() const {
    return new Partons0118SubtractionConstantModel(*this);
}

Partons0118SubtractionConstantModel::Partons0118SubtractionConstantModel(
        const Partons0118SubtractionConstantModel& other) :
        GPDSubtractionConstantModule(other), MathIntegratorModule(other) {

    if (other.m_pPdf) {
        m_pPdf = other.m_pPdf->clone();
    } else {
        m_pPdf = 0;
    }

    m_pdfReplica = other.m_pdfReplica;

    m_tmpQuark = other.m_tmpQuark;
    m_tmpIsSea = other.m_tmpIsSea;

    m_tmpPar_H_skew = other.m_tmpPar_H_skew;
    m_tmpPar_H_tDep = other.m_tmpPar_H_tDep;

    m_tmpEffBeta = other.m_tmpEffBeta;
    m_tmpD0 = other.m_tmpD0;
    m_tmpD1 = other.m_tmpD1;

    m_par_H_skew_uVal = other.m_par_H_skew_uVal;
    m_par_H_skew_uSea = other.m_par_H_skew_uSea;

    m_par_H_skew_dVal = other.m_par_H_skew_dVal;
    m_par_H_skew_dSea = other.m_par_H_skew_dSea;

    m_par_H_skew_sSea = other.m_par_H_skew_sSea;

    m_par_H_tDep_uVal = other.m_par_H_tDep_uVal;
    m_par_H_tDep_uSea = other.m_par_H_tDep_uSea;

    m_par_H_tDep_dVal = other.m_par_H_tDep_dVal;
    m_par_H_tDep_dSea = other.m_par_H_tDep_dSea;

    m_par_H_tDep_sSea = other.m_par_H_tDep_sSea;

    initFunctorsForIntegrations();
}

void Partons0118SubtractionConstantModel::resolveObjectDependencies() {
    setIntegrator(NumA::IntegratorType1D::DEXP);
}

void Partons0118SubtractionConstantModel::initFunctorsForIntegrations() {

    m_pint_IntegralFunction = NumA::Integrator1D::newIntegrationFunctor(this,
            &Partons0118SubtractionConstantModel::integralFunction);
}

double Partons0118SubtractionConstantModel::integralFunction(double x,
        const std::vector<double>& par) {

    double value = m_pPdf->lambda(m_tmpQuark, m_tmpIsSea, false, m_pdfReplica,
            x, m_t, m_MuF2, m_tmpPar_H_skew, m_tmpPar_H_tDep, 0);

    return (value - m_tmpD0 - x * m_tmpD1)
            * pow(x, -1 * (1. + m_tmpEffBeta));
}

double Partons0118SubtractionConstantModel::computeSubtractionConstantForOneFlavor() {

    std::vector<double> emptyParameters;

    const double intMin = 1.E-6;
    const double intMax = 1.;

    m_tmpEffBeta = m_pPdf->effBeta(m_tmpQuark, m_tmpIsSea, false, m_pdfReplica,
            m_t, m_MuF2, m_tmpPar_H_tDep);
    m_tmpD0 = m_pPdf->lambda(m_tmpQuark, m_tmpIsSea, false, m_pdfReplica,
            intMin, m_t, m_MuF2, m_tmpPar_H_skew, m_tmpPar_H_tDep, 0);
    m_tmpD1 = m_pPdf->lambda(m_tmpQuark, m_tmpIsSea, false, m_pdfReplica,
            intMin, m_t, m_MuF2, m_tmpPar_H_skew, m_tmpPar_H_tDep, 1);

    double result = integrate(m_pint_IntegralFunction, intMin, intMax,
            emptyParameters);

   return 2
            * (result + m_tmpD0 / (0. - m_tmpEffBeta)
                    + m_tmpD1 / (1. - m_tmpEffBeta));
}

PhysicalType<double> Partons0118SubtractionConstantModel::computeSubtractionConstant() {

    //up val
    m_tmpQuark = QuarkFlavor::UP;
    m_tmpIsSea = false;
    m_tmpPar_H_skew = m_par_H_skew_uVal;
    m_tmpPar_H_tDep = m_par_H_tDep_uVal;

    double upVal = computeSubtractionConstantForOneFlavor();

    //up sea
    m_tmpQuark = QuarkFlavor::UP;
    m_tmpIsSea = true;
    m_tmpPar_H_skew = m_par_H_skew_uSea;
    m_tmpPar_H_tDep = m_par_H_tDep_uSea;

    double upSea = computeSubtractionConstantForOneFlavor();

    //down val
    m_tmpQuark = QuarkFlavor::DOWN;
    m_tmpIsSea = false;
    m_tmpPar_H_skew = m_par_H_skew_dVal;
    m_tmpPar_H_tDep = m_par_H_tDep_dVal;

    double downVal = computeSubtractionConstantForOneFlavor();

    //down sea
    m_tmpQuark = QuarkFlavor::DOWN;
    m_tmpIsSea = true;
    m_tmpPar_H_skew = m_par_H_skew_dSea;
    m_tmpPar_H_tDep = m_par_H_tDep_dSea;

    double downSea = computeSubtractionConstantForOneFlavor();

    //strange sea
    m_tmpQuark = QuarkFlavor::STRANGE;
    m_tmpIsSea = true;
    m_tmpPar_H_skew = m_par_H_skew_sSea;
    m_tmpPar_H_tDep = m_par_H_tDep_sSea;

    double strangeSea = computeSubtractionConstantForOneFlavor();




   //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************

    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************
    //*************REMOVE -1 ********************//*************REMOVE -1 ********************
    //*************REMOVE -1 ********************






    //return
    return PhysicalType<double>(-1*(pow(2. / 3., 2) * upVal + pow(1. / 3., 2) * downVal
            + 2 * pow(2. / 3., 2) * upSea + 2 * pow(1. / 3., 2) * downSea
            + 2 * pow(1. / 3., 2) * strangeSea), PhysicalUnit::NONE);
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHSkewDSea() const {
    return m_par_H_skew_dSea;
}

void Partons0118SubtractionConstantModel::setParHSkewDSea(
        const std::vector<double>& parHSkewDSea) {
    m_par_H_skew_dSea = parHSkewDSea;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHSkewDVal() const {
    return m_par_H_skew_dVal;
}

void Partons0118SubtractionConstantModel::setParHSkewDVal(
        const std::vector<double>& parHSkewDVal) {
    m_par_H_skew_dVal = parHSkewDVal;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHSkewSSea() const {
    return m_par_H_skew_sSea;
}

void Partons0118SubtractionConstantModel::setParHSkewSSea(
        const std::vector<double>& parHSkewSSea) {
    m_par_H_skew_sSea = parHSkewSSea;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHSkewUSea() const {
    return m_par_H_skew_uSea;
}

void Partons0118SubtractionConstantModel::setParHSkewUSea(
        const std::vector<double>& parHSkewUSea) {
    m_par_H_skew_uSea = parHSkewUSea;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHSkewUVal() const {
    return m_par_H_skew_uVal;
}

void Partons0118SubtractionConstantModel::setParHSkewUVal(
        const std::vector<double>& parHSkewUVal) {
    m_par_H_skew_uVal = parHSkewUVal;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHTDepDSea() const {
    return m_par_H_tDep_dSea;
}

void Partons0118SubtractionConstantModel::setParHTDepDSea(
        const std::vector<double>& parHTDepDSea) {
    m_par_H_tDep_dSea = parHTDepDSea;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHTDepDVal() const {
    return m_par_H_tDep_dVal;
}

void Partons0118SubtractionConstantModel::setParHTDepDVal(
        const std::vector<double>& parHTDepDVal) {
    m_par_H_tDep_dVal = parHTDepDVal;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHTDepSSea() const {
    return m_par_H_tDep_sSea;
}

void Partons0118SubtractionConstantModel::setParHTDepSSea(
        const std::vector<double>& parHTDepSSea) {
    m_par_H_tDep_sSea = parHTDepSSea;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHTDepUSea() const {
    return m_par_H_tDep_uSea;
}

void Partons0118SubtractionConstantModel::setParHTDepUSea(
        const std::vector<double>& parHTDepUSea) {
    m_par_H_tDep_uSea = parHTDepUSea;
}

const std::vector<double>& Partons0118SubtractionConstantModel::getParHTDepUVal() const {
    return m_par_H_tDep_uVal;
}

void Partons0118SubtractionConstantModel::setParHTDepUVal(
        const std::vector<double>& parHTDepUVal) {
    m_par_H_tDep_uVal = parHTDepUVal;
}
