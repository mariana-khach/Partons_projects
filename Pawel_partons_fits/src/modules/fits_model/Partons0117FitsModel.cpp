#include "../../../include/modules/fits_model/Partons0117FitsModel.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/GenericType.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/beans/channel/ChannelType.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/process/ProcessModule.h>
#include <partons/Partons.h>
#include <partons/ServiceObjectRegistry.h>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

#include "../../../include/modules/convol_coeff_function/DVCSCFFDispersionForHLikeStandardForELike.h"
#include "../../../include/modules/gpd_border_function/Partons0117BorderFunctionModel.h"
#include "../../../include/modules/gpd_subtraction_constant/Partons0117SubtractionConstantModel.h"
#include "../../../include/modules/minimizer_model/FitsMinimizerModule.h"
#include "../../../include/services/FitsService.h"

using namespace PARTONS;

const unsigned int Partons0117FitsModel::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new Partons0117FitsModel("Partons0117FitsModel"));

Partons0117FitsModel::Partons0117FitsModel(const std::string& className) :
        FitsModelModule(className), m_pPartons0117BorderFunctionModel(0), m_pPartons0117SubtractionConstantModel(
                0) {

    m_parameters.add("par_H_skew_uVal", 0.);
    m_parameters.add("par_H_skew_uSea", 0.);
    m_parameters.add("par_H_skew_dVal", 0.);
    m_parameters.add("par_H_skew_dSea", 0.);
    m_parameters.add("par_Ht_skew_uVal", 0.);
    m_parameters.add("par_Ht_skew_uSea", 0.);
    m_parameters.add("par_Ht_skew_dVal", 0.);
    m_parameters.add("par_Ht_skew_dSea", 0.);

    m_parameters.add("par_H_a_Val", 7.3625E-01);
    m_parameters.add("par_H_a_Sea", 5.);
    m_parameters.add("par_Ht_a_Val", 2.5);
    m_parameters.add("par_Ht_a_Sea", 0.2);

    m_parameters.add("par_Sub_C", -1.);
    m_parameters.add("par_Sub_a", 0.);

    m_parameters.add("par_E_N", -5.);
    m_parameters.add("par_Et_N", -0.5);
}

Partons0117FitsModel::Partons0117FitsModel(const Partons0117FitsModel &other) :
        FitsModelModule(other), m_pPartons0117BorderFunctionModel(
                other.m_pPartons0117BorderFunctionModel), m_pPartons0117SubtractionConstantModel(
                other.m_pPartons0117SubtractionConstantModel) {
}

Partons0117FitsModel::~Partons0117FitsModel() {
}

void Partons0117FitsModel::resolveObjectDependencies() {
}

void Partons0117FitsModel::configure(const ElemUtils::Parameters &parameters) {

    //mother class
    FitsModelModule::configure(parameters);

    //loop over parameters
    for (size_t i = 0; i < m_parameters.size(); i++) {

        //get key
        std::string key = m_parameters.key(i);

        //check if there
        if (parameters.isAvailable(key)) {

            //set
            m_parameters.update(key, parameters.getLastAvailable().toDouble());

            //info
            info(__func__,
                    ElemUtils::Formatter() << "Parameter " << key
                            << " updated to : "
                            << parameters.getLastAvailable().toDouble());
        }
    }
}

Partons0117FitsModel* Partons0117FitsModel::clone() const {
    return new Partons0117FitsModel(*this);
}

void Partons0117FitsModel::initModule() {

    //run mother class
    FitsModelModule::initModule();

    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //get cff model
    if (pFitsService->getCFFModule(ChannelType::DVCS)->getClassName()
            != "DVCSCFFDispersionForHLikeStandardForELike") {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "This fit model requires DVCSCFFDispersionForHLikeStandardForELike to be used as DVCS/CFF model");
    }

    DVCSCFFDispersionForHLikeStandardForELike* cffModule =
            static_cast<DVCSCFFDispersionForHLikeStandardForELike*>(pFitsService->getCFFModule(
                    ChannelType::DVCS));

    //get border function and subtraction constant models
    if (pFitsService->getGPDModule(ChannelType::DVCS)->getClassName()
            != "Partons0117BorderFunctionModel") {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "This fit model requires Partons0117BorderFunctionModel to be used as border function model");
    }

    m_pPartons0117BorderFunctionModel =
            static_cast<Partons0117BorderFunctionModel*>(pFitsService->getGPDModule(
                    ChannelType::DVCS));

    if (cffModule->getSubtractionConstantModule()->getClassName()
            != "Partons0117SubtractionConstantModel") {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "This fit model requires Partons0117SubtractionConstantModel to be used as subtraction constant model");
    }

    m_pPartons0117SubtractionConstantModel =
            static_cast<Partons0117SubtractionConstantModel*>(cffModule->getSubtractionConstantModule());
}

void Partons0117FitsModel::isModuleWellConfigured() {
    FitsModelModule::isModuleWellConfigured();
}

void Partons0117FitsModel::initParameters() {

    //TODO probably one should add FitsServiceObjectRegistry to PartonsFits to avoid casting
    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //get minimizer
    FitsMinimizerModule* pMinimizer = pFitsService->getMinimizer();

    //parameters
    pMinimizer->SetFixedVariable("par_H_skew_uVal",
            getSkewVal(m_pPartons0117BorderFunctionModel->getDeltaUVal()));
    pMinimizer->SetFixedVariable("par_H_skew_uSea",
            getSkewSea(m_pPartons0117BorderFunctionModel->getDeltaUSea()));
    pMinimizer->SetFixedVariable("par_H_skew_dVal",
            getSkewVal(m_pPartons0117BorderFunctionModel->getDeltaDVal()));
    pMinimizer->SetFixedVariable("par_H_skew_dSea",
            getSkewSea(m_pPartons0117BorderFunctionModel->getDeltaDSea()));
    pMinimizer->SetFixedVariable("par_Ht_skew_uVal",
            getSkewVal(m_pPartons0117BorderFunctionModel->getDeltaUValPol()));
    pMinimizer->SetFixedVariable("par_Ht_skew_uSea",
            getSkewSea(m_pPartons0117BorderFunctionModel->getDeltaUSeaPol()));
    pMinimizer->SetFixedVariable("par_Ht_skew_dVal",
            getSkewVal(m_pPartons0117BorderFunctionModel->getDeltaDValPol()));
    pMinimizer->SetFixedVariable("par_Ht_skew_dSea",
            getSkewSea(m_pPartons0117BorderFunctionModel->getDeltaDSeaPol()));

    double initialStep = 1.E-1;

    pMinimizer->SetFixedVariable("par_H_a_Val",
            m_parameters.get("par_H_a_Val").toDouble());
    pMinimizer->SetLimitedVariable("par_H_a_Sea",
            m_parameters.get("par_H_a_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 100.));

    pMinimizer->SetLimitedVariable("par_Ht_a_Val",
            m_parameters.get("par_Ht_a_Val").toDouble(), initialStep,
            std::pair<double, double>(0., 100.));
    pMinimizer->SetLimitedVariable("par_Ht_a_Sea",
            m_parameters.get("par_Ht_a_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 100.));

    pMinimizer->SetVariable("par_Sub_C",
            m_parameters.get("par_Sub_C").toDouble(), initialStep);
    pMinimizer->SetVariable("par_Sub_a",
            m_parameters.get("par_Sub_a").toDouble(), initialStep);

    pMinimizer->SetLimitedVariable("par_E_N",
            m_parameters.get("par_E_N").toDouble(), initialStep,
            std::pair<double, double>(-10., 10.));
    pMinimizer->SetLimitedVariable("par_Et_N",
            m_parameters.get("par_Et_N").toDouble(), initialStep,
            std::pair<double, double>(-10., 10.));
}

void Partons0117FitsModel::updateParameters(const double* Var) {

    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //reset kinematics
    // resetPreviousKinematics() removed in new PARTONS

    m_pPartons0117BorderFunctionModel->setParHSkewUVal(
            std::vector<double>(1, Var[0]));
    m_pPartons0117BorderFunctionModel->setParHSkewUSea(
            std::vector<double>(1, Var[1]));
    m_pPartons0117BorderFunctionModel->setParHSkewDVal(
            std::vector<double>(1, Var[2]));
    m_pPartons0117BorderFunctionModel->setParHSkewDSea(
            std::vector<double>(1, Var[3]));
    m_pPartons0117BorderFunctionModel->setParHtSkewUVal(
            std::vector<double>(1, Var[4]));
    m_pPartons0117BorderFunctionModel->setParHtSkewUSea(
            std::vector<double>(1, Var[5]));
    m_pPartons0117BorderFunctionModel->setParHtSkewDVal(
            std::vector<double>(1, Var[6]));
    m_pPartons0117BorderFunctionModel->setParHtSkewDSea(
            std::vector<double>(1, Var[7]));

    m_pPartons0117BorderFunctionModel->setParHTDepUVal(
            std::vector<double>(1, Var[8]));
    m_pPartons0117BorderFunctionModel->setParHTDepDVal(
            std::vector<double>(1, Var[8]));
    m_pPartons0117BorderFunctionModel->setParHTDepUSea(
            std::vector<double>(1, Var[9]));
    m_pPartons0117BorderFunctionModel->setParHTDepDSea(
            std::vector<double>(1, Var[9]));
    m_pPartons0117BorderFunctionModel->setParHtTDepUVal(
            std::vector<double>(1, Var[10]));
    m_pPartons0117BorderFunctionModel->setParHtTDepDVal(
            std::vector<double>(1, Var[10]));
    m_pPartons0117BorderFunctionModel->setParHtTDepUSea(
            std::vector<double>(1, Var[11]));
    m_pPartons0117BorderFunctionModel->setParHtTDepDSea(
            std::vector<double>(1, Var[11]));

    m_pPartons0117SubtractionConstantModel->setParC(Var[12]);
    m_pPartons0117SubtractionConstantModel->setParA(Var[13]);

    m_pPartons0117BorderFunctionModel->setParEN(Var[14]);
    m_pPartons0117BorderFunctionModel->setParEtN(Var[15]);

    std::stringstream ss;
    ss << "\n";

    for (size_t i = 0; i < m_parameters.size(); i++) {
        ss << "Actual parameters: par_" << i << " " << Var[i] << "\n";
    }

    info(__func__, ss.str());
}

double Partons0117FitsModel::getSkewVal(double delta) const {
    return 6 * tgamma(1. - delta) / tgamma(4. - delta) * pow(2., -delta)
            * (1. - delta);
}

double Partons0117FitsModel::getSkewSea(double delta) const {
    return 60 * tgamma(1. - delta) / tgamma(6. - delta) * pow(2., -delta)
            * (delta * delta - 3 * delta + 2);
}
