#include "../../../include/modules/fits_model/Partons0118FitsModel.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/GenericType.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/beans/channel/ChannelType.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/process/ProcessModule.h>
#include <partons/Partons.h>
#include <partons/ServiceObjectRegistry.h>
#include <stddef.h>
#include <sstream>
#include <utility>
#include <vector>

#include "../../../include/modules/convol_coeff_function/DVCSCFFDispersionForHLikeStandardForELike.h"
#include "../../../include/modules/gpd_border_function/Partons0118BorderFunctionModel.h"
#include "../../../include/modules/gpd_subtraction_constant/Partons0118SubtractionConstantModel.h"
#include "../../../include/modules/minimizer_model/FitsMinimizerModule.h"
#include "../../../include/services/FitsService.h"

using namespace PARTONS;

const unsigned int Partons0118FitsModel::classId =
        BaseObjectRegistry::getInstance()->registerBaseObject(
                new Partons0118FitsModel("Partons0118FitsModel"));

Partons0118FitsModel::Partons0118FitsModel(const std::string& className) :
        FitsModelModule(className), m_pPartons0118BorderFunctionModel(0), m_pPartons0118SubtractionConstantModel(
                0) {

    m_parameters.add("par_H_skew_Val", 1.1);
    m_parameters.add("par_H_skew_Sea", 1.1);
    m_parameters.add("par_Ht_skew_Val", 1.1);
    m_parameters.add("par_Ht_skew_Sea", 1.1);

    m_parameters.add("par_H_tDep_A0_uVal", 0.95);
    m_parameters.add("par_H_tDep_A1_uVal", 0.475);
    m_parameters.add("par_H_tDep_A2_uVal", 1.31);
    m_parameters.add("par_H_tDep_A0_dVal", 0.85);
    m_parameters.add("par_H_tDep_A1_dVal", 0.6);
    m_parameters.add("par_H_tDep_A2_dVal", 3.15);

    m_parameters.add("par_H_tDep_A0_Sea", 0.95);
    m_parameters.add("par_H_tDep_A1_Sea", 0.4);
    m_parameters.add("par_H_tDep_A2_Sea", 100.);
    m_parameters.add("par_Ht_tDep_A0_Val", 0.2);
    m_parameters.add("par_Ht_tDep_A1_Val", 0.3);
    m_parameters.add("par_Ht_tDep_A2_Val", 0.4);
    m_parameters.add("par_Ht_tDep_A0_Sea", 0.2);
    m_parameters.add("par_Ht_tDep_A1_Sea", 0.3);
    m_parameters.add("par_Ht_tDep_A2_Sea", 0.4);

    m_parameters.add("par_E_N", -5.);
    m_parameters.add("par_Et_N", -0.5);
}

Partons0118FitsModel::Partons0118FitsModel(const Partons0118FitsModel &other) :
        FitsModelModule(other), m_pPartons0118BorderFunctionModel(
                other.m_pPartons0118BorderFunctionModel), m_pPartons0118SubtractionConstantModel(
                other.m_pPartons0118SubtractionConstantModel) {
}

Partons0118FitsModel::~Partons0118FitsModel() {
}

void Partons0118FitsModel::resolveObjectDependencies() {
}

void Partons0118FitsModel::configure(const ElemUtils::Parameters &parameters) {

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

Partons0118FitsModel* Partons0118FitsModel::clone() const {
    return new Partons0118FitsModel(*this);
}

void Partons0118FitsModel::initModule() {

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
            != "Partons0118BorderFunctionModel") {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "This fit model requires Partons0118BorderFunctionModel to be used as border function model");
    }

    m_pPartons0118BorderFunctionModel =
            static_cast<Partons0118BorderFunctionModel*>(pFitsService->getGPDModule(
                    ChannelType::DVCS));

    if (cffModule->getSubtractionConstantModule()->getClassName()
            != "Partons0118SubtractionConstantModel") {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "This fit model requires Partons0118SubtractionConstantModel to be used as subtraction constant model");
    }

    m_pPartons0118SubtractionConstantModel =
            static_cast<Partons0118SubtractionConstantModel*>(cffModule->getSubtractionConstantModule());
}

void Partons0118FitsModel::isModuleWellConfigured() {
    FitsModelModule::isModuleWellConfigured();
}

void Partons0118FitsModel::initParameters() {

    //TODO probably one should add FitsServiceObjectRegistry to PartonsFits to avoid casting
    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //get minimizer
    FitsMinimizerModule* pMinimizer = pFitsService->getMinimizer();

    //parameters
    double initialStep = 1.E-1;

    pMinimizer->SetLimitedVariable("par_H_skew_Val",
            m_parameters.get("par_H_skew_Val").toDouble(), initialStep,
            std::pair<double, double>(0.2, 2.));
    pMinimizer->SetLimitedVariable("par_H_skew_Sea",
            m_parameters.get("par_H_skew_Sea").toDouble(), initialStep,
            std::pair<double, double>(0.2, 2.));
    pMinimizer->SetLimitedVariable("par_Ht_skew_Val",
            m_parameters.get("par_Ht_skew_Val").toDouble(), initialStep,
            std::pair<double, double>(0.2, 2.));
    pMinimizer->SetLimitedVariable("par_Ht_skew_Sea",
            m_parameters.get("par_Ht_skew_Sea").toDouble(), initialStep,
            std::pair<double, double>(0.2, 2.));

    pMinimizer->SetFixedVariable("par_H_tDep_A0_uVal",
            m_parameters.get("par_H_tDep_A0_uVal").toDouble());
    pMinimizer->SetFixedVariable("par_H_tDep_A1_uVal",
            m_parameters.get("par_H_tDep_A1_uVal").toDouble());
    pMinimizer->SetFixedVariable("par_H_tDep_A2_uVal",
            m_parameters.get("par_H_tDep_A2_uVal").toDouble());
    pMinimizer->SetFixedVariable("par_H_tDep_A0_dVal",
            m_parameters.get("par_H_tDep_A0_dVal").toDouble());
    pMinimizer->SetFixedVariable("par_H_tDep_A1_dVal",
            m_parameters.get("par_H_tDep_A1_dVal").toDouble());
    pMinimizer->SetFixedVariable("par_H_tDep_A2_dVal",
            m_parameters.get("par_H_tDep_A2_dVal").toDouble());

    pMinimizer->SetLimitedVariable("par_H_tDep_A0_Sea",
            m_parameters.get("par_H_tDep_A0_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_H_tDep_A1_Sea",
            m_parameters.get("par_H_tDep_A1_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_H_tDep_A2_Sea",
            m_parameters.get("par_H_tDep_A2_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_Ht_tDep_A0_Val",
            m_parameters.get("par_Ht_tDep_A0_Val").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_Ht_tDep_A1_Val",
            m_parameters.get("par_Ht_tDep_A1_Val").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_Ht_tDep_A2_Val",
            m_parameters.get("par_Ht_tDep_A2_Val").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_Ht_tDep_A0_Sea",
            m_parameters.get("par_Ht_tDep_A0_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_Ht_tDep_A1_Sea",
            m_parameters.get("par_Ht_tDep_A1_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));
    pMinimizer->SetLimitedVariable("par_Ht_tDep_A2_Sea",
            m_parameters.get("par_Ht_tDep_A2_Sea").toDouble(), initialStep,
            std::pair<double, double>(0., 1000.));

    pMinimizer->SetLimitedVariable("par_E_N",
            m_parameters.get("par_E_N").toDouble(), initialStep,
            std::pair<double, double>(-10., 10.));
    pMinimizer->SetLimitedVariable("par_Et_N",
            m_parameters.get("par_Et_N").toDouble(), initialStep,
            std::pair<double, double>(-10., 10.));
}

void Partons0118FitsModel::updateParameters(const double* Var) {

    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //reset kinematics
    // resetPreviousKinematics() removed in new PARTONS

    m_pPartons0118BorderFunctionModel->setParHSkewUVal(
            std::vector<double>(1, Var[0]));
    m_pPartons0118BorderFunctionModel->setParHSkewDVal(
            std::vector<double>(1, Var[0]));
    m_pPartons0118BorderFunctionModel->setParHSkewUSea(
            std::vector<double>(1, Var[1]));
    m_pPartons0118BorderFunctionModel->setParHSkewDSea(
            std::vector<double>(1, Var[1]));
    m_pPartons0118BorderFunctionModel->setParHSkewSSea(
            std::vector<double>(1, Var[1]));
    m_pPartons0118SubtractionConstantModel->setParHSkewUVal(
            std::vector<double>(1, Var[0]));
    m_pPartons0118SubtractionConstantModel->setParHSkewDVal(
            std::vector<double>(1, Var[0]));
    m_pPartons0118SubtractionConstantModel->setParHSkewUSea(
            std::vector<double>(1, Var[1]));
    m_pPartons0118SubtractionConstantModel->setParHSkewDSea(
            std::vector<double>(1, Var[1]));
    m_pPartons0118SubtractionConstantModel->setParHSkewSSea(
            std::vector<double>(1, Var[1]));
    m_pPartons0118BorderFunctionModel->setParHtSkewUVal(
            std::vector<double>(1, Var[2]));
    m_pPartons0118BorderFunctionModel->setParHtSkewDVal(
            std::vector<double>(1, Var[2]));
    m_pPartons0118BorderFunctionModel->setParHtSkewUSea(
            std::vector<double>(1, Var[3]));
    m_pPartons0118BorderFunctionModel->setParHtSkewDSea(
            std::vector<double>(1, Var[3]));
    m_pPartons0118BorderFunctionModel->setParHtSkewSSea(
            std::vector<double>(1, Var[3]));

    std::vector<double> parameters3(3, 0.);

    for (size_t i = 0; i < 3; i++)
        parameters3[i] = Var[4 + i];
    m_pPartons0118BorderFunctionModel->setParHTDepUVal(parameters3);
    m_pPartons0118SubtractionConstantModel->setParHTDepUVal(parameters3);

    for (size_t i = 0; i < 3; i++)
        parameters3[i] = Var[7 + i];
    m_pPartons0118BorderFunctionModel->setParHTDepDVal(parameters3);
    m_pPartons0118SubtractionConstantModel->setParHTDepDVal(parameters3);

    for (size_t i = 0; i < 3; i++)
        parameters3[i] = Var[10 + i];
    m_pPartons0118BorderFunctionModel->setParHTDepUSea(parameters3);
    m_pPartons0118BorderFunctionModel->setParHTDepDSea(parameters3);
    m_pPartons0118BorderFunctionModel->setParHTDepSSea(parameters3);
    m_pPartons0118SubtractionConstantModel->setParHTDepUSea(parameters3);
    m_pPartons0118SubtractionConstantModel->setParHTDepDSea(parameters3);
    m_pPartons0118SubtractionConstantModel->setParHTDepSSea(parameters3);

    for (size_t i = 0; i < 3; i++)
        parameters3[i] = Var[13 + i];
    m_pPartons0118BorderFunctionModel->setParHtTDepUVal(parameters3);
    m_pPartons0118BorderFunctionModel->setParHtTDepDVal(parameters3);

    for (size_t i = 0; i < 3; i++)
        parameters3[i] = Var[16 + i];
    m_pPartons0118BorderFunctionModel->setParHtTDepUSea(parameters3);
    m_pPartons0118BorderFunctionModel->setParHtTDepDSea(parameters3);
    m_pPartons0118BorderFunctionModel->setParHtTDepSSea(parameters3);

    m_pPartons0118BorderFunctionModel->setParEN(Var[19]);
    m_pPartons0118BorderFunctionModel->setParEtN(Var[20]);

    std::stringstream ss;
    ss << "\n";

    for (size_t i = 0; i < m_parameters.size(); i++) {
        ss << "Actual parameters: par_" << i << " " << Var[i] << "\n";
    }

    info(__func__, ss.str());
}

//double Partons0118FitsModel::getSkewVal(double delta) const {
//    return 6 * tgamma(1. - delta) / tgamma(4. - delta) * pow(2., -delta)
//            * (1. - delta);
//}
//
//double Partons0118FitsModel::getSkewSea(double delta) const {
//    return 60 * tgamma(1. - delta) / tgamma(6. - delta) * pow(2., -delta)
//            * (delta * delta - 3 * delta + 2);
//}
