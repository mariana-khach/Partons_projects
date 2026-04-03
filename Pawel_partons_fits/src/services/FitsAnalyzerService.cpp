#include "../../include/services/FitsAnalyzerService.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/GenericType.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <ElementaryUtils/string_utils/StringUtils.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/automation/Scenario.h>
#include <partons/beans/automation/Task.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <partons/modules/observable/DVCS/DVCSObservable.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/services/automation/AutomationService.h>
#include <partons/services/DVCSConvolCoeffFunctionService.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/ServiceObjectRegistry.h>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <iterator>
#include <utility>

#include "../../include/modules/fits_model/FitsModelModule.h"
#include "../../include/services/FitsService.h"

using namespace PARTONS;

const std::string FitsAnalyzerService::METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE =
        "configureFitAnalyzerService";
const std::string FitsAnalyzerService::METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILES =
        "files";
const std::string FitsAnalyzerService::METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILE_OUTPUT =
        "outputFile";

const std::string FitsAnalyzerService::METHOD_NAME_CONFIGURE_FIT_SERVICE =
        "configureFitService";
const std::string FitsAnalyzerService::METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES =
        "files";
const std::string FitsAnalyzerService::METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_INPUT =
        "inputFile";
const std::string FitsAnalyzerService::METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_OUTPUT =
        "outputFile";

const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_NSTEPS = "nSteps";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_ISLOG = "isLog";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_RANGE_MIN =
        "rangeMin";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_RANGE_MAX =
        "rangeMax";

const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CFF = "evaluateCFF";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CFF_DVCS_AFO_XI =
        "evaluateDVCSCFFAsFunctionOfXi";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CFF_DVCS_AFO_T =
        "evaluateDVCSCFFAsFunctionOfT";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CFF_DVCS_AFO_Q2 =
        "evaluateDVCSCFFAsFunctionOfQ2";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CFF_DVCS_AFO_MUF2 =
        "evaluateDVCSCFFAsFunctionOfF2";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CFF_DVCS_AFO_MUR2 =
        "evaluateDVCSCFFAsFunctionOfuR2";

const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE =
        "evaluateObservable";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE_NAME =
        "observableName";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE_KINEMATICS =
        "kinematics";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_XB =
        "evaluateDVCSObservableAsFunctionOfXb";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_T =
        "evaluateDVCSObservableAsFunctionOfT";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_Q2 =
        "evaluateDVCSObservableAsFunctionOfQ2";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_E =
        "evaluateDVCSObservableAsFunctionOfE";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_PHI =
        "evaluateDVCSObservableAsFunctionOfPhi";

const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2 =
        "evaluateChi2";

const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2_SHAPE_1D =
        "evaluateChi21D";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR =
        "parameter";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR_NAME =
        "name";

const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2_SHAPE_2D =
        "evaluateChi22D";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR =
        "parameter";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_1 =
        "name1";
const std::string FitsAnalyzerService::METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_2 =
        "name2";

// initialize [class]::classId with unique name
const unsigned int FitsAnalyzerService::classId =
        Partons::getInstance()->getBaseObjectRegistry()->registerBaseObject(
                new FitsAnalyzerService("FitsAnalyzerService"));

FitsAnalyzerService::FitsAnalyzerService(const std::string &className) :
        ServiceObject(className) {
}

FitsAnalyzerService::~FitsAnalyzerService() {
}

void FitsAnalyzerService::resolveObjectDependencies() {
}

void FitsAnalyzerService::computeTask(Task& task) {

    // if task method is configureFitAnalyzerService then call the right method
    if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE)) {
        configureFitAnalyzerService(task);
    }

    // if task method is configureFitService then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            METHOD_NAME_CONFIGURE_FIT_SERVICE)) {
        configureFitService(task);
    }

    // if task method is evaluateCFF then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            METHOD_NAME_EVALUATE_CFF)) {
        evaluateCFF(task);
    }

    // if task method is evaluateObservable then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            METHOD_NAME_EVALUATE_OBSERVABLE)) {
        evaluateObservable(task);
    }

    // if task method is evaluateChi2 then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            METHOD_NAME_EVALUATE_CHI2)) {
        evaluateChi2(task);
    }

    // if task method is evaluateChi21D then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            METHOD_NAME_EVALUATE_CHI2_SHAPE_1D)) {
        evaluateChi2Shape1D(task);
    }

    // if task method is evaluateChi22D then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            METHOD_NAME_EVALUATE_CHI2_SHAPE_2D)) {
        evaluateChi2Shape2D(task);
    }
}

void FitsAnalyzerService::configureFitAnalyzerService(const Task &task) {

    // search for "files" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILES)) {

        // search for "outputFile" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILE_OUTPUT)) {

            //get path
            m_outputFilePath =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();

            //info
            info(__func__,
                    ElemUtils::Formatter() << "Fit analyzer output file is "
                            << m_outputFilePath);

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <param name=\""
                            << METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILE_OUTPUT
                            << "\"> element ; Or check case");
        }

    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform operation ; Missing <kineamtics type=\""
                        << METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILES
                        << "\"> element ; Or check case");
    }
}

void FitsAnalyzerService::configureFitService(const Task &task) {

    //paths
    std::string inputFitScenarioFilePath;
    std::string inputFitResultFilePath;

    // search for "files" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES)) {

        // search for "inputFile" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_INPUT)) {

            //get path
            inputFitScenarioFilePath =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();

            //info
            info(__func__,
                    ElemUtils::Formatter() << "Fit input scenario file is "
                            << inputFitScenarioFilePath);

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <param name=\""
                            << METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_INPUT
                            << "\"> element ; Or check case");
        }

        // search for "outputFile" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_OUTPUT)) {

            //get path
            inputFitResultFilePath =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();

            //info
            info(__func__,
                    ElemUtils::Formatter() << "Fit result file is "
                            << inputFitResultFilePath);

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <param name=\""
                            << METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_OUTPUT
                            << "\"> element ; Or check case");
        }

    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform operation ; Missing <kineamtics type=\""
                        << METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES
                        << "\"> element ; Or check case");
    }

    //read input fit scenario xml file
    AutomationService* pAutomationService =
            Partons::getInstance()->getServiceObjectRegistry()->getAutomationService();

    Scenario* inputFitScenario = pAutomationService->parseXMLFile(
            inputFitScenarioFilePath);

    //get fit service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //loop over tasks
    for (size_t i = 0; i != inputFitScenario->size(); i++) {

        //check if allowed
        if (inputFitScenario->getTask(i).getFunctionName()
                == FitsService::METHOD_NAME_SELECT_MINIMIZER_MODEL)
            continue;
        if (inputFitScenario->getTask(i).getFunctionName()
                == FitsService::METHOD_NAME_COMPUTE)
            continue;
        if (inputFitScenario->getTask(i).getFunctionName()
                == FitsService::METHOD_NAME_WRITE_OUTPUT)
            continue;

        pFitsService->computeTask(inputFitScenario->getTask(i));
    }

    //read input fit result xml file (XML input disabled - Qt not available)
    //m_fitResult.readFromXmlFile(inputFitResultFilePath);

    //init fit module
    pFitsService->getFitsModelModule()->initModule();
    pFitsService->getFitsModelModule()->isModuleWellConfigured();

    //set parameters
    double* fitParameters = new double[m_fitResult.getVariableNames().size()];

    std::vector<std::string>::const_iterator it;

    for (it = m_fitResult.getVariableNames().begin();
            it != m_fitResult.getVariableNames().end(); it++) {

        if (m_fitResult.getVariableFixedValues().find(*it)->second) {
            fitParameters[it - m_fitResult.getVariableNames().begin()] =
                    m_fitResult.getVariableStartValues().find(*it)->second;
        } else {
            fitParameters[it - m_fitResult.getVariableNames().begin()] =
                    m_fitResult.getVariableFitValues().find(*it)->second;
        }
    }

    pFitsService->getFitsModelModule()->updateParameters(fitParameters);

    delete[] fitParameters;
    fitParameters = 0;
}

void FitsAnalyzerService::evaluateCFF(const Task &task) {

    // search for "evaluateDVCSCFFAsFunctionOfXi" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_CFF_DVCS_AFO_XI)) {
        evaluateCffDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::xi);
    }

    // search for "evaluateDVCSCFFAsFunctionOfT" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_CFF_DVCS_AFO_T)) {
        evaluateCffDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::t);
    }

    // search for "evaluateDVCSCFFAsFunctionOfQ2" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_CFF_DVCS_AFO_Q2)) {
        evaluateCffDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::Q2);
    }

    // search for "evaluateDVCSCFFAsFunctionOfMuF2" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_CFF_DVCS_AFO_MUF2)) {
        evaluateCffDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::muF2);
    }

    // search for "evaluateDVCSCFFAsFunctionOfMuR2" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_CFF_DVCS_AFO_MUR2)) {
        evaluateCffDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::muR2);
    }
}

void FitsAnalyzerService::evaluateObservable(const Task &task) {

    // search for "evaluateDVCSObservableAsFunctionOfXb" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_XB)) {
        evaluateObservableDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::xB);
    }

    // search for "evaluateDVCSObservableAsFunctionOfT" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_T)) {
        evaluateObservableDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::t);
    }

    // search for "evaluateDVCSObservableAsFunctionOfQ2" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_Q2)) {
        evaluateObservableDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::Q2);
    }

    // search for "evaluateDVCSObservableAsFunctionOfMuE" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_E)) {
        evaluateObservableDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::E);
    }

    // search for "evaluateDVCSObservableAsFunctionOfMuPhi" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_PHI)) {
        evaluateObservableDVCS(task.getKinematicsData().getParameters(),
                KinematicVariableType::phi);
    }
}

void FitsAnalyzerService::evaluateChi2(const Task &task) {
    evaluateChiValue();
}

void FitsAnalyzerService::evaluateChi2Shape1D(const Task &task) {

    // search for "parameter" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR)) {

        //parametr name
        std::string parName;

        // search for "name" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR_NAME)) {
            parName =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <kineamtics type=\""
                            << METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR_NAME
                            << "\"> element ; Or check case");
        }

        //evaluate
        evaluateChi2Shape1DPar(task.getKinematicsData().getParameters(),
                parName);

    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform operation ; Missing <kineamtics type=\""
                        << METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR
                        << "\"> element ; Or check case");
    }
}

void FitsAnalyzerService::evaluateChi2Shape2D(const Task &task) {

    // search for "parameter" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR)) {

        //parametr name
        std::string parName1, parName2;

        // search for "name1" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_1)) {
            parName1 =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <kineamtics type=\""
                            << METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_1
                            << "\"> element ; Or check case");
        }

        // search for "name2" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_2)) {
            parName2 =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <kineamtics type=\""
                            << METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_2
                            << "\"> element ; Or check case");
        }

        //evaluate
        evaluateChi2Shape2DPar(task.getKinematicsData().getParameters(),
                parName1, parName2);

    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform operation ; Missing <kineamtics type=\""
                        << METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR
                        << "\"> element ; Or check case");
    }
}

void FitsAnalyzerService::getDataForEvaluation(
        const ElemUtils::Parameters& parameters,
        std::map<KinematicVariableType::Type, double>& kinematicsFixed,
        size_t nPar, std::vector<size_t>& nSteps, std::vector<bool>& isLog,
        std::vector<double>& min, std::vector<double>& max) const {

    //clear
    kinematicsFixed.clear();
    nSteps.clear();
    isLog.clear();
    min.clear();
    max.clear();

    //kinematics
    for (size_t i = static_cast<size_t>(KinematicVariableType::UNDEFINED) + 1;
            i < static_cast<size_t>(KinematicVariableType::LAST); i++) {
        if (parameters.isAvailable(
                KinematicVariableType(
                        static_cast<KinematicVariableType::Type>(i)).toString())) {
            kinematicsFixed.insert(
                    std::make_pair(static_cast<KinematicVariableType::Type>(i),
                            parameters.getLastAvailable().toDouble()));
        }
    }

    //tag
    std::string tag;

    //loop over expected parameters
    for (size_t i = 0; i < nPar; i++) {

        // search for "nSteps" in task
        tag = METHOD_NAME_EVALUATE_NSTEPS;
        if (nPar != 1)
            tag += std::string(ElemUtils::Formatter() << i + 1);

        if (parameters.isAvailable(tag)) {
            nSteps.push_back(parameters.getLastAvailable().toSize_t());
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <param name=\""
                            << tag << "\"> element ; Or check case");
        }

        // search for "isLog" in task
        tag = METHOD_NAME_EVALUATE_ISLOG;
        if (nPar != 1)
            tag += std::string(ElemUtils::Formatter() << i + 1);

        if (parameters.isAvailable(tag)) {
            isLog.push_back(parameters.getLastAvailable().toBoolean());
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <param name=\""
                            << tag << "\"> element ; Or check case");
        }

        // search for "rangeMin" in task
        tag = METHOD_NAME_EVALUATE_RANGE_MIN;
        if (nPar != 1)
            tag += std::string(ElemUtils::Formatter() << i + 1);

        if (parameters.isAvailable(tag)) {
            min.push_back(parameters.getLastAvailable().toDouble());
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <param name=\""
                            << tag << "\"> element ; Or check case");
        }

        // search for "rangeMax" in task
        tag = METHOD_NAME_EVALUATE_RANGE_MAX;
        if (nPar != 1)
            tag += std::string(ElemUtils::Formatter() << i + 1);

        if (parameters.isAvailable(tag)) {
            max.push_back(parameters.getLastAvailable().toDouble());
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform operation ; Missing <param name=\""
                            << tag << "\"> element ; Or check case");
        }
    }
}

void FitsAnalyzerService::evaluateCffDVCS(
        const ElemUtils::Parameters& parameters,
        KinematicVariableType::Type dependence) const {

    //get ranges, steps, ifLog and output file
    std::map<KinematicVariableType::Type, double> kinematicsFixed;
    std::vector<size_t> nSteps;
    std::vector<bool> isLog;
    std::vector<double> min;
    std::vector<double> max;

    getDataForEvaluation(parameters, kinematicsFixed, 1, nSteps, isLog, min,
            max);

    //retrieve fits service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //retrieve CFF service
    DVCSConvolCoeffFunctionService* pDVCSConvolCoeffFunctionService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSConvolCoeffFunctionService();

    //retrieve CFF module
    DVCSConvolCoeffFunctionModule* pDVCSConvolCoeffFunctionModule =
            static_cast<DVCSProcessModule*>(pFitsService->getDVCSProcessModule())->getConvolCoeffFunctionModule();

    //open file
    std::ofstream file;
    file.open(m_outputFilePath.c_str());
    file << std::scientific;

    std::map<KinematicVariableType::Type, double>::iterator it;
    for (it = kinematicsFixed.begin(); it != kinematicsFixed.end(); it++) {
        warn(__func__,
                ElemUtils::Formatter()
                        << KinematicVariableType(it->first).toString() << "\t"
                        << it->second);
    }

    //kinematics
    DVCSConvolCoeffFunctionKinematic kinematics;
    std::map<KinematicVariableType::Type, double>::const_iterator itKinematicsFixed;

    if (dependence != KinematicVariableType::xi) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::xi);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::xi).toString());
        } else {
            kinematics.setXi(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::t) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::t);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::t).toString());
        } else {
            kinematics.setT(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::Q2) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::Q2);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::Q2).toString());
        } else {
            kinematics.setQ2(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::muF2) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::muF2);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(
                                    KinematicVariableType::muF2).toString());
        } else {
            kinematics.setMuF2(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::muR2) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::muR2);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(
                                    KinematicVariableType::muR2).toString());
        } else {
            kinematics.setMuR2(itKinematicsFixed->second);
        }
    }

    //loop
    for (size_t i = 0; i <= nSteps.at(0); i++) {

        //value
        double value;

        if (isLog.at(0)) {
            value = pow(10.,
                    log10(min.at(0))
                            + i * (log10(max.at(0)) - log10(min.at(0)))
                                    / double(nSteps.at(0)));
        } else {
            value = min.at(0)
                    + i * (max.at(0) - min.at(0)) / double(nSteps.at(0));
        }

        switch (dependence) {

        case KinematicVariableType::xi: {
            kinematics.setXi(value);
            break;
        }

        case KinematicVariableType::t: {
            kinematics.setT(value);
            break;
        }

        case KinematicVariableType::Q2: {
            kinematics.setQ2(value);
            break;
        }

        case KinematicVariableType::muF2: {
            kinematics.setMuF2(value);
            break;
        }

        case KinematicVariableType::muR2: {
            kinematics.setMuR2(value);
            break;
        }

        default: {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Wrong variable: "
                            << KinematicVariableType(dependence).toString());
            break;
        }

        }

        //compute
        DVCSConvolCoeffFunctionResult cffResult =
                pDVCSConvolCoeffFunctionService->computeSingleKinematic(
                        kinematics, pDVCSConvolCoeffFunctionModule);

        file << value << "\t";

        std::map<GPDType::Type, std::complex<double> >::const_iterator itCffResult;

        for (itCffResult = cffResult.getResultsByGpdType().begin();
                itCffResult != cffResult.getResultsByGpdType().end();
                itCffResult++) {
            file << itCffResult->second.real() << "\t";
            file << itCffResult->second.imag() << "\t";
        }

        file << std::endl;
    }

    //close file
    file.close();
}

void FitsAnalyzerService::evaluateObservableDVCS(
        const ElemUtils::Parameters& parameters,
        KinematicVariableType::Type dependence) const {

    //observable name
    std::string observableName;

    // search for "name" in task
    if (parameters.isAvailable(METHOD_NAME_EVALUATE_OBSERVABLE_NAME)) {
        observableName = parameters.getLastAvailable().getString();
    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform operation ; Missing <kineamtics type=\""
                        << METHOD_NAME_EVALUATE_OBSERVABLE_NAME
                        << "\"> element ; Or check case");
    }

    //get ranges, steps, ifLog and output file
    std::map<KinematicVariableType::Type, double> kinematicsFixed;
    std::vector<size_t> nSteps;
    std::vector<bool> isLog;
    std::vector<double> min;
    std::vector<double> max;

    getDataForEvaluation(parameters, kinematicsFixed, 1, nSteps, isLog, min,
            max);

    //retrieve fits service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //get observable service
    DVCSObservableService* pObservableService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();

    //get observable
    DVCSObservable* pObservable =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    observableName);

    pObservable->setProcessModule(
            static_cast<DVCSProcessModule*>(pFitsService->getDVCSProcessModule()));

    //open file
    std::ofstream file;
    file.open(m_outputFilePath.c_str());
    file << std::scientific;

    //kinematics
    DVCSObservableKinematic kinematics;
    std::map<KinematicVariableType::Type, double>::const_iterator itKinematicsFixed;

    if (dependence != KinematicVariableType::xB) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::xB);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::xB).toString());
        } else {
            kinematics.setXB(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::t) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::t);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::t).toString());
        } else {
            kinematics.setT(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::Q2) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::Q2);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::Q2).toString());
        } else {
            kinematics.setQ2(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::E) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::E);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::E).toString());
        } else {
            kinematics.setE(itKinematicsFixed->second);
        }
    }

    if (dependence != KinematicVariableType::phi) {

        itKinematicsFixed = kinematicsFixed.find(KinematicVariableType::phi);

        if (itKinematicsFixed == kinematicsFixed.end()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Unable to determine value of: "
                            << KinematicVariableType(KinematicVariableType::phi).toString());
        } else {
            kinematics.setPhi(itKinematicsFixed->second);
        }
    }

    //loop
    for (size_t i = 0; i <= nSteps.at(0); i++) {

        //value
        double value;

        if (isLog.at(0)) {
            value = pow(10.,
                    log10(min.at(0))
                            + i * (log10(max.at(0)) - log10(min.at(0)))
                                    / double(nSteps.at(0)));
        } else {
            value = min.at(0)
                    + i * (max.at(0) - min.at(0)) / double(nSteps.at(0));
        }

        switch (dependence) {

        case KinematicVariableType::xB: {
            kinematics.setXB(value);
            break;
        }

        case KinematicVariableType::t: {
            kinematics.setT(value);
            break;
        }

        case KinematicVariableType::Q2: {
            kinematics.setQ2(value);
            break;
        }

        case KinematicVariableType::E: {
            kinematics.setE(value);
            break;
        }

        case KinematicVariableType::phi: {
            kinematics.setPhi(value);
            break;
        }

        default: {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "Wrong variable: "
                            << KinematicVariableType(dependence).toString());
            break;
        }

        }

        //compute
        DVCSObservableResult observableResult =
                pObservableService->computeSingleKinematic(kinematics, pObservable);

        file << value << "\t" << observableResult.getValue().getValue() << std::endl;
    }

    //close file
    file.close();
}

void FitsAnalyzerService::evaluateChiValue() const {

    //open file
    std::ofstream file;
    file.open(m_outputFilePath.c_str());
    file << std::scientific;

    //retrieve fits service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //set parameters
    double* fitParameters = new double[m_fitResult.getVariableNames().size()];

    std::vector<std::string>::const_iterator it;

    for (it = m_fitResult.getVariableNames().begin();
            it != m_fitResult.getVariableNames().end(); it++) {

        if (m_fitResult.getVariableFixedValues().find(*it)->second) {
            fitParameters[it - m_fitResult.getVariableNames().begin()] =
                    m_fitResult.getVariableStartValues().find(*it)->second;
        } else {
            fitParameters[it - m_fitResult.getVariableNames().begin()] =
                    m_fitResult.getVariableFitValues().find(*it)->second;
        }
    }

    //evaluate
    file << pFitsService->computeChi2(fitParameters) << std::endl;

    //close file
    file.close();

    //delete
    delete[] fitParameters;
    fitParameters = 0;
}

void FitsAnalyzerService::evaluateChi2Shape1DPar(
        const ElemUtils::Parameters& parameters,
        const std::string& parName) const {

    //get ranges, steps, ifLog and output file
    std::map<KinematicVariableType::Type, double> kinematicsFixed;
    std::vector<size_t> nSteps;
    std::vector<bool> isLog;
    std::vector<double> min;
    std::vector<double> max;

    getDataForEvaluation(parameters, kinematicsFixed, 1, nSteps, isLog, min,
            max);

    //retrieve fits service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //open file
    std::ofstream file;
    file.open(m_outputFilePath.c_str());
    file << std::scientific;

    //set parameters and find id on that being varied
    size_t parId;
    bool parIdIsFound = false;
    double* fitParameters = new double[m_fitResult.getVariableNames().size()];

    std::vector<std::string>::const_iterator it;

    for (it = m_fitResult.getVariableNames().begin();
            it != m_fitResult.getVariableNames().end(); it++) {

        if (*it == parName) {
            parId = it - m_fitResult.getVariableNames().begin();
            parIdIsFound = true;
        } else {
            if (m_fitResult.getVariableFixedValues().find(*it)->second) {
                fitParameters[it - m_fitResult.getVariableNames().begin()] =
                        m_fitResult.getVariableStartValues().find(*it)->second;
            } else {
                fitParameters[it - m_fitResult.getVariableNames().begin()] =
                        m_fitResult.getVariableFitValues().find(*it)->second;
            }
        }
    }

    if (!parIdIsFound) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Unable to find id for parameter name: " << parName);
    }

    //loop
    for (size_t i = 0; i <= nSteps.at(0); i++) {

        //value
        if (isLog.at(0)) {
            fitParameters[parId] = pow(10.,
                    log10(min.at(0))
                            + i * (log10(max.at(0)) - log10(min.at(0)))
                                    / double(nSteps.at(0)));
        } else {
            fitParameters[parId] = min.at(0)
                    + i * (max.at(0) - min.at(0)) / double(nSteps.at(0));
        }

        //evaluate and store
        file << fitParameters[parId] << "\t"
                << pFitsService->computeChi2(fitParameters) << std::endl;
    }

    //clean
    delete[] fitParameters;
    fitParameters = 0;

    //close file
    file.close();
}

void FitsAnalyzerService::evaluateChi2Shape2DPar(
        const ElemUtils::Parameters& parameters, const std::string& parName1,
        const std::string& parName2) const {

    //get ranges, steps, ifLog and output file
    std::map<KinematicVariableType::Type, double> kinematicsFixed;
    std::vector<size_t> nSteps;
    std::vector<bool> isLog;
    std::vector<double> min;
    std::vector<double> max;

    getDataForEvaluation(parameters, kinematicsFixed, 2, nSteps, isLog, min,
            max);

    //retrieve fits service
    FitsService* pFitsService =
            static_cast<FitsService*>(Partons::getInstance()->getServiceObjectRegistry()->get(
                    FitsService::classId));

    //open file
    std::ofstream file;
    file.open(m_outputFilePath.c_str());
    file << std::scientific;

    //set parameters and find id on that being varied
    size_t parId1;
    bool parIdIsFound1 = false;
    size_t parId2;
    bool parIdIsFound2 = false;
    double* fitParameters = new double[m_fitResult.getVariableNames().size()];

    std::vector<std::string>::const_iterator it;

    for (it = m_fitResult.getVariableNames().begin();
            it != m_fitResult.getVariableNames().end(); it++) {

        if (*it == parName1) {
            parId1 = it - m_fitResult.getVariableNames().begin();
            parIdIsFound1 = true;
        } else if (*it == parName2) {
            parId2 = it - m_fitResult.getVariableNames().begin();
            parIdIsFound2 = true;
        } else {
            if (m_fitResult.getVariableFixedValues().find(*it)->second) {
                fitParameters[it - m_fitResult.getVariableNames().begin()] =
                        m_fitResult.getVariableStartValues().find(*it)->second;
            } else {
                fitParameters[it - m_fitResult.getVariableNames().begin()] =
                        m_fitResult.getVariableFitValues().find(*it)->second;
            }
        }
    }

    if (!parIdIsFound1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Unable to find id for parameter (first) name: "
                        << parName1);
    }

    if (!parIdIsFound2) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Unable to find id for parameter (second) name: "
                        << parName2);
    }

    //loop
    for (size_t i = 0; i <= nSteps.at(0); i++) {

        //value 1
        if (isLog.at(0)) {
            fitParameters[parId1] = pow(10.,
                    log10(min.at(0))
                            + i * (log10(max.at(0)) - log10(min.at(0)))
                                    / double(nSteps.at(0)));
        } else {
            fitParameters[parId1] = min.at(0)
                    + i * (max.at(0) - min.at(0)) / double(nSteps.at(0));
        }

        for (size_t j = 0; j <= nSteps.at(1); j++) {

            //value 2
            if (isLog.at(1)) {
                fitParameters[parId2] = pow(10.,
                        log10(min.at(1))
                                + j * (log10(max.at(1)) - log10(min.at(1)))
                                        / double(nSteps.at(1)));
            } else {
                fitParameters[parId2] = min.at(1)
                        + j * (max.at(1) - min.at(1)) / double(nSteps.at(1));
            }

            //evaluate and store
            file << fitParameters[parId1] << "\t" << fitParameters[parId2]
                    << "\t" << pFitsService->computeChi2(fitParameters)
                    << std::endl;
        }
    }

    //clean
    delete[] fitParameters;
    fitParameters = 0;

    //close file
    file.close();
}
