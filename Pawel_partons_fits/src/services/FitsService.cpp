#include "../../include/services/FitsService.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/GenericType.h>
#include <ElementaryUtils/parameters/Parameters.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <ElementaryUtils/string_utils/StringUtils.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/automation/Task.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionResult.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/Scales.h>
#include <partons/BaseObjectRegistry.h>
//#include <partons/database/observable/service/ObservableResultDaoService.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/ModuleObject.h>
#include <partons/modules/xi_converter/XiConverterModule.h>
#include <partons/modules/scales/ScalesModule.h>
#include <partons/Partons.h>
#include <partons/services/ConvolCoeffFunctionService.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/beans/observable/DVCS/DVCSObservableResult.h>
#include <partons/ServiceObjectRegistry.h>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

#include "../../include/algorithms/random_generator/RandomGenerator.h"
#include "../../include/FitsModuleObjectFactory.h"
#include "../../include/modules/fits_model/FitsModelModule.h"
#include "../../include/modules/minimizer_model/FitsMinimizerModule.h"
#include "../../include/PartonsFits.h"
#include "../../include/utils/data/DataFitUtils.h"

using namespace PARTONS;

const std::string FitsService::METHOD_NAME_CONFIGURE_RANDOM_GENERATOR =
        "configureRandomGenerator";
const std::string FitsService::METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED =
        "seed";
const std::string FitsService::METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED_VALUE =
        "value";
const std::string FitsService::METHOD_NAME_DEFINE_KINEMATIC_CUTS =
        "defineKinematicCuts";
const std::string FitsService::METHOD_NAME_DEFINE_KINEMATIC_CUTS_CUTS =
        "kinematicCuts";
const std::string FitsService::METHOD_NAME_DEFINE_KINEMATIC_CUTS_CUTS_LIST =
        "list";
const std::string FitsService::METHOD_NAME_SELECT_OBSERVABLES =
        "selectObservables";
const std::string FitsService::METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT =
        "ObservableResult";
const std::string FitsService::METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_QUERY =
        "SQLQuery";
const std::string FitsService::METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_REPLICA =
        "MakeReplica";
const std::string FitsService::METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOM =
        "MakeRandom";
const std::string FitsService::METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOMFRACTION =
        "MakeRandomFraction";
const std::string FitsService::METHOD_NAME_CONFIGURE_DVCS_CHANNEL =
        "configureDVCSChannel";
const std::string FitsService::METHOD_NAME_SELECT_MINIMIZER_MODEL =
        "configureMinimizerModule";
const std::string FitsService::METHOD_NAME_SELECT_MINIMIZER_MODEL_MODULE =
        "MinimizerModule";
const std::string FitsService::METHOD_NAME_SELECT_FITS_MODEL =
        "configureFitsModule";
const std::string FitsService::METHOD_NAME_SELECT_FITS_MODEL_MODULE =
        "FitsModelModule";
const std::string FitsService::METHOD_NAME_COMPUTE = "compute";
const std::string FitsService::METHOD_NAME_WRITE_OUTPUT = "writeOutput";
const std::string FitsService::METHOD_NAME_WRITE_OUTPUT_FILE = "outputFile";
const std::string FitsService::METHOD_NAME_WRITE_OUTPUT_FILE_PATH = "path";

// initialize [class]::classId with unique name
const unsigned int FitsService::classId =
        Partons::getInstance()->getBaseObjectRegistry()->registerBaseObject(
                new FitsService("FitsService"));

FitsService::FitsService(const std::string &className) :
        ServiceObject(className), m_pObservableService(0), m_pMinimizer(0), m_pFitsModelModule(
                0), m_pDVCSProcessModule(0), m_pDVMPProcessModule(0), m_pTCSProcessModule(
                0) {
}

FitsService::~FitsService() {

    // delete minimizer if necessary
    if (m_pMinimizer) {
        delete m_pMinimizer;
        m_pMinimizer = 0;
    }

    //delete fits module if necessary
    if (m_pFitsModelModule) {
        delete m_pFitsModelModule;
        m_pFitsModelModule = 0;
    }
}

void FitsService::resolveObjectDependencies() {

    // set observable service
    m_pObservableService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();
}

void FitsService::computeTask(Task& task) {

    // if task method is configureRandomGenerator then call the right method
    if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_CONFIGURE_RANDOM_GENERATOR)) {
        configureRandomGenerator(task);
    }
    // if task method is selectKinematicCuts then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_DEFINE_KINEMATIC_CUTS)) {
        defineKinematicCuts(task);
    }
    // if task method is selectObservables then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_SELECT_OBSERVABLES)) {
        selectObservables(task);
    }
    // else if task method is configureDVCSChannel then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_CONFIGURE_DVCS_CHANNEL)) {
        configureDVCSChannel(task);
    }
    // else if task method is selectMinimizerModel then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_SELECT_MINIMIZER_MODEL)) {
        selectMinimizerModel(task);
    }
    // else if task method is selectFitModel then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_SELECT_FITS_MODEL)) {
        selectFitsModel(task);
    }
    // else if task method is compute then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_COMPUTE)) {
        compute();
    }
    // else if task method is writeOutput then call the right method
    else if (ElemUtils::StringUtils::equals(task.getFunctionName(),
            FitsService::METHOD_NAME_WRITE_OUTPUT)) {
        writeOutput(task);
    }
}

void FitsService::configureRandomGenerator(const Task &task) {

    // search for "ObservableResult" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED)) {

        // search for "SQLQuery" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED_VALUE)) {

            //set
            RandomGenerator::getInstance()->setSeed(
                    task.getKinematicsData().getParameters().getLastAvailable().toUInt());
        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform random number configuration ; Missing <param name=\""
                            << METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED_VALUE
                            << "\"> element ; Or check case");
        }
    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform random number configuration ; Missing <kineamtics type=\""
                        << METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED
                        << "\"> element ; Or check case");
    }
}

void FitsService::defineKinematicCuts(const Task& task) {

    //clear
    m_kinematicCutList.clear();

    // search for "KinematicCuts" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_DEFINE_KINEMATIC_CUTS_CUTS)) {

        // search for "KinematicCut" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_DEFINE_KINEMATIC_CUTS_CUTS_LIST)) {

            // get kinematic cuts
            std::string kinematicCuts =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();

            //get tokens
            std::vector<std::string> kinematicCutsTokens;

            std::size_t start = 0;
            std::size_t end = 0;

            while ((end = kinematicCuts.find(';', start)) != std::string::npos) {
                kinematicCutsTokens.push_back(
                        kinematicCuts.substr(start, end - start));
                start = end + 1;
            }
            kinematicCutsTokens.push_back(kinematicCuts.substr(start));

            //create
            std::vector<std::string>::iterator itKinematicCutsTokens;

            for (itKinematicCutsTokens = kinematicCutsTokens.begin();
                    itKinematicCutsTokens != kinematicCutsTokens.end();
                    itKinematicCutsTokens++) {

                KinematicCut kinematicCut(*itKinematicCutsTokens);
                info(__func__,
                        ElemUtils::Formatter() << "Kinematic cut added: "
                                << kinematicCut.toString());
                m_kinematicCutList.add(kinematicCut);
            }

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform observables selection ; Missing <param name=\""
                            << METHOD_NAME_DEFINE_KINEMATIC_CUTS_CUTS
                            << "\"> element ; Or check case");
        }
    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform observables selection ; Missing <kineamtics type=\""
                        << METHOD_NAME_DEFINE_KINEMATIC_CUTS
                        << "\"> element ; Or check case");
    }
}

void FitsService::selectObservables(const Task& task) {

    //clear
    m_observableListToFit.clear();

    // search for "ObservableResult" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT)) {

        //selected observables
        List<DVCSObservableResult> selectedObservableFromDatabaseList;

        // search for "SQLQuery" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_QUERY)) {

            // get sql query
            std::string sqlQuery =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();

            // Database DAO service no longer available in this version of PARTONS.
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "ObservableResultDaoService is not available. "
                    << "SQL query: " << sqlQuery);

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform observables selection ; Missing <param name=\""
                            << METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_QUERY
                            << "\"> element ; Or check case");
        }

        // search for "MakeReplica" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_REPLICA)) {

            if (task.getKinematicsData().getParameters().getLastAvailable().toInt()) {

                //make replica
                smearDataForReplicaMethod(selectedObservableFromDatabaseList);

                info(__func__,
                        ElemUtils::Formatter() << "Replica method is ON");

            } else {

                info(__func__,
                        ElemUtils::Formatter() << "Replica method is OFF");
            }

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform observables selection ; Missing <param name=\""
                            << METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_REPLICA
                            << "\"> element ; Or check case");
        }

        // search for "MakeRandom" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOM)) {

            //mode
            int makeRandom =
                    task.getKinematicsData().getParameters().getLastAvailable().toInt();

            if (makeRandom != -1 && makeRandom != 0 && makeRandom != 1) {
                throw ElemUtils::CustomException(getClassName(), __func__,
                        ElemUtils::Formatter() << "Illegal value of "
                                << METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOM
                                << " equals " << makeRandom);
            }

            //is used
            if (makeRandom != 0) {

                //fraction
                double fractionRandom;

                if (task.getKinematicsData().getParameters().isAvailable(
                        METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOMFRACTION)) {

                    fractionRandom =
                            task.getKinematicsData().getParameters().getLastAvailable().toDouble();

                    if (fractionRandom < 0. || fractionRandom > 1.) {
                        throw ElemUtils::CustomException(getClassName(),
                                __func__,
                                ElemUtils::Formatter() << "Illegal value of "
                                        << METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOMFRACTION
                                        << " equals " << makeRandom);
                    }

                } else {
                    throw ElemUtils::CustomException(getClassName(), __func__,
                            ElemUtils::Formatter()
                                    << "Cannot perform observables selection ; Missing <param name=\""
                                    << METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOMFRACTION
                                    << "\"> element ; Or check case");
                }

                //select random
                selectDataForRandomMethod(selectedObservableFromDatabaseList,
                        makeRandom, fractionRandom);

                info(__func__,
                        ElemUtils::Formatter()
                                << "Random method is ON with mode "
                                << makeRandom << " and fraction "
                                << fractionRandom);

            } else {

                info(__func__,
                        ElemUtils::Formatter() << "Random method is OFF");
            }

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform observables selection ; Missing <param name=\""
                            << METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOM
                            << "\"> element ; Or check case");
        }

        // sort ObservableResult previously retrieved from database in a better structure for fits.
        makeFitsConfigurationFromDatabaseObjects(
                selectedObservableFromDatabaseList);

        //average, min and max kinematics
        DVCSObservableKinematic averageKinematics =
                DataFitUtils::getAverageKinematic(m_observableListToFit);
        DVCSObservableKinematic minimumKinematics =
                DataFitUtils::getMinimumKinematic(m_observableListToFit);
        DVCSObservableKinematic maximumKinematics =
                DataFitUtils::getMaximumKinematic(m_observableListToFit);

        info(__func__,
                ElemUtils::Formatter()
                        << selectedObservableFromDatabaseList.size()
                        << " observables have been selected from database");

        info(__func__,
                ElemUtils::Formatter() << m_observableListToFit.size()
                        << " observables have been accepted after checking kinematic cuts");

        info(__func__,
                ElemUtils::Formatter() << "Average kinematics is: "
                        << averageKinematics.toString());

        info(__func__,
                ElemUtils::Formatter() << "Mininum kinematics is: "
                        << minimumKinematics.toString());

        info(__func__,
                ElemUtils::Formatter() << "Maximum kinematics is: "
                        << maximumKinematics.toString());

    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform observables selection ; Missing <kineamtics type=\""
                        << METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT
                        << "\"> element ; Or check case");
    }
}

void FitsService::configureDVCSChannel(const Task& task) {
    m_pDVCSProcessModule = m_pObservableService->newDVCSProcessModuleFromTask(task);
}

void FitsService::configureDVMPChannel(const Task& task) {
    throw ElemUtils::CustomException(getClassName(), __func__,
            "DVMP channel is not yet implemented");
}

void FitsService::configureTCShannel(const Task& task) {
    throw ElemUtils::CustomException(getClassName(), __func__,
            "TCS channel is not yet implemented");
}

void FitsService::selectMinimizerModel(const Task &task) {

    // search for "MinimizerModule" in task
    if (ElemUtils::StringUtils::equals(
            task.getModuleComputationConfiguration().getModuleType(),
            METHOD_NAME_SELECT_MINIMIZER_MODEL_MODULE)) {

        //set FitsModule
        m_pMinimizer =
                PartonsFits::getInstance()->getFitsModuleObjectFactory()->newFitsMinimizerModule(
                        task.getModuleComputationConfiguration().getModuleClassName());

        //configure FitsModule
        m_pMinimizer->configure(
                task.getModuleComputationConfiguration().getParameters());

    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform MinimizerModel selection ; Missing <param name=\""
                        << METHOD_NAME_SELECT_MINIMIZER_MODEL_MODULE
                        << "\"> element ; Or check case");
    }
}

void FitsService::selectFitsModel(const Task &task) {

    // search for "FitsModelModule" in task
    if (ElemUtils::StringUtils::equals(
            task.getModuleComputationConfiguration().getModuleType(),
            METHOD_NAME_SELECT_FITS_MODEL_MODULE)) {

        //set FitsModule
        m_pFitsModelModule =
                PartonsFits::getInstance()->getFitsModuleObjectFactory()->newFitsModule(
                        task.getModuleComputationConfiguration().getModuleClassName());

        //configure FitsModule
        m_pFitsModelModule->configure(
                task.getModuleComputationConfiguration().getParameters());

    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform FitsModel selection ; Missing <param name=\""
                        << METHOD_NAME_SELECT_FITS_MODEL_MODULE
                        << "\"> element ; Or check case");
    }
}

void FitsService::compute() {

    //1. check if pointers set
    if (!m_pMinimizer) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Pointer to Minimizer is null");
    }

    if (!m_pFitsModelModule) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                "Pointer to FitModel is null");
    }

    //2. set function
    //TODO I think no need to define parameter names in FitsModelModule, one one needs is only the number of them that can be set in constructor of given FitsModelModule
    m_pMinimizer->setFCN(this, &FitsService::computeChi2,
            m_pFitsModelModule->getParameters().size());

    //3. init module and check if well configured
    m_pFitsModelModule->initModule();
    m_pFitsModelModule->isModuleWellConfigured();

    //4. init parameters
    m_pFitsModelModule->initParameters();

    //5. run
    m_pMinimizer->runMinimization();

    //6. gather information
    m_fitResult.gatherInformationFromMinimizerModule(m_pMinimizer);
    m_fitResult.gatherInformationFromDataFitList(m_observableListToFit);

    //7. print result
    m_fitResult.writeToLogger();
}

void FitsService::writeOutput(const Task &task) {

    // search for "OutputFile" in task
    if (ElemUtils::StringUtils::equals(
            task.getKinematicsData().getModuleClassName(),
            METHOD_NAME_WRITE_OUTPUT_FILE)) {

        // search for "Path" in task
        if (task.getKinematicsData().getParameters().isAvailable(
                METHOD_NAME_WRITE_OUTPUT_FILE_PATH)) {

            //get path
            std::string outputFilePath =
                    task.getKinematicsData().getParameters().getLastAvailable().getString();

            //write (XML output disabled - Qt not available)
            //m_fitResult.writeToXmlFile(outputFilePath);

            //info
            info(__func__,
                    ElemUtils::Formatter() << "Output file writing disabled (no Qt)");

        } else {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Cannot perform writing ; Missing <param name=\""
                            << METHOD_NAME_WRITE_OUTPUT_FILE_PATH
                            << "\"> element ; Or check case");
        }
    } else {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter()
                        << "Cannot perform writing ; Missing <kineamtics type=\""
                        << METHOD_NAME_WRITE_OUTPUT_FILE
                        << "\"> element ; Or check case");
    }
}

ModuleObject* FitsService::getProcessModule(
        const ChannelType& channel) const {

    switch (channel.getType()) {

    case ChannelType::DVCS: {
        return getDVCSProcessModule();
        break;
    }

    case ChannelType::DVMP: {
        return getDVMPProcessModule();
        break;
    }

    case ChannelType::TCS: {
        return getTCSProcessModule();
        break;
    }

    default:
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << ChannelType(channel).toString()
                        << " process not configured");
        return 0;
    }
}
DVCSConvolCoeffFunctionModule* FitsService::getCFFModule(
        const ChannelType &channel) const {
    return static_cast<DVCSProcessModule*>(getProcessModule(channel))->getConvolCoeffFunctionModule();
}

GPDModule* FitsService::getGPDModule(const ChannelType& channel) const {
    return getCFFModule(channel)->getGPDModule();
}

ModuleObject* FitsService::getDVCSProcessModule() const {
    return m_pDVCSProcessModule;
}

ModuleObject* FitsService::getDVMPProcessModule() const {
    return m_pDVMPProcessModule;
}

ModuleObject* FitsService::getTCSProcessModule() const {
    return m_pTCSProcessModule;
}

const List<ObservableDataFit>& FitsService::getObservableListToFit() const {
    return m_observableListToFit;
}

const FitResult& FitsService::getFitResult() const {
    return m_fitResult;
}

FitsMinimizerModule* FitsService::getMinimizer() const {
    return m_pMinimizer;
}

FitsModelModule* FitsService::getFitsModelModule() const {
    return m_pFitsModelModule;
}

double FitsService::computeChi2(const double* params) {

    // 1. update parameters
    m_pFitsModelModule->updateParameters(params);

    // 2. compute observable
    return computeObservablesAndReturnChi2();
}

double FitsService::computeObservablesAndReturnChi2() {

    //reset chi2 value
    double chi2 = 0.;

    //chi2 element
    double chi2Element;

    //error
    double err;

    //compute CFFs
    PARTONS::List<PARTONS::DVCSConvolCoeffFunctionResult> cffDVCSResultsToFit =
            Partons::getInstance()->getServiceObjectRegistry()->getConvolCoeffFunctionService()->computeForOneCCFModelAndManyKinematics(
                    m_cffDVCSKinematicsToFit,
                    m_pDVCSProcessModule->getConvolCoeffFunctionModule());

    //loop over points
    for (unsigned int i = 0; i != m_observableListToFit.size(); i++) {

        //index
        size_t id = m_observableListToFit[i].getCffKinematicsIndex();

        //check index
        if (id >= cffDVCSResultsToFit.size()) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "No CFF result for this observable");
        }

        //experiment cff kinematics
        const DVCSConvolCoeffFunctionKinematic& experimentCFFKinematics =
                m_observableListToFit[i].getCffKinematic();

        //fit cff result
        const DVCSConvolCoeffFunctionResult& fitCFFResult =
                cffDVCSResultsToFit[id];

        //fit cff kinematics
        const DVCSConvolCoeffFunctionKinematic& fitCFFKinematics =
                fitCFFResult.getKinematic();

        //check of both the same
        if ((experimentCFFKinematics.getXi() != fitCFFKinematics.getXi())
                || (experimentCFFKinematics.getT() != fitCFFKinematics.getT())
                || (experimentCFFKinematics.getQ2() != fitCFFKinematics.getQ2())
                || (experimentCFFKinematics.getMuF2()
                        != fitCFFKinematics.getMuF2())
                || (experimentCFFKinematics.getMuR2()
                        != fitCFFKinematics.getMuR2())) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter()
                            << "Not able to find calculated CFF for given kinematics "
                            << cffDVCSResultsToFit.toString() << " for: "
                            << experimentCFFKinematics.toString() << " index: "
                            << id);
        }

        //caluclate
        m_observableListToFit[i].compute(fitCFFResult);

        //check if upper or lower errors
        if (m_observableListToFit[i].getReferenceObservableResult().getValue()
                - m_observableListToFit[i].getFittedObservableResult().getValue()
                > 0.) {
            err =
                    sqrt(
                            pow(
                                    m_observableListToFit[i].getReferenceObservableResult().getStatError().getLowerBound(),
                                    2)
                                    + pow(
                                            m_observableListToFit[i].getReferenceObservableResult().getSystError().getLowerBound(),
                                            2));
        } else {
            err =
                    sqrt(
                            pow(
                                    m_observableListToFit[i].getReferenceObservableResult().getStatError().getUpperBound(),
                                    2)
                                    + pow(
                                            m_observableListToFit[i].getReferenceObservableResult().getSystError().getUpperBound(),
                                            2));
        }

        //chi2
        chi2Element =
                calculateChi2Contribution(
                        m_observableListToFit[i].getReferenceObservableResult().getValue(),
                        m_observableListToFit[i].getFittedObservableResult().getValue(),
                        err);

        //check
        if (std::isnan(chi2Element)) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "NaN for: "
                            << m_observableListToFit[i].toString());
        }

        //check
        if (std::isinf(chi2Element)) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    ElemUtils::Formatter() << "INF for: "
                            << m_observableListToFit[i].toString());
        }

        //add
        chi2 += chi2Element;
    }

    //info
    info(__func__,
            ElemUtils::Formatter() << "Actual: chi2: " << chi2 << " nData: "
                    << m_observableListToFit.size() << " chi2/nData: "
                    << chi2 / m_observableListToFit.size());

    //return
    return chi2;
}

double FitsService::calculateChi2Contribution(
        const double observableValueReference,
        const double observableValueComputed,
        const double observableErrorReference) const {

    //calculate and return
    return pow(
            (observableValueReference - observableValueComputed)
                    / observableErrorReference, 2);
}

void FitsService::makeFitsConfigurationFromDatabaseObjects(
        const List<DVCSObservableResult>& selectedObservableFromDatabaseList) {

    // for each data base entry add to FitsConfigurationData
    for (size_t i = 0; i != selectedObservableFromDatabaseList.size(); i++) {

        //check kinematic cuts
        bool isFine = true;

        for (size_t j = 0; j < m_kinematicCutList.size(); j++) {

            if (!m_kinematicCutList[j].evaluate(
                    selectedObservableFromDatabaseList[i].getKinematic())) {

                info(__func__,
                        ElemUtils::Formatter() << "Point with kinematics "
                                << selectedObservableFromDatabaseList[i].getKinematic().toString()
                                << " removed to due to cut "
                                << m_kinematicCutList[j].toString());

                isFine = false;
                break;
            }
        }

        if (!isFine)
            continue;

        //create ObservableDataFit
        ObservableDataFit observableDataFit(
                selectedObservableFromDatabaseList[i]);

        //resolve process module pointer for each observable module to fit before running minimization
        observableDataFit.resolveObjectDependencies();

        //define cff kinematics
        double xi = m_pDVCSProcessModule->getXiConverterModule()->compute(
                selectedObservableFromDatabaseList[i].getKinematic().getXB(),
                selectedObservableFromDatabaseList[i].getKinematic().getT(),
                selectedObservableFromDatabaseList[i].getKinematic().getQ2());
        double t = selectedObservableFromDatabaseList[i].getKinematic().getT();
        double Q2 =
                selectedObservableFromDatabaseList[i].getKinematic().getQ2();
        Scales scales = m_pDVCSProcessModule->getScalesModule()->compute(
                selectedObservableFromDatabaseList[i].getKinematic().getQ2());

        DVCSConvolCoeffFunctionKinematic cffKinematics(xi, t, Q2,
                scales.getMuF2(), scales.getMuR2());

        //check if already exist
        bool found = false;

        for (size_t j = 0; j < m_cffDVCSKinematicsToFit.size(); j++) {

            if ((cffKinematics.getXi() == m_cffDVCSKinematicsToFit[j].getXi())
                    && (cffKinematics.getT()
                            == m_cffDVCSKinematicsToFit[j].getT())
                    && (cffKinematics.getQ2()
                            == m_cffDVCSKinematicsToFit[j].getQ2())
                    && (cffKinematics.getMuF2()
                            == m_cffDVCSKinematicsToFit[j].getMuF2())
                    && (cffKinematics.getMuR2()
                            == m_cffDVCSKinematicsToFit[j].getMuR2())) {

                found = true;
                break;
            }
        }

        if (!found) {

            //set indexId (needed for sorting)
            cffKinematics.setIndexId(m_cffDVCSKinematicsToFit.size());

            //add kinematics
            m_cffDVCSKinematicsToFit.add(cffKinematics);
        }

        //cff kinematics
        observableDataFit.setCffKinematic(cffKinematics);

        //add to the list
        m_observableListToFit.add(observableDataFit);
    }

    //sort cff kinematics
    m_cffDVCSKinematicsToFit.sort();

    //assign id
    for (size_t i = 0; i < m_observableListToFit.size(); i++) {

        const DVCSConvolCoeffFunctionKinematic& cffKinematics =
                m_observableListToFit[i].getCffKinematic();
        bool found = false;

        for (size_t j = 0; j < m_cffDVCSKinematicsToFit.size(); j++) {
            if ((cffKinematics.getXi() == m_cffDVCSKinematicsToFit[j].getXi())
                    && (cffKinematics.getT()
                            == m_cffDVCSKinematicsToFit[j].getT())
                    && (cffKinematics.getQ2()
                            == m_cffDVCSKinematicsToFit[j].getQ2())
                    && (cffKinematics.getMuF2()
                            == m_cffDVCSKinematicsToFit[j].getMuF2())
                    && (cffKinematics.getMuR2()
                            == m_cffDVCSKinematicsToFit[j].getMuR2())) {

                m_observableListToFit[i].setCffKinematicsIndex(j);
                found = true;
                break;
            }
        }

        if (!found) {
            throw ElemUtils::CustomException(getClassName(), __func__,
                    "Inconsistency problem");
        }
    }

}

void FitsService::smearDataForReplicaMethod(
        List<DVCSObservableResult>& selectedObservableFromDatabaseList) const {

//loop over data
    for (size_t i = 0; i != selectedObservableFromDatabaseList.size(); i++) {

        //errors
        double errl =
                sqrt(
                        pow(
                                selectedObservableFromDatabaseList[i].getStatError().getLowerBound(),
                                2)
                                + pow(
                                        selectedObservableFromDatabaseList[i].getSystError().getLowerBound(),
                                        2));

        double erru =
                sqrt(
                        pow(
                                selectedObservableFromDatabaseList[i].getStatError().getUpperBound(),
                                2)
                                + pow(
                                        selectedObservableFromDatabaseList[i].getSystError().getUpperBound(),
                                        2));

        //u or l unc.
        if (RandomGenerator::getInstance()->diceFlat() < errl / (errl + erru)) {
            selectedObservableFromDatabaseList[i].setValue(
                    selectedObservableFromDatabaseList[i].getValue()
                            - errl
                                    * fabs(
                                            RandomGenerator::getInstance()->diceNormal()));
        } else {
            selectedObservableFromDatabaseList[i].setValue(
                    selectedObservableFromDatabaseList[i].getValue()
                            + erru
                                    * fabs(
                                            RandomGenerator::getInstance()->diceNormal()));
        }
    }
}

void FitsService::selectDataForRandomMethod(
        PARTONS::List<PARTONS::DVCSObservableResult>& selectedObservableFromDatabaseList,
        int mode, double fraction) const {

	//check mode value
    if (mode != -1 && mode != 1) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Illegal value of mode equals "
                        << mode);
    }

	//check fraction value
    if (fraction < 0. || fraction > 1.) {
        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Illegal value of fraction equals "
                        << fraction);
    }

	//result
    PARTONS::List<PARTONS::DVCSObservableResult> result;

	//loop over data
    for (size_t i = 0; i != selectedObservableFromDatabaseList.size(); i++) {

        //dice
        bool accepted = (RandomGenerator::getInstance()->diceFlat() < fraction);

        if (mode == 1) {
            if (accepted)
                result.add(selectedObservableFromDatabaseList[i]);
        }

        if (mode == -1) {
            if (!accepted)
                result.add(selectedObservableFromDatabaseList[i]);
        }
    }

	//return
    selectedObservableFromDatabaseList = result;
}
