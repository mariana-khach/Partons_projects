#ifndef FITS_SERVICE_H
#define FITS_SERVICE_H

/**
 * @file FitsService.h
 * @author: Bryan BERTHOU (SPhN / CEA Saclay)
 * @date April 11, 2016
 * @version 1.0
 */

#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/List.h>
#include <partons/beans/channel/ChannelType.h>
#include <partons/beans/observable/DVCS/DVCSObservableResult.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/ServiceObject.h>
#include <string>

#include "../beans/fit_result/FitResult.h"
#include "../beans/kinematic_cut/KinematicCut.h"
#include "../beans/observable_data_fit/ObservableDataFit.h"

class FitsModelModule;
namespace PARTONS {
class DVCSConvolCoeffFunctionModule;
class GPDModule;
class DVCSProcessModule;
class ModuleObject;
} /* namespace PARTONS */

/**
 * Service to perform fits
 */
class FitsService: public PARTONS::ServiceObject {

public:

    static const std::string METHOD_NAME_CONFIGURE_RANDOM_GENERATOR;
    static const std::string METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED;
    static const std::string METHOD_NAME_CONFIGURE_RANDOM_GENERATOR_SEED_VALUE;
    static const std::string METHOD_NAME_DEFINE_KINEMATIC_CUTS;
    static const std::string METHOD_NAME_DEFINE_KINEMATIC_CUTS_CUTS;
    static const std::string METHOD_NAME_DEFINE_KINEMATIC_CUTS_CUTS_LIST;
    static const std::string METHOD_NAME_SELECT_OBSERVABLES;
    static const std::string METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT;
    static const std::string METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_QUERY;
    static const std::string METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_REPLICA;
    static const std::string METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOM;
    static const std::string METHOD_NAME_SELECT_OBSERVABLES_OBSERVABLE_RESULT_RANDOMFRACTION;
    static const std::string METHOD_NAME_CONFIGURE_DVCS_CHANNEL;
    static const std::string METHOD_NAME_SELECT_MINIMIZER_MODEL;
    static const std::string METHOD_NAME_SELECT_MINIMIZER_MODEL_MODULE;
    static const std::string METHOD_NAME_SELECT_FITS_MODEL;
    static const std::string METHOD_NAME_SELECT_FITS_MODEL_MODULE;
    static const std::string METHOD_NAME_COMPUTE;
    static const std::string METHOD_NAME_WRITE_OUTPUT;
    static const std::string METHOD_NAME_WRITE_OUTPUT_FILE;
    static const std::string METHOD_NAME_WRITE_OUTPUT_FILE_PATH;

    /** Unique ID to automatically register the class in the registry
     */
    static const unsigned int classId;

    /** Constructor
     @param className Class name
     */
    FitsService(const std::string &className);

    /** Destructor
     */
    virtual ~FitsService();

    /** Resolve dependencies related to this object
     */
    virtual void resolveObjectDependencies();

    /** Perform give task
     @param task Task to be performed
     */
    virtual void computeTask(PARTONS::Task &task);

    // #################################################################
    // ############################# tasks #############################
    // #################################################################

    /** Configure random generator for given task
     @param task Task to be performed
     */
    void configureRandomGenerator(const PARTONS::Task &task);

    /** Define kinematic cuts for given task
     @param task Task to be performed
     */
    void defineKinematicCuts(const PARTONS::Task& task);

    /** Select observables for given task
     @param task Task to be performed
     */
    void selectObservables(const PARTONS::Task &task);

    /** Configure DVCS channel for given task
     @param task Task to be performed
     */
    void configureDVCSChannel(const PARTONS::Task &task);

    /** Configure DVMP channel for given task
     @param task Task to be performed
     */
    void configureDVMPChannel(const PARTONS::Task &task);

    /** Configure TCS channel for given task
     @param task Task to be performed
     */
    void configureTCShannel(const PARTONS::Task &task);

    /** Selects MinimizerModel for given task
     @param task Task to be performed
     */
    void selectMinimizerModel(const PARTONS::Task &task);

    /** Selects FitsModel for given task
     @param task Task to be performed
     */
    void selectFitsModel(const PARTONS::Task &task);

    /** Run fit
     */
    void compute();

    /** Write output
     */
    void writeOutput(const PARTONS::Task &task);

    // #################################################################
    // ############################ getters ############################
    // #################################################################

    /** Get process module set to given channel
     @param channel Channel
     */
    PARTONS::ModuleObject* getProcessModule(
            const PARTONS::ChannelType &channel =
                    PARTONS::ChannelType::DVCS) const;

    /** Get CFF module set for given channel
     @param channel Channel
     */
    PARTONS::DVCSConvolCoeffFunctionModule* getCFFModule(
            const PARTONS::ChannelType &channel =
                    PARTONS::ChannelType::DVCS) const;

    /** Get GPD module set for given channel
     @param channel Channel
     */
    PARTONS::GPDModule* getGPDModule(const PARTONS::ChannelType &channel =
            PARTONS::ChannelType::DVCS) const;

    /** Get pointer to DVCS process module set by computation_configuration from XML file
     */
    PARTONS::ModuleObject* getDVCSProcessModule() const;

    /** Get pointer to DVMP process module set by computation_configuration from XML file
     */
    PARTONS::ModuleObject* getDVMPProcessModule() const;

    /** Get pointer to TCS process module set by computation_configuration from XML file
     */
    PARTONS::ModuleObject* getTCSProcessModule() const;

    /**
     * Get list of data to be fitted
     */
    const PARTONS::List<ObservableDataFit>& getObservableListToFit() const;

    /** Get object to store fit result
     */
    const FitResult& getFitResult() const;

    /** Get pointer to minimizer module
     */
    FitsMinimizerModule* getMinimizer() const;

    /** Get pointer to model module
     */
    FitsModelModule* getFitsModelModule() const;

    // #################################################################
    // ############################# other #############################
    // #################################################################

    /** Run minimization for given set of parameters
     @param params Parameters
     */
    double computeChi2(const double* params);

private:

    /** List of data to be fitted
     */
    PARTONS::List<ObservableDataFit> m_observableListToFit;

    /** List of CFF kinematics to be fitted
     */
    PARTONS::List<PARTONS::DVCSConvolCoeffFunctionKinematic> m_cffDVCSKinematicsToFit;

    /** List of kinematics cuts applied to data
     */
    PARTONS::List<KinematicCut> m_kinematicCutList;

    /** Object to store fit result
     */
    FitResult m_fitResult;

    /** Pointer to ObservableService
     */
    PARTONS::DVCSObservableService* m_pObservableService;

    /** Pointer to minimizer module
     */
    FitsMinimizerModule* m_pMinimizer;

    /** Pointer to model module to be fitted
     */
    FitsModelModule* m_pFitsModelModule;

    /** Pointer to DVCS process module set by computation_configuration from XML file
     */
    PARTONS::ModuleObject* m_pDVCSProcessModule;

    /** Pointer to DVMP process module set by computation_configuration from XML file
     */
    PARTONS::ModuleObject* m_pDVMPProcessModule;

    /** Pointer to TCS process module set by computation_configuration from XML file
     */
    PARTONS::ModuleObject* m_pTCSProcessModule;

    /** Compute observables and return chi2
     */
    double computeObservablesAndReturnChi2();

    /** Calculate contribution of points to chi2 value
     @param observableValueReference Reference (experimental) value
     @param observableValueComputed Calculated (model) value
     @param observableErrorReference Related uncertainty of reference value
     */
    double calculateChi2Contribution(const double observableValueReference,
            const double observableValueComputed,
            const double observableErrorReference) const;

    /** Copy and organize data retrieved from data base
     * @param selectedObservableFromDatabaseList List of ObservableResults to be stored in FitsConfiguration
     */
    void makeFitsConfigurationFromDatabaseObjects(
            const PARTONS::List<PARTONS::DVCSObservableResult>& selectedObservableFromDatabaseList);

    /** Smear data for replica method
     */
    void smearDataForReplicaMethod(
            PARTONS::List<PARTONS::DVCSObservableResult>& selectedObservableFromDatabaseList) const;

    /** Select data for random method
     */
    void selectDataForRandomMethod(
            PARTONS::List<PARTONS::DVCSObservableResult>& selectedObservableFromDatabaseList,
            int mode, double fraction) const;
};

#endif /* FITS_SERVICE_H */
