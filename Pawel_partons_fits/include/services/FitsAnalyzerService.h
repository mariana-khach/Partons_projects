#ifndef FITSANALYZERSERVICE_H
#define FITSANALYZERSERVICE_H

/*
 * FitsAnalyzerService.h
 *
 *  Created on: Oct 30, 2016
 *      Author: Pawel Sznajder
 */

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/ServiceObject.h>
#include <stddef.h>
#include <map>
#include <string>
#include <vector>

#include "../beans/fit_result/FitResult.h"
#include "../beans/type/KinematicVariableType.h"

/**
 * Utility service to analyze fit results
 */
class FitsAnalyzerService: public PARTONS::ServiceObject {

public:

    /** Unique ID to automatically register the class in the registry
     */
    static const unsigned int classId;

    /** Constructor
     @param className Class name
     */
    FitsAnalyzerService(const std::string &className);

    /** Destructor
     */
    virtual ~FitsAnalyzerService();

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

    /** Configure fit analyzer service
     @param task Task to be performed
     */
    void configureFitAnalyzerService(const PARTONS::Task &task);

    /** Configure fit service
     @param task Task to be performed
     */
    void configureFitService(const PARTONS::Task &task);

    /** Evaluate CFF
     @param task Task to be performed
     */
    void evaluateCFF(const PARTONS::Task &task);

    /** Evaluate observable
     @param task Task to be performed
     */
    void evaluateObservable(const PARTONS::Task &task);

    /** Evaluate chi2 value
     @param task Task to be performed
     */
    void evaluateChi2(const PARTONS::Task &task);

    /** Evaluate chi2 shape 1D
     @param task Task to be performed
     */
    void evaluateChi2Shape1D(const PARTONS::Task &task);

    /** Evaluate chi2 shape 2D
     @param task Task to be performed
     */
    void evaluateChi2Shape2D(const PARTONS::Task &task);

private:

    static const std::string METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE;
    static const std::string METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILES;
    static const std::string METHOD_NAME_CONFIGURE_FIT_ANALYZER_SERVICE_FILE_OUTPUT;

    static const std::string METHOD_NAME_CONFIGURE_FIT_SERVICE;
    static const std::string METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES;
    static const std::string METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_INPUT;
    static const std::string METHOD_NAME_CONFIGURE_FIT_SERVICE_FILES_OUTPUT;

    static const std::string METHOD_NAME_EVALUATE_NSTEPS;
    static const std::string METHOD_NAME_EVALUATE_ISLOG;
    static const std::string METHOD_NAME_EVALUATE_RANGE_MIN;
    static const std::string METHOD_NAME_EVALUATE_RANGE_MAX;

    static const std::string METHOD_NAME_EVALUATE_CFF;
    static const std::string METHOD_NAME_EVALUATE_CFF_DVCS_AFO_XI;
    static const std::string METHOD_NAME_EVALUATE_CFF_DVCS_AFO_T;
    static const std::string METHOD_NAME_EVALUATE_CFF_DVCS_AFO_Q2;
    static const std::string METHOD_NAME_EVALUATE_CFF_DVCS_AFO_MUF2;
    static const std::string METHOD_NAME_EVALUATE_CFF_DVCS_AFO_MUR2;

    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE;
    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE_NAME;
    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE_KINEMATICS;
    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_XB;
    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_T;
    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_Q2;
    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_E;
    static const std::string METHOD_NAME_EVALUATE_OBSERVABLE_DVCS_AFO_PHI;

    static const std::string METHOD_NAME_EVALUATE_CHI2;

    static const std::string METHOD_NAME_EVALUATE_CHI2_SHAPE_1D;
    static const std::string METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR;
    static const std::string METHOD_NAME_EVALUATE_CHI2_SHAPE_1D_PAR_NAME;

    static const std::string METHOD_NAME_EVALUATE_CHI2_SHAPE_2D;
    static const std::string METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR;
    static const std::string METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_1;
    static const std::string METHOD_NAME_EVALUATE_CHI2_SHAPE_2D_PAR_NAME_2;

    /** Get data for evaluation
     * @param parameters Parameters
     * @param kinematicsFixed Values of kinematic variables fixed in evaluation
     * @param nPar Number of parameters to be extracted from xml file
     * @param nSteps Number of steps
     * @param isLog Is log
     * @param min Range min
     * @param max Range max
     * @param outputFile Output file path
     */
    void getDataForEvaluation(const ElemUtils::Parameters& parameters,
            std::map<KinematicVariableType::Type, double>& kinematicsFixed,
            size_t nPar, std::vector<size_t>& nSteps, std::vector<bool>& isLog,
            std::vector<double>& min, std::vector<double>& max) const;

    /**
     * Evaluate DVCS CFFs
     * @param parameters Parameters
     * @param dependence Kind of dependence
     */
    void evaluateCffDVCS(const ElemUtils::Parameters& parameters,
            KinematicVariableType::Type dependence) const;

    /**
     * Evaluate DVCS CFFs
     * @param parameters Parameters
     * @param dependence Kind of dependence
     */
    void evaluateObservableDVCS(const ElemUtils::Parameters& parameters,
            KinematicVariableType::Type dependence) const;

    /**
     * Evaluate chi2 value
     */
    void evaluateChiValue() const;

    /**
     * Evaluate chi2 shape for one parameter
     * @param parameters Parameters
     * @param parName Parameter name
     */
    void evaluateChi2Shape1DPar(const ElemUtils::Parameters& parameters,
            const std::string& parName) const;

    /**
     * Evaluate chi2 shape for two parameters
     * @param parameters Parameters
     * @param parName1 Name of first parameter
     * @param parName2 Name of second parameter
     */
    void evaluateChi2Shape2DPar(const ElemUtils::Parameters& parameters,
            const std::string& parName1, const std::string& parName2) const;

    /** Fit result
     */
    FitResult m_fitResult;

    /** Output file
     */
    std::string m_outputFilePath;
};

#endif /* FITSANALYZERSERVICE_H */
