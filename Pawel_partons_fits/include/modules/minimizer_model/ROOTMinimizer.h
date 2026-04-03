/*
 * ROOTMinimizer.h
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef ROOTMINIMIZER_H_
#define ROOTMINIMIZER_H_

#include <ElementaryUtils/parameters/Parameters.h>
#include <Math/Functor.h>
#include <stddef.h>
#include <string>
#include <utility>

#include "../../beans/type/FitStatusType.h"
#include "FitsMinimizerModule.h"
#include "ROOTMinimizerAlgorithmType.h"
#include "ROOTMinimizerPackageType.h"

namespace ROOT {
namespace Math {
class Minimizer;
} /* namespace Math */
} /* namespace ROOT */

class ROOTMinimizer: public FitsMinimizerModule {

public:

    static const std::string PARAMETER_NAME_PACKAGE_NAME;
    static const std::string PARAMETER_NAME_ALGORITHM_NAME;
    static const std::string PARAMETER_NAME_PRINT_LEVEL;
    static const std::string PARAMETER_NAME_TOLERANCE;
    static const std::string PARAMETER_NAME_MAX_ITERATIONS;
    static const std::string PARAMETER_NAME_MAX_FCN_CALLS;

    static const size_t classId;

    ROOTMinimizer(const std::string& className);
    virtual ~ROOTMinimizer();

    virtual void resolveObjectDependencies();
    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual ROOTMinimizer* clone() const;

    virtual size_t SetVariable(const std::string& parName, double initialValue,
            double initialStep);
    virtual size_t SetLimitedVariable(const std::string& parName,
            double initialValue, double initialStep,
            const std::pair<double, double>& limits);
    virtual size_t SetFixedVariable(const std::string& parName,
            double initialValue);
    virtual double getVariableValue(size_t i) const;
    virtual double getVariableValue(const std::string& varName) const;
    virtual double getVariableError(size_t i) const;
    virtual double getVariableError(const std::string& varName) const;
    virtual double getFCNValue() const;
    virtual FitStatusType::Type getFitStatus() const;
    virtual size_t getNCalls() const;
    virtual void runMinimization();
    virtual void setFCN(FitsService* c, fcnType fcn, size_t nPar);
    virtual NumA::MatrixD getCovarianceMatrix() const;
    virtual NumA::MatrixD getCorrelationMatrix() const;

protected:

    ROOTMinimizer(const ROOTMinimizer &other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    ROOT::Math::Minimizer* m_pMinimizer;
    ROOT::Math::Functor m_fcnFunctor;

    ROOTMinimizerPackageType::Type m_packageType;
    ROOTMinimizerAlgorithmType::Type m_algorithmType;

    void setNewMinimizer();
};

#endif /* ROOTMINIMIZER_H_ */
