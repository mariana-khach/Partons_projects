#ifndef OBSERVABLE_DATA_FIT_H
#define OBSERVABLE_DATA_FIT_H

/**
 * @file ObservableDataFit.h
 * @author: Bryan BERTHOU (SPhN / CEA Saclay)
 * @date April 11, 2016
 * @version 1.0
 */

#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionResult.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/observable/DVCS/DVCSObservableResult.h>
#include <string>

class FitsService;

namespace PARTONS {
class DVCSConvolCoeffFunctionKinematic;
class DVCSConvolCoeffFunctionResult;
class DVCSObservableKinematic;
class DVCSObservableResult;
class DVCSObservableService;
} /* namespace PARTONS */

namespace PARTONS {
class DVCSObservable;
} /* namespace PARTONS */

/**
 * @class ObservableDataFit
 * @brief Container for single fitted point
 */
class ObservableDataFit: public PARTONS::BaseObject {

public:

    /**
     * Constructor
     * @param observableResult PARTONS::DVCSObservableResult used in initialization of this object
     */
    ObservableDataFit(const PARTONS::DVCSObservableResult &observableResult);

    virtual ~ObservableDataFit();

    void resolveObjectDependencies();
    virtual std::string toString() const;

    /**
     * Self call ObservableService::compute(...)
     * Run at each minimization loop.
     */
    void compute(
            const PARTONS::DVCSConvolCoeffFunctionResult& dvcsConvolCoeffFunctionResult);

    /**
     * Get reference observable result
     */
    const PARTONS::DVCSObservableResult& getReferenceObservableResult() const;

    /**
     * Get fitted observable result
     */
    const PARTONS::DVCSObservableResult& getFittedObservableResult() const;

    /**
     * Get kinematics
     */
    const PARTONS::DVCSObservableKinematic& getObservableKinematic() const;

    /**
     * Get pointer to Observable object
     */
    PARTONS::DVCSObservable* getObservable() const;

    /**
     * Get related cff kinematics
     */
    const PARTONS::DVCSConvolCoeffFunctionKinematic& getCffKinematic() const;

    /**
     * Set related cff kinematics
     */
    void setCffKinematic(
            const PARTONS::DVCSConvolCoeffFunctionKinematic& cffKinematic);

    /**
     * Get index of related kinematics in FitsService::m_cffDVCSKinematicsToFit
     */
    size_t getCffKinematicsIndex() const;

    /**
     * Set index of related kinematics in FitsService::m_cffDVCSKinematicsToFit
     */
    void setCffKinematicsIndex(size_t cffKinematicsIndex);

private:

    FitsService* m_pFitsService; ///< pointer to FitsService
    PARTONS::DVCSObservableService* m_pObservableService; ///< pointer to DVCSObservableService

    PARTONS::DVCSObservable* m_pObservable; ///< pointer to DVCSObservable to calculate value

    PARTONS::DVCSConvolCoeffFunctionKinematic m_cffKinematic; ///< related cff kinematics
    size_t m_cffKinematicsIndex; ///< index of related kinematics in FitsService::m_cffDVCSKinematicsToFit

    PARTONS::DVCSObservableKinematic m_observableKinematic; ///< kinematic to save access time.
    PARTONS::DVCSObservableResult m_referenceObservableResult; ///< reference result
    PARTONS::DVCSObservableResult m_fittedObservableResult; ///< fitted result
};

#endif /* OBSERVABLE_DATA_FIT_H */
