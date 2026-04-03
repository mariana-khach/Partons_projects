#ifndef FITS_MODEL_MODULE_H
#define FITS_MODEL_MODULE_H

/**
 * @file FitsService.h
 * @author: Bryan BERTHOU (SPhN / CEA Saclay)
 * @date April 18, 2016
 * @version 1.0
 */

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/channel/ChannelType.h>
#include <partons/ModuleObject.h>
#include <map>
#include <string>

/**
 * Mother class for module to be fitted
 */
class FitsModelModule: public PARTONS::ModuleObject {

public:

    /** Constructor
     @param className Class name
     */
    FitsModelModule(const std::string &className);

    /** Destructor
     */
    virtual ~FitsModelModule();

    virtual void resolveObjectDependencies();
    virtual void configure(const ElemUtils::Parameters &parameters);
    virtual FitsModelModule* clone() const = 0;

    /** Initialize parameters
     */
    virtual void initParameters() = 0;

    /** Update parameters
     @param Var Parameters to be updated
     */
    virtual void updateParameters(const double* Var) = 0;

    /** Get parameters
     */
    const ElemUtils::Parameters& getParameters() const;

    /** Set parameters
     @param parameters Parameters to be set
     */
    void setParameters(const ElemUtils::Parameters& parameters);

    void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    virtual void initModule();
    virtual void isModuleWellConfigured();

protected:

    /** Copy constructor
     * @param other Object to be copied
     */
    FitsModelModule(const FitsModelModule &other);

    /** Parameters
     */
    ElemUtils::Parameters m_parameters;

private:

    // ModuleObject* m_pModuleObject;
};

#endif /* FITS_MODEL_MODULE_H */
