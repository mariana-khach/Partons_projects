/*
 * FitsModuleObjectFactory.h
 *
 *  Created on: Nov 2, 2016
 *      Author: debian
 */

#ifndef FITSMODULEOBJECTFACTORY_H_
#define FITSMODULEOBJECTFACTORY_H_

#include <string>

class FitsMinimizerModule;
class FitsModelModule;
namespace PARTONS {
class BaseObjectFactory;
class ModuleObject;
} /* namespace PARTONS */

class FitsModuleObjectFactory {

public:

    virtual ~FitsModuleObjectFactory();

    PARTONS::ModuleObject* newModuleObject(const std::string& className);
    PARTONS::ModuleObject* newModuleObject(unsigned int classId);

    FitsModelModule* newFitsModule(const std::string& className);
    FitsModelModule* newFitsModule(unsigned int classId);

    FitsMinimizerModule* newFitsMinimizerModule(const std::string& className);
    FitsMinimizerModule* newFitsMinimizerModule(unsigned int classId);

private:

    // To allow only PartonsFits class to create a new instance of this class.
    // Used to avoid multiple singleton class and to avoid multithreading problem especially when getInstance() is called.
    // There is a bad behaviour with first instance initialization and mutex.
    friend class PartonsFits;

    /**
     * Private default constructor to ensure the creation of a single instance of the class, managed by PartonFits's class.
     *
     * @param pBaseObjectFactory
     */
    FitsModuleObjectFactory(PARTONS::BaseObjectFactory* pBaseObjectFactory);

    PARTONS::BaseObjectFactory* m_pBaseObjectFactory;
};

#endif /* FITSMODULEOBJECTFACTORY_H_ */
