/*
 * FitsModuleObjectFactory.cpp
 *
 *  Created on: Nov 2, 2016
 *      Author: debian
 */

#include "../include/FitsModuleObjectFactory.h"

#include <partons/BaseObjectFactory.h>

#include "../include/modules/fits_model/FitsModelModule.h"
#include "../include/modules/minimizer_model/FitsMinimizerModule.h"

using namespace PARTONS;

FitsModuleObjectFactory::FitsModuleObjectFactory(
        BaseObjectFactory* pBaseObjectFactory) :
        m_pBaseObjectFactory(pBaseObjectFactory) {
}

FitsModuleObjectFactory::~FitsModuleObjectFactory() {
}

ModuleObject* FitsModuleObjectFactory::newModuleObject(unsigned int classId) {
    return static_cast<ModuleObject*>(m_pBaseObjectFactory->newBaseObject(
            classId));
}

ModuleObject* FitsModuleObjectFactory::newModuleObject(
        const std::string& className) {
    return static_cast<ModuleObject*>(m_pBaseObjectFactory->newBaseObject(
            className));
}

FitsModelModule* FitsModuleObjectFactory::newFitsModule(
        const std::string& className) {
    return static_cast<FitsModelModule*>(newModuleObject(className));
}

FitsModelModule* FitsModuleObjectFactory::newFitsModule(unsigned int classId) {
    return static_cast<FitsModelModule*>(newModuleObject(classId));
}

FitsMinimizerModule* FitsModuleObjectFactory::newFitsMinimizerModule(
        const std::string& className) {
    return static_cast<FitsMinimizerModule*>(newModuleObject(className));
}

FitsMinimizerModule* FitsModuleObjectFactory::newFitsMinimizerModule(
        unsigned int classId) {
    return static_cast<FitsMinimizerModule*>(newModuleObject(classId));
}
