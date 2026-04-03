/*
 * PartonsFits.cpp
 *
 *  Created on: Nov 2, 2016
 *      Author: debian
 */

#include "../include/PartonsFits.h"

#include <partons/Partons.h>

#include "../include/FitsModuleObjectFactory.h"

using namespace PARTONS;

PartonsFits* PartonsFits::m_pInstance = 0;

PartonsFits::PartonsFits() {

    m_pPartons = Partons::getInstance();
    m_pFitsModuleObjectFactory = new FitsModuleObjectFactory(
            m_pPartons->getBaseObjectFactory());
}

PartonsFits* PartonsFits::getInstance() {

    if (!m_pInstance) {
        m_pInstance = new PartonsFits();
    }

    return m_pInstance;
}

PartonsFits::~PartonsFits() {

    if (m_pFitsModuleObjectFactory) {
        delete m_pFitsModuleObjectFactory;
        m_pFitsModuleObjectFactory = 0;
    }
}

void PartonsFits::init(int argc, char** argv) {
    m_pPartons->init(argc, argv);
}

void PartonsFits::close() {
    m_pPartons->close();
}

FitsModuleObjectFactory* PartonsFits::getFitsModuleObjectFactory() const{
    return m_pFitsModuleObjectFactory;
}
