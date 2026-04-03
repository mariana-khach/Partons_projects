/*
 * PartonsFits.h
 *
 *  Created on: Nov 2, 2016
 *      Author: debian
 */

#ifndef PARTONSFITS_H_
#define PARTONSFITS_H_

class FitsModuleObjectFactory;
namespace PARTONS {
class Partons;
} /* namespace PARTONS */

class PartonsFits {

public:

    /**
     * Share a unique pointer of this class
     */
    static PartonsFits* getInstance();

    /**
     * Default destructor
     */
    virtual ~PartonsFits();

    /**
     * Initialize
     * @param argc Number of arguments
     * @param argv Table of arguments
     */
    void init(int argc, char** argv);

    /**
     * Close
     */
    void close();

    /**
     * Get pointer to FitsModuleObjectFactory
     */
    FitsModuleObjectFactory* getFitsModuleObjectFactory() const;

private:

    /**
     * Default constructor
     */
    PartonsFits();

    /**
     * Private pointer of this class for a unique instance
     */
    static PartonsFits* m_pInstance;

    /**
     * Pointer to PARTONS
     */
    PARTONS::Partons* m_pPartons;

    /**
     * Pointer to FitsModuleObjectFactory
     */
    FitsModuleObjectFactory* m_pFitsModuleObjectFactory;
};

#endif /* PARTONSFITS_H_ */
