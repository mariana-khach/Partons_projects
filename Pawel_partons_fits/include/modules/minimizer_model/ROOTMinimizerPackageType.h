/*
 * ROOTMinimizerPackageType.h
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef ROOTMINIMIZERPACKAGETYPE_H_
#define ROOTMINIMIZERPACKAGETYPE_H_

#include <partons/BaseObject.h>
#include <string>

/**
 * ROOT minimizer package type object
 */
class ROOTMinimizerPackageType: public PARTONS::BaseObject {

public:

    /**
     * Minimizer package
     */
    enum Type {
        UNDEFINED = 0,
        Default = 1,
        Minuit = 2,
        Minuit2 = 3,
        Fumili = 4,
        GSLMultiMin = 5,
        GSLMultiFit = 6,
        GSLSimAn = 7,
        Genetic = 8
    };

    /**
     * Default constructor
     */
    ROOTMinimizerPackageType();

    /**
     * Constructor
     * @param type Type to be set
     */
    ROOTMinimizerPackageType(ROOTMinimizerPackageType::Type type);

    /**
     * Constructor
     * @param fitStatusTypeString Name of type to be set
     */
    ROOTMinimizerPackageType(const std::string& rootMinimizerPackageTypeString);

    /**
     * Get type name
     */
    const std::string toString();

    /**
     * Get type
     */
    ROOTMinimizerPackageType::Type getType() const;

    /**
     * Set type
     * @param type Type to be set
     */
    void setType(ROOTMinimizerPackageType::Type type);

private:

    ROOTMinimizerPackageType::Type m_type; ///< Type
};

#endif /* ROOTMINIMIZERPACKAGETYPE_H_ */
