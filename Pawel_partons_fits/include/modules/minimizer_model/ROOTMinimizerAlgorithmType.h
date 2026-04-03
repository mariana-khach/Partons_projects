/*
 * ROOTMinimizerAlgorithmType.h
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef ROOTMINIMIZERALGORITHMTYPE_H_
#define ROOTMINIMIZERALGORITHMTYPE_H_

#include <partons/BaseObject.h>
#include <string>

/**
 * ROOT minimizer algorithm type object
 */
class ROOTMinimizerAlgorithmType: public PARTONS::BaseObject {

public:

    /**
     * Minimizer algorithm
     */
    enum Type {
        UNDEFINED = 0,
        Default = 1,
        Migrad = 2,
        Simplex = 3,
        Combined = 4,
        Scan = 5,
        Fumili2 = 6,
        ConjugateFR = 7,
        ConjugatePR = 8,
        BFGS = 9,
        BFGS2 = 10,
        SteepestDescent = 11
    };

    /**
     * Default constructor
     */
    ROOTMinimizerAlgorithmType();

    /**
     * Constructor
     * @param type Type to be set
     */
    ROOTMinimizerAlgorithmType(ROOTMinimizerAlgorithmType::Type type);

    /**
     * Constructor
     * @param fitStatusTypeString Name of type to be set
     */
    ROOTMinimizerAlgorithmType(const std::string& rootMinimizerAlgorithmTypeString);

    /**
     * Get type name
     */
    const std::string toString();

    /**
     * Get type
     */
    ROOTMinimizerAlgorithmType::Type getType() const;

    /**
     * Set type
     * @param type Type to be set
     */
    void setType(ROOTMinimizerAlgorithmType::Type type);

private:

    ROOTMinimizerAlgorithmType::Type m_type; ///< Type
};

#endif /* ROOTMINIMIZERALGORITHMTYPE_H_ */
