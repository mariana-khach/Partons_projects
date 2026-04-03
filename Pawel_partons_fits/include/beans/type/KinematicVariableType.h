/*
 * KinematicVariableType.h
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef KINEMATICVARIABLETYPE_H_
#define KINEMATICVARIABLETYPE_H_

#include <partons/BaseObject.h>
#include <string>

/**
 * Fit status type object
 */
class KinematicVariableType: public PARTONS::BaseObject {

public:

    /**
     * Fit status
     */
    enum Type {
        UNDEFINED = 0,
        xi = 1,
        xB = 2,
        t = 3,
        Q2 = 4,
        muF2 = 5,
        muR2 = 6,
        phi = 7,
        E = 8,
        LAST = 9
    };

    /**
     * Default constructor
     */
    KinematicVariableType();

    /**
     * Constructor
     * @param type Type to be set
     */
    KinematicVariableType(KinematicVariableType::Type type);

    /**
     * Constructor
     * @param fitStatusTypeString Name of type to be set
     */
    KinematicVariableType(const std::string& fitStatusTypeString);

    /**
     * Get type name
     */
    const std::string toString();

    /**
     * Get type
     */
    KinematicVariableType::Type getType() const;

    /**
     * Set type
     * @param type Type to be set
     */
    void setType(KinematicVariableType::Type type);

private:

    KinematicVariableType::Type m_type; ///< Type
};

#endif /* KINEMATICVARIABLETYPE_H_ */
