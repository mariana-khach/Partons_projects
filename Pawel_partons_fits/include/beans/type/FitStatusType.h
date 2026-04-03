/*
 * FitStatusType.h
 *
 *  Created on: Nov 3, 2016
 *      Author: Pawel Sznajder (IPNO)
 */

#ifndef FITSTATUSTYPE_H_
#define FITSTATUSTYPE_H_

#include <partons/BaseObject.h>
#include <string>

/**
 * Fit status type object
 */
class FitStatusType: public PARTONS::BaseObject {

public:

    /**
     * Fit status
     */
    enum Type {
        UNDEFINED = 0, SUCCESSFUL = 1, PROBLEMS = 2, FAILED = 3
    };

    /**
     * Default constructor
     */
    FitStatusType();

    /**
     * Constructor
     * @param type Type to be set
     */
    FitStatusType(FitStatusType::Type type);

    /**
     * Constructor
     * @param fitStatusTypeString Name of type to be set
     */
    FitStatusType(const std::string& fitStatusTypeString);

    /**
     * Get type name
     */
    const std::string toString();

    /**
     * Get type
     */
    FitStatusType::Type getType() const;

    /**
     * Set type
     * @param type Type to be set
     */
    void setType(FitStatusType::Type type);

private:

    FitStatusType::Type m_type; ///< Type
};

#endif /* FITSTATUSTYPE_H_ */
