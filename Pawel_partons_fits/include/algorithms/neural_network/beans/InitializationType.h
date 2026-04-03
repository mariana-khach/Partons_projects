/*
 * InitializationType.h
 *
 *  Created on: Jul 28, 2016
 *      Author: Pawel Sznajder
 */

#ifndef INITIALIZATIONTYPE_H_
#define INITIALIZATIONTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class InitializationType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0, Static = 1, Random = 2
    };

    InitializationType();
    InitializationType(InitializationType::Type type);
    InitializationType(const InitializationType& other);
    ~InitializationType();
    InitializationType* clone() const;

    std::string toString() const;

private:

    InitializationType::Type m_type;
};

#endif /* INITIALIZATIONTYPE_H_ */
