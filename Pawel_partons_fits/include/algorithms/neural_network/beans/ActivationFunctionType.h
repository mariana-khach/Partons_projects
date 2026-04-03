/*
 * ActivationFunctionType.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef ACTIVATIONFUNCTIONTYPE_H_
#define ACTIVATIONFUNCTIONTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class ActivationFunctionType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0,
        Hyperbolic = 1,
        Linear = 2,
        Logistic = 3,
        SymetricThreshold = 4,
        Threshold = 5
    };

    ActivationFunctionType();
    ActivationFunctionType(ActivationFunctionType::Type type);
    ActivationFunctionType(const ActivationFunctionType& other);
    ~ActivationFunctionType();
    ActivationFunctionType* clone() const;

    std::string toString() const;

private:

    ActivationFunctionType::Type m_type;
};

#endif /* ACTIVATIONFUNCTIONTYPE_H_ */
