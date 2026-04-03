/*
 * CombinationFunctionType.h
 *
 *  Created on: Apr 24, 2016
 *      Author: Pawel Sznajder
 */

#ifndef COMBINATIONFUNCTIONTYPE_H_
#define COMBINATIONFUNCTIONTYPE_H_

#include <partons/BaseObject.h>
#include <string>

class CombinationFunctionType: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0, ScalarProduct = 1
    };

    CombinationFunctionType();
    CombinationFunctionType(CombinationFunctionType::Type type);
    CombinationFunctionType(const CombinationFunctionType& other);
    ~CombinationFunctionType();
    CombinationFunctionType* clone() const;

    std::string toString() const;

private:

    CombinationFunctionType::Type m_type;
};

#endif /* COMBINATIONFUNCTIONTYPE_H_ */
