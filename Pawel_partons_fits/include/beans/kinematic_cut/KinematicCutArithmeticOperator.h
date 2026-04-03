/*
 * KinematicCutArithmeticOperator.h
 *
 *  Created on: Jan 22, 2017
 *      Author: Pawel Sznajder
 */

#ifndef KINEMATICCUTARITHMETICOPERATOR_H_
#define KINEMATICCUTARITHMETICOPERATOR_H_

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <partons/BaseObject.h>
#include <string>

class KinematicCutArithmeticOperator: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0, eq = 1, ne = 2, lt = 3, le = 4, gt = 5, ge = 6, LAST = 7
    };

    KinematicCutArithmeticOperator();
    KinematicCutArithmeticOperator(KinematicCutArithmeticOperator::Type type);
    KinematicCutArithmeticOperator(const KinematicCutArithmeticOperator& other);
    virtual ~KinematicCutArithmeticOperator();
    KinematicCutArithmeticOperator* clone();

    std::string toString() const;

    template<typename T> bool evaluate(T lhs, T rhs) const {

        switch (m_type) {

        case KinematicCutArithmeticOperator::eq: {
            return lhs == rhs;
        }

        case KinematicCutArithmeticOperator::ne: {
            return lhs != rhs;
        }

        case KinematicCutArithmeticOperator::lt: {
            return lhs < rhs;
        }

        case KinematicCutArithmeticOperator::le: {
            return lhs <= rhs;
        }

        case KinematicCutArithmeticOperator::gt: {
            return lhs > rhs;
        }

        case KinematicCutArithmeticOperator::ge: {
            return lhs >= rhs;
        }

        default: {

            throw ElemUtils::CustomException("KinematicCutArithmeticOperator",
                    __func__,
                    ElemUtils::Formatter() << "Unable to evaluate for "
                            << toString());
            return false;
        }

        }
    }

private:

    KinematicCutArithmeticOperator::Type m_type;
};

#endif /* KINEMATICCUTARITHMETICOPERATOR_H_ */
