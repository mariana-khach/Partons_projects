/*
 * KinematicCutVariable.h
 *
 *  Created on: Jan 22, 2017
 *      Author: Pawel Sznajder
 */

#ifndef KINEMATICCUTVARIABLE_H_
#define KINEMATICCUTVARIABLE_H_

#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <string>

class KinematicCutVariable: public PARTONS::BaseObject {

public:

    enum Type {
        UNDEFINED = 0,
        Xb = 1,
        T = 2,
        mT = 3,
        Q2 = 4,
        E = 5,
        Phi = 6,
        ToverQ2 = 7,
        mToverQ2 = 8,
        LAST = 9
    };

    KinematicCutVariable();
    KinematicCutVariable(KinematicCutVariable::Type type);
    KinematicCutVariable(const KinematicCutVariable& other);
    virtual ~KinematicCutVariable();
    KinematicCutVariable* clone();

    std::string toString() const;

    double evaluate(const PARTONS::DVCSObservableKinematic& kinematics) const;

private:

    KinematicCutVariable::Type m_type;
};

#endif /* KINEMATICCUTVARIABLE_H_ */
