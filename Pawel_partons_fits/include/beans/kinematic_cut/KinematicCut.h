/*
 * KinematicCut.h
 *
 *  Created on: Jan 22, 2017
 *      Author: Pawel Sznajder
 */

#ifndef KINEMATICCUT_H_
#define KINEMATICCUT_H_

#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <string>

#include "KinematicCutArithmeticOperator.h"
#include "KinematicCutVariable.h"

class KinematicCut: public PARTONS::BaseObject {

public:

    KinematicCut();
    KinematicCut(const std::string& data);
    KinematicCut(const KinematicCut& other);
    virtual ~KinematicCut();
    KinematicCut* clone();

    std::string toString() const;

    bool evaluate(const PARTONS::DVCSObservableKinematic& kinematics) const;

private:

    KinematicCutVariable::Type parseCutVariable(const std::string& data) const;
    KinematicCutArithmeticOperator::Type parseCutArithmeticOperator(const std::string& data) const;
    double parseCutValue(const std::string& data) const;

    KinematicCutVariable::Type m_variable;

    bool m_lActive;
    KinematicCutArithmeticOperator::Type m_lOperator;
    double m_lValue;

    bool m_rActive;
    KinematicCutArithmeticOperator::Type m_rOperator;
    double m_rValue;
};

#endif /* KINEMATICCUT_H_ */
