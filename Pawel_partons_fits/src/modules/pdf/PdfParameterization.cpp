/*
 * PdfParameterization.cpp
 *
 *  Created on: Feb 6, 2018
 *      Author: partons
 */

#include "../../../include/modules/pdf/PdfParameterization.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/string_utils/Formatter.h>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../../../include/modules/pdf/downSea.h"
#include "../../../include/modules/pdf/downSeaPol.h"
#include "../../../include/modules/pdf/downVal.h"
#include "../../../include/modules/pdf/downValPol.h"
#include "../../../include/modules/pdf/strangeSea.h"
#include "../../../include/modules/pdf/strangeSeaPol.h"
#include "../../../include/modules/pdf/upSea.h"
#include "../../../include/modules/pdf/upSeaPol.h"
#include "../../../include/modules/pdf/upVal.h"
#include "../../../include/modules/pdf/upValPol.h"

using namespace PARTONS;

PdfParameterization::PdfParameterization() :
        BaseObject("PdfParameterization") {

    m_lastQuark = QuarkFlavor::UNDEFINED;
    m_lastIsSea = false;
    m_lastIsPolarized = false;
    m_lastIReplica = 0;
}

PdfParameterization::PdfParameterization(const PdfParameterization& other) :
        BaseObject(other) {

    m_lastQuark = other.m_lastQuark;
    m_lastIsSea = other.m_lastIsSea;
    m_lastIsPolarized = other.m_lastIsPolarized;
    m_lastIReplica = other.m_lastIReplica;

    m_parameters = other.m_parameters;
}

PdfParameterization* PdfParameterization::clone() const {
    return new PdfParameterization(*this);
}

PdfParameterization::~PdfParameterization() {
}

void PdfParameterization::getParameters(const QuarkFlavor::Type quark,
        const bool isSea, const bool isPolarized, const size_t iReplica) {

    //get appropriate table
    const double (*data)[17];
    const size_t* nReplicas;

    //get value
    switch (quark) {

    case QuarkFlavor::UP: {
        if (isPolarized) {
            data = (isSea) ? (upSeaPol) : (upValPol);
            nReplicas = (isSea) ? (&nUpSeaPol) : (&nUpValPol);
        } else {
            data = (isSea) ? (upSea) : (upVal);
            nReplicas = (isSea) ? (&nUpSea) : (&nUpVal);
        }
        break;
    }
    case QuarkFlavor::DOWN: {
        if (isPolarized) {
            data = (isSea) ? (downSeaPol) : (downValPol);
            nReplicas = (isSea) ? (&nDownSeaPol) : (&nDownValPol);
        } else {
            data = (isSea) ? (downSea) : (downVal);
            nReplicas = (isSea) ? (&nDownSea) : (&nDownVal);
        }
        break;
    }
    case QuarkFlavor::STRANGE: {
        if (isPolarized) {
            data = strangeSeaPol;
            nReplicas = &nStrangeSeaPol;
        } else {
            data = strangeSea;
            nReplicas = &nStrangeSea;
        }
        break;
    }

    default: {

        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "No parameterization for parton "
                        << QuarkFlavor(quark).toString());

        break;
    }

    }

    //check replica id
    if (iReplica >= *nReplicas) {

        throw ElemUtils::CustomException(getClassName(), __func__,
                ElemUtils::Formatter() << "Illegal replica id " << iReplica);
    }

    //set parameters
    m_parameters.Q2ref = data[iReplica][2];

    m_parameters.beta = std::make_pair(data[iReplica][3], data[iReplica][4]);
    m_parameters.alpha = std::make_pair(data[iReplica][5], data[iReplica][6]);

    for (size_t j = 0; j < 5; j++) {
        m_parameters.p[j] = std::make_pair(data[iReplica][7 + 2 * j],
                data[iReplica][7 + 2 * j + 1]);
    }
}

bool PdfParameterization::isTheSame(const PARTONS::QuarkFlavor::Type quark,
        const bool isSea, const bool isPolarized, const size_t iReplica) {

    if (quark != m_lastQuark || isSea != m_lastIsSea
            || isPolarized != m_lastIsPolarized || iReplica != m_lastIReplica) {

        m_lastQuark = quark;
        m_lastIsSea = isSea;
        m_lastIsPolarized = isPolarized;
        m_lastIReplica = iReplica;

        return false;
    }

    return true;
}

double PdfParameterization::effParameter(const double Q2, const double Q2ref,
        const std::pair<double, double>& p) const {
    return p.first + p.second * log(Q2 / Q2ref);
}

double PdfParameterization::pdf(const PARTONS::QuarkFlavor::Type quark,
        const bool isSea, const bool isPolarized, const size_t iReplica,
        const double x, const double Q2) {

    if (!isTheSame(quark, isSea, isPolarized, iReplica))
        getParameters(quark, isSea, isPolarized, iReplica);

    return pow(1. - x, effParameter(Q2, m_parameters.Q2ref, m_parameters.alpha))
            * pow(x,
                    -1.
                            * effParameter(Q2, m_parameters.Q2ref,
                                    m_parameters.beta))
            * (pow(x, 0)
                    * effParameter(Q2, m_parameters.Q2ref, m_parameters.p[0])
                    + pow(x, 1)
                            * effParameter(Q2, m_parameters.Q2ref,
                                    m_parameters.p[1])
                    + pow(x, 2)
                            * effParameter(Q2, m_parameters.Q2ref,
                                    m_parameters.p[2])
                    + pow(x, 3)
                            * effParameter(Q2, m_parameters.Q2ref,
                                    m_parameters.p[3])
                    + pow(x, 4)
                            * effParameter(Q2, m_parameters.Q2ref,
                                    m_parameters.p[4]));
}

double PdfParameterization::lambda(const PARTONS::QuarkFlavor::Type quark,
        const bool isSea, const bool isPolarized, const size_t iReplica,
        const double x, const double t, const double Q2,
        const std::vector<double>& skewPar, const std::vector<double>& tDepPar,
        const size_t derivative) {

    if (!isTheSame(quark, isSea, isPolarized, iReplica))
        getParameters(quark, isSea, isPolarized, iReplica);

    double alpha = effParameter(Q2, m_parameters.Q2ref, m_parameters.alpha);

    double s0 = effParameter(Q2, m_parameters.Q2ref, m_parameters.p[0]);
    double s1 = effParameter(Q2, m_parameters.Q2ref, m_parameters.p[1]);
    double s2 = effParameter(Q2, m_parameters.Q2ref, m_parameters.p[2]);
    double s3 = effParameter(Q2, m_parameters.Q2ref, m_parameters.p[3]);
    double s4 = effParameter(Q2, m_parameters.Q2ref, m_parameters.p[4]);

    double c = skewPar.at(0);

    double a0 = tDepPar.at(0);
    double a1 = tDepPar.at(1);
    double a2 = tDepPar.at(2);

    switch (derivative) {

    case 0: {

        return -2
                * pow(M_E,
                        t
                                * (a1 * pow(1 - x, 3) + a2 * pow(1 - x, 2) * x
                                        + a0 * (-1 + pow(1 - x, 3)) * log(1 / x)))
                * pow(1. - x, alpha)
                * (s0 + s1 * x + s2 * pow(x, 2) + s3 * pow(x, 3)
                        + s4 * pow(x, 4)) * (-1. + c / pow(1 - pow(x, 2), 2));

        break;
    }

    case 1: {

        return (-8 * c
                * pow(M_E,
                        t
                                * (a1 * pow(1 - x, 3) + a2 * pow(1 - x, 2) * x
                                        + a0 * (-1 + pow(1 - x, 3)) * log(1 / x)))
                * pow(1. - x, alpha) * x
                * (s0 + s1 * x + s2 * pow(x, 2) + s3 * pow(x, 3)
                        + s4 * pow(x, 4))) / pow(1 - pow(x, 2), 3)
                - 2
                        * pow(M_E,
                                t
                                        * (a1 * pow(1 - x, 3)
                                                + a2 * pow(1 - x, 2) * x
                                                + a0 * (-1 + pow(1 - x, 3))
                                                        * log(1 / x)))
                        * pow(1. - x, alpha)
                        * (s1 + 2 * s2 * x + 3 * s3 * pow(x, 2)
                                + 4 * s4 * pow(x, 3))
                        * (-1. + c / pow(1 - pow(x, 2), 2))
                + 2 * alpha
                        * pow(M_E,
                                t
                                        * (a1 * pow(1 - x, 3)
                                                + a2 * pow(1 - x, 2) * x
                                                + a0 * (-1 + pow(1 - x, 3))
                                                        * log(1 / x)))
                        * pow(1. - x, -1 + alpha)
                        * (s0 + s1 * x + s2 * pow(x, 2) + s3 * pow(x, 3)
                                + s4 * pow(x, 4))
                        * (-1. + c / pow(1 - pow(x, 2), 2))
                - 2
                        * pow(M_E,
                                t
                                        * (a1 * pow(1 - x, 3)
                                                + a2 * pow(1 - x, 2) * x
                                                + a0 * (-1 + pow(1 - x, 3))
                                                        * log(1 / x))) * t
                        * pow(1. - x, alpha)
                        * (s0 + s1 * x + s2 * pow(x, 2) + s3 * pow(x, 3)
                                + s4 * pow(x, 4))
                        * (-1. + c / pow(1 - pow(x, 2), 2))
                        * (-3 * a1 * pow(1 - x, 2) + a2 * pow(1 - x, 2)
                                - (a0 * (-1 + pow(1 - x, 3))) / x
                                - 2 * a2 * (1 - x) * x
                                - 3 * a0 * pow(1 - x, 2) * log(1 / x));

        break;
    }

    case 2: {

        return -2
                * pow(M_E,
                        t
                                * (a1 * pow(1 - x, 3) + a2 * pow(1 - x, 2) * x
                                        + a0 * (-1 + pow(1 - x, 3)) * log(1 / x)))
                * pow(1. - x, alpha)
                * ((8 * c * x
                        * (s1 + 2 * s2 * x + 3 * s3 * pow(x, 2)
                                + 4 * s4 * pow(x, 3))) / pow(1 - pow(x, 2), 3)
                        + c
                                * (s0 + s1 * x + s2 * pow(x, 2) + s3 * pow(x, 3)
                                        + s4 * pow(x, 4))
                                * ((24 * pow(x, 2)) / pow(1 - pow(x, 2), 4)
                                        + 4 / pow(1 - pow(x, 2), 3))
                        + (2 * s2 + 6 * s3 * x + 12 * s4 * pow(x, 2))
                                * (-1. + c / pow(1 - pow(x, 2), 2)))
                + 2
                        * ((4 * c * x
                                * (s0 + s1 * x + s2 * pow(x, 2) + s3 * pow(x, 3)
                                        + s4 * pow(x, 4)))
                                / pow(1 - pow(x, 2), 3)
                                + (s1 + 2 * s2 * x + 3 * s3 * pow(x, 2)
                                        + 4 * s4 * pow(x, 3))
                                        * (-1. + c / pow(1 - pow(x, 2), 2)))
                        * (2 * alpha
                                * pow(M_E,
                                        t
                                                * (a1 * pow(1 - x, 3)
                                                        + a2 * pow(1 - x, 2) * x
                                                        + a0
                                                                * (-1
                                                                        + pow(
                                                                                1
                                                                                        - x,
                                                                                3))
                                                                * log(1 / x)))
                                * pow(1. - x, -1 + alpha)
                                - 2
                                        * pow(M_E,
                                                t
                                                        * (a1 * pow(1 - x, 3)
                                                                + a2
                                                                        * pow(
                                                                                1
                                                                                        - x,
                                                                                2)
                                                                        * x
                                                                + a0
                                                                        * (-1
                                                                                + pow(
                                                                                        1
                                                                                                - x,
                                                                                        3))
                                                                        * log(
                                                                                1
                                                                                        / x)))
                                        * t * pow(1. - x, alpha)
                                        * (-3 * a1 * pow(1 - x, 2)
                                                + a2 * pow(1 - x, 2)
                                                - (a0 * (-1 + pow(1 - x, 3)))
                                                        / x
                                                - 2 * a2 * (1 - x) * x
                                                - 3 * a0 * pow(1 - x, 2)
                                                        * log(1 / x)))
                + (s0 + s1 * x + s2 * pow(x, 2) + s3 * pow(x, 3)
                        + s4 * pow(x, 4)) * (-1. + c / pow(1 - pow(x, 2), 2))
                        * (-2 * (-1 + alpha) * alpha
                                * pow(M_E,
                                        t
                                                * (a1 * pow(1 - x, 3)
                                                        + a2 * pow(1 - x, 2) * x
                                                        + a0
                                                                * (-1
                                                                        + pow(
                                                                                1
                                                                                        - x,
                                                                                3))
                                                                * log(1 / x)))
                                * pow(1. - x, -2 + alpha)
                                + 4 * alpha
                                        * pow(M_E,
                                                t
                                                        * (a1 * pow(1 - x, 3)
                                                                + a2
                                                                        * pow(
                                                                                1
                                                                                        - x,
                                                                                2)
                                                                        * x
                                                                + a0
                                                                        * (-1
                                                                                + pow(
                                                                                        1
                                                                                                - x,
                                                                                        3))
                                                                        * log(
                                                                                1
                                                                                        / x)))
                                        * t * pow(1. - x, -1 + alpha)
                                        * (-3 * a1 * pow(1 - x, 2)
                                                + a2 * pow(1 - x, 2)
                                                - (a0 * (-1 + pow(1 - x, 3)))
                                                        / x
                                                - 2 * a2 * (1 - x) * x
                                                - 3 * a0 * pow(1 - x, 2)
                                                        * log(1 / x))
                                - 2 * pow(1. - x, alpha)
                                        * (pow(M_E,
                                                t
                                                        * (a1 * pow(1 - x, 3)
                                                                + a2
                                                                        * pow(
                                                                                1
                                                                                        - x,
                                                                                2)
                                                                        * x
                                                                + a0
                                                                        * (-1
                                                                                + pow(
                                                                                        1
                                                                                                - x,
                                                                                        3))
                                                                        * log(
                                                                                1
                                                                                        / x)))
                                                * t
                                                * (6 * a1 * (1 - x)
                                                        - 4 * a2 * (1 - x)
                                                        + (a0
                                                                * (-1
                                                                        + pow(
                                                                                1
                                                                                        - x,
                                                                                3)))
                                                                / pow(x, 2)
                                                        + (6 * a0
                                                                * pow(1 - x, 2))
                                                                / x + 2 * a2 * x
                                                        + 6 * a0 * (1 - x)
                                                                * log(1 / x))
                                                + pow(M_E,
                                                        t
                                                                * (a1
                                                                        * pow(
                                                                                1
                                                                                        - x,
                                                                                3)
                                                                        + a2
                                                                                * pow(
                                                                                        1
                                                                                                - x,
                                                                                        2)
                                                                                * x
                                                                        + a0
                                                                                * (-1
                                                                                        + pow(
                                                                                                1
                                                                                                        - x,
                                                                                                3))
                                                                                * log(
                                                                                        1
                                                                                                / x)))
                                                        * pow(t, 2)
                                                        * pow(
                                                                -3 * a1
                                                                        * pow(
                                                                                1
                                                                                        - x,
                                                                                2)
                                                                        + a2
                                                                                * pow(
                                                                                        1
                                                                                                - x,
                                                                                        2)
                                                                        - (a0
                                                                                * (-1
                                                                                        + pow(
                                                                                                1
                                                                                                        - x,
                                                                                                3)))
                                                                                / x
                                                                        - 2 * a2
                                                                                * (1
                                                                                        - x)
                                                                                * x
                                                                        - 3 * a0
                                                                                * pow(
                                                                                        1
                                                                                                - x,
                                                                                        2)
                                                                                * log(
                                                                                        1
                                                                                                / x),
                                                                2)));
        break;
    }

    default: {

        throw ElemUtils::CustomException(getClassName(), __func__,
                "Derivative not implemented");
        return 0.;
        break;
    }
    }

}

double PdfParameterization::effBeta(const PARTONS::QuarkFlavor::Type quark,
        const bool isSea, const bool isPolarized, const size_t iReplica,
        const double t, const double Q2, const std::vector<double>& tDepPar) {

    if (!isTheSame(quark, isSea, isPolarized, iReplica))
        getParameters(quark, isSea, isPolarized, iReplica);

    return effParameter(Q2, m_parameters.Q2ref, m_parameters.beta)
            + tDepPar.at(0) * t;
}
