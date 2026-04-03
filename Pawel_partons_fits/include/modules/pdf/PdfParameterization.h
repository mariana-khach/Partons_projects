/*
 * PdfParameterization.h
 *
 *  Created on: Feb 6, 2018
 *      Author: partons
 */

#ifndef MODULES_PDF_PDFPARAMETERIZATION_H_
#define MODULES_PDF_PDFPARAMETERIZATION_H_

#include <partons/beans/QuarkFlavor.h>
#include <partons/BaseObject.h>
#include <stddef.h>
#include <utility>
#include <vector>

struct PdfParameterizationParameters {
    double Q2ref;
    std::pair<double, double> p[5], beta, alpha;
};

class PdfParameterization: public PARTONS::BaseObject {

public:

    PdfParameterization();
    virtual ~PdfParameterization();

    virtual PdfParameterization* clone() const;

    double pdf(const PARTONS::QuarkFlavor::Type quark, const bool isSea,
            const bool isPolarized, const size_t iReplica, const double x,
            const double Q2);

    double effBeta(const PARTONS::QuarkFlavor::Type quark,
            const bool isSea, const bool isPolarized, const size_t iReplica,const double t,
            const double Q2,const std::vector<double>& tDepPar);

    double lambda(const PARTONS::QuarkFlavor::Type quark, const bool isSea,
            const bool isPolarized, const size_t iReplica, const double x,
            const double t, const double Q2, const std::vector<double>& skewPar,
            const std::vector<double>& tDepPar, const size_t derivative);

private:

    PdfParameterization(const PdfParameterization& other);

private:

    void getParameters(const PARTONS::QuarkFlavor::Type quark, const bool isSea,
            const bool isPolarized, const size_t iReplica);

    bool isTheSame(const PARTONS::QuarkFlavor::Type quark, const bool isSea,
            const bool isPolarized, const size_t iReplica);

    double effParameter(const double Q2, const double Q2ref,
            const std::pair<double, double>& p) const;

    PARTONS::QuarkFlavor::Type m_lastQuark;
    bool m_lastIsSea;
    bool m_lastIsPolarized;
    size_t m_lastIReplica;

    PdfParameterizationParameters m_parameters;

};

#endif /* MODULES_PDF_PDFPARAMETERIZATION_H_ */
