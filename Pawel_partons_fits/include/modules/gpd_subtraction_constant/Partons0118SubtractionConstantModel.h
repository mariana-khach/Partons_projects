/*
 * Partons0118SubtractionConstantModel.h
 *
 *  Created on: Nov 07, 2017
 *      Author: Pawel Sznajder (NCBJ)
 */

#ifndef PARTONS0118SUBTRACTIONCONSTANTMODEL_H_
#define PARTONS0118SUBTRACTIONCONSTANTMODEL_H_

#include <partons/beans/QuarkFlavor.h>
#include <partons/modules/gpd_subtraction_constant/GPDSubtractionConstantModule.h>
#include <partons/modules/MathIntegratorModule.h>
#include <partons/utils/type/PhysicalType.h>
#include <stddef.h>
#include <string>
#include <vector>

class PdfParameterization;

class Partons0118SubtractionConstantModel: public PARTONS::GPDSubtractionConstantModule,
        public PARTONS::MathIntegratorModule {

public:

    static const unsigned int classId;

    static const std::string PARAMETER_NAME_PARTONS0118SUBTRACTIONCONSTANTMODEL_PDF_REPLICA;

    Partons0118SubtractionConstantModel(const std::string &className);
    virtual ~Partons0118SubtractionConstantModel();
    virtual Partons0118SubtractionConstantModel* clone() const;
    virtual void resolveObjectDependencies();

    const std::vector<double>& getParHSkewDSea() const;
    void setParHSkewDSea(const std::vector<double>& parHSkewDSea);
    const std::vector<double>& getParHSkewDVal() const;
    void setParHSkewDVal(const std::vector<double>& parHSkewDVal);
    const std::vector<double>& getParHSkewSSea() const;
    void setParHSkewSSea(const std::vector<double>& parHSkewSSea);
    const std::vector<double>& getParHSkewUSea() const;
    void setParHSkewUSea(const std::vector<double>& parHSkewUSea);
    const std::vector<double>& getParHSkewUVal() const;
    void setParHSkewUVal(const std::vector<double>& parHSkewUVal);
    const std::vector<double>& getParHTDepDSea() const;
    void setParHTDepDSea(const std::vector<double>& parHTDepDSea);
    const std::vector<double>& getParHTDepDVal() const;
    void setParHTDepDVal(const std::vector<double>& parHTDepDVal);
    const std::vector<double>& getParHTDepSSea() const;
    void setParHTDepSSea(const std::vector<double>& parHTDepSSea);
    const std::vector<double>& getParHTDepUSea() const;
    void setParHTDepUSea(const std::vector<double>& parHTDepUSea);
    const std::vector<double>& getParHTDepUVal() const;
    void setParHTDepUVal(const std::vector<double>& parHTDepUVal);

protected:

    Partons0118SubtractionConstantModel(
            const Partons0118SubtractionConstantModel& other);

    virtual PARTONS::PhysicalType<double> computeSubtractionConstant();

private:

    double integralFunction(double x, const std::vector<double>& par);
    double computeSubtractionConstantForOneFlavor();

    PdfParameterization* m_pPdf;

    size_t m_pdfReplica;

    PARTONS::QuarkFlavor::Type m_tmpQuark;
    bool m_tmpIsSea;

    std::vector<double> m_tmpPar_H_skew;
    std::vector<double> m_tmpPar_H_tDep;

    double m_tmpEffBeta;
    double m_tmpD0;
    double m_tmpD1;

    std::vector<double> m_par_H_skew_uVal;
    std::vector<double> m_par_H_skew_uSea;

    std::vector<double> m_par_H_skew_dVal;
    std::vector<double> m_par_H_skew_dSea;

    std::vector<double> m_par_H_skew_sSea;

    std::vector<double> m_par_H_tDep_uVal;
    std::vector<double> m_par_H_tDep_uSea;

    std::vector<double> m_par_H_tDep_dVal;
    std::vector<double> m_par_H_tDep_dSea;

    std::vector<double> m_par_H_tDep_sSea;

    NumA::FunctionType1D* m_pint_IntegralFunction;
    void initFunctorsForIntegrations();
};

#endif /* PARTONS0118SUBTRACTIONCONSTANTMODEL_H_ */
