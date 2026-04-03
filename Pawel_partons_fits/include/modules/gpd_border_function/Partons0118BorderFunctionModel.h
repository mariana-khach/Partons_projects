/*
 * Partons0118BorderFunctionModel.h
 *
 *  Created on: Jan 17, 2017
 *      Author: Pawel Sznajder (NCBJ)
 */

#ifndef PARTONS0118BORDERFUNCTIONMODEL_H_
#define PARTONS0118BORDERFUNCTIONMODEL_H_

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/modules/gpd/GPDModule.h>
#include <stddef.h>
#include <string>
#include <vector>

class PdfParameterization;

namespace PARTONS {
class GPDService;
} /* namespace PARTONS */

class Partons0118BorderFunctionModel: public PARTONS::GPDModule {

public:

    static const unsigned int classId;

    static const std::string PARAMETER_NAME_PARTONS0118BORDERFUNCTIONMODEL_PDF_REPLICA;
    static const std::string PARAMETER_NAME_PARTONS0118BORDERFUNCTIONMODEL_PDF_REPLICA_POL;

    Partons0118BorderFunctionModel(const std::string &className);
    virtual ~Partons0118BorderFunctionModel();

    virtual Partons0118BorderFunctionModel* clone() const;
    virtual void resolveObjectDependencies();
    void virtual configure(const ElemUtils::Parameters &parameters);
    virtual std::string toString() const;

    virtual PARTONS::PartonDistribution computeH();
    virtual PARTONS::PartonDistribution computeHt();
    virtual PARTONS::PartonDistribution computeE();
    virtual PARTONS::PartonDistribution computeEt();

    PARTONS::GPDModule* getGPDModule() const;
    void setGPDModule(PARTONS::GPDModule* gpdModule);

    double getParEN() const;
    void setParEN(double parEN);
    double getParEtN() const;
    void setParEtN(double parEtN);

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
    const std::vector<double>& getParHtSkewDSea() const;
    void setParHtSkewDSea(const std::vector<double>& parHtSkewDSea);
    const std::vector<double>& getParHtSkewDVal() const;
    void setParHtSkewDVal(const std::vector<double>& parHtSkewDVal);
    const std::vector<double>& getParHtSkewSSea() const;
    void setParHtSkewSSea(const std::vector<double>& parHtSkewSSea);
    const std::vector<double>& getParHtSkewUSea() const;
    void setParHtSkewUSea(const std::vector<double>& parHtSkewUSea);
    const std::vector<double>& getParHtSkewUVal() const;
    void setParHtSkewUVal(const std::vector<double>& parHtSkewUVal);
    const std::vector<double>& getParHtTDepDSea() const;
    void setParHtTDepDSea(const std::vector<double>& parHtTDepDSea);
    const std::vector<double>& getParHtTDepDVal() const;
    void setParHtTDepDVal(const std::vector<double>& parHtTDepDVal);
    const std::vector<double>& getParHtTDepSSea() const;
    void setParHtTDepSSea(const std::vector<double>& parHtTDepSSea);
    const std::vector<double>& getParHtTDepUSea() const;
    void setParHtTDepUSea(const std::vector<double>& parHtTDepUSea);
    const std::vector<double>& getParHtTDepUVal() const;
    void setParHtTDepUVal(const std::vector<double>& parHtTDepUVal);

protected:

    Partons0118BorderFunctionModel(const Partons0118BorderFunctionModel& other);

private:

    virtual double skewnessFunction(
            const std::vector<double>& parameters) const;
    virtual double tDependance(const std::vector<double>& parameters) const;

    PdfParameterization* m_pPdf;

    size_t m_pdfReplica;
    size_t m_pdfReplicaPol;

    PARTONS::GPDModule* m_pGPDModule;
    PARTONS::GPDService* m_pGPDService;

    std::vector<double> m_par_H_skew_uVal;
    std::vector<double> m_par_H_skew_uSea;

    std::vector<double> m_par_H_skew_dVal;
    std::vector<double> m_par_H_skew_dSea;

    std::vector<double> m_par_H_skew_sSea;

    std::vector<double> m_par_Ht_skew_uVal;
    std::vector<double> m_par_Ht_skew_uSea;

    std::vector<double> m_par_Ht_skew_dVal;
    std::vector<double> m_par_Ht_skew_dSea;

    std::vector<double> m_par_Ht_skew_sSea;

    std::vector<double> m_par_H_tDep_uVal;
    std::vector<double> m_par_H_tDep_uSea;

    std::vector<double> m_par_H_tDep_dVal;
    std::vector<double> m_par_H_tDep_dSea;

    std::vector<double> m_par_H_tDep_sSea;

    std::vector<double> m_par_Ht_tDep_uVal;
    std::vector<double> m_par_Ht_tDep_uSea;

    std::vector<double> m_par_Ht_tDep_dVal;
    std::vector<double> m_par_Ht_tDep_dSea;

    std::vector<double> m_par_Ht_tDep_sSea;

    double m_par_E_N;
    double m_par_Et_N;
};

#endif /* PARTONS0118BORDERFUNCTIONMODEL_H_ */
