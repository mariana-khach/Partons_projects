//
// Created by Mariana Khachatryan on 9/21/26.
//

#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFScalarTorch.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionResult.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/Scales.h>

#include <complex>
#include <map>
#include <vector>

namespace {

/// The four CFFs the BMJ12 tensor layer consumes, in the order it stores them.
const PARTONS::GPDType::Type kTypes[4] = { PARTONS::GPDType::H,
        PARTONS::GPDType::E, PARTONS::GPDType::Ht, PARTONS::GPDType::Et };

} // namespace

DVCSCFFScalarTorch::DVCSCFFScalarTorch(
        PARTONS::DVCSConvolCoeffFunctionModule* pScalarCFF,
        PARTONS::DVCSXiConverterModule* pXiConverter,
        PARTONS::DVCSScalesModule* pScales)
        : m_pScalarCFF(pScalarCFF), m_pXiConverter(pXiConverter),
          m_pScales(pScales) {

    if (!m_pScalarCFF || !m_pXiConverter || !m_pScales) {
        throw ElemUtils::CustomException("DVCSCFFScalarTorch", __func__,
                "Needs a scalar CFF module, an xi converter and a scales module.");
    }
}

DVCSCFFScalarTorch::AllCFFsTensorBatch
DVCSCFFScalarTorch::computeAllCFFsTensorBatch(const torch::Tensor& xB,
        const torch::Tensor& t, const torch::Tensor& Q2,
        const torch::Tensor& E) {

    const int64_t N = xB.size(0);

    // [4][N] real and imaginary parts, in kTypes order.
    std::vector<std::vector<double> > re(4, std::vector<double>(N, 0.));
    std::vector<std::vector<double> > im(4, std::vector<double>(N, 0.));

    PARTONS::List<PARTONS::GPDType> gpdTypes;
    for (int k = 0; k < 4; ++k)
        gpdTypes.add(PARTONS::GPDType(kTypes[k]));

    for (int64_t i = 0; i < N; ++i) {

        // Same construction as DVCSProcessModule::computeConvolCoeffFunction:
        // xi from the converter, muF2/muR2 from the scales module, both fed the
        // full observable kinematics. phi is irrelevant here (neither module
        // reads it) but the bean requires a value.
        PARTONS::DVCSObservableKinematic kin(xB[i].item<double>(),
                t[i].item<double>(), Q2[i].item<double>(), E[i].item<double>(),
                0.);

        PARTONS::Scales scale = m_pScales->compute(kin);
        PARTONS::PhysicalType<double> xi = m_pXiConverter->compute(kin);

        PARTONS::DVCSConvolCoeffFunctionKinematic ccfKin(xi, kin.getT(),
                kin.getQ2(), scale.getMuF2(), scale.getMuR2());

        PARTONS::DVCSConvolCoeffFunctionResult result =
                m_pScalarCFF->compute(ccfKin, gpdTypes);

        // Read through the map rather than getResult(): a model that does not
        // provide a given CFF leaves it at zero instead of throwing, matching
        // what the network does for types outside its output layer.
        const std::map<PARTONS::GPDType::Type, std::complex<double> >& values =
                result.getResultsByGpdType();

        for (int k = 0; k < 4; ++k) {
            std::map<PARTONS::GPDType::Type, std::complex<double> >::const_iterator it =
                    values.find(kTypes[k]);
            if (it != values.end()) {
                re[k][i] = it->second.real();
                im[k][i] = it->second.imag();
            }
        }
    }

    const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor cff[4];
    for (int k = 0; k < 4; ++k) {
        cff[k] = torch::complex(torch::tensor(re[k], f64),
                torch::tensor(im[k], f64)); // [N] complex double, no grad
    }

    AllCFFsTensorBatch cffs;
    cffs.H  = cff[0];
    cffs.E  = cff[1];
    cffs.Ht = cff[2];
    cffs.Et = cff[3];
    return cffs;
}