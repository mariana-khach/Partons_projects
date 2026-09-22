//
// Created by Mariana Khachatryan on 9/22/26.
//

#ifndef DVCS_AC_COS2PHI_TORCH_H
#define DVCS_AC_COS2PHI_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAcTorch.h"

/**
 * @class DVCSAcCos2PhiTorch
 *
 * @brief Differentiable twin of PARTONS::DVCSAcCos2Phi: the cos(2phi) Fourier moment of A_C (weight cos(2phi), normalized by pi).
 *
 * Mirrors the scalar class exactly -- same weight, same normalization, same
 * reuse of the parent's pointwise asymmetry as the integrand.
 */
class DVCSAcCos2PhiTorch: public DVCSAcTorch, public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAcCos2PhiTorch(const std::string& className);
    virtual ~DVCSAcCos2PhiTorch();

    virtual DVCSAcCos2PhiTorch* clone() const override;

protected:

    DVCSAcCos2PhiTorch(const DVCSAcCos2PhiTorch& other);

    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    torch::Tensor computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;
};

#endif /* DVCS_AC_COS2PHI_TORCH_H */
