//
// Created by Mariana Khachatryan on 6/15/26.
//

#ifndef DVCS_ALU_MINUS_SIN1PHI_TORCH_H
#define DVCS_ALU_MINUS_SIN1PHI_TORCH_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <torch/torch.h>

#include <string>

#include "NNFit/Theory/Modules/MathIntegratorModuleTorch.h"
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusTorch.h"

/**
 * @class DVCSAluMinusSin1PhiTorch
 *
 * @brief Differentiable (libtorch) twin of PARTONS::DVCSAluMinusSin1Phi.
 *
 * Computes the beam-spin asymmetry Fourier moment
 *   A_LU^{sin1phi} = (1/pi) * integral_0^{2pi} A_LU(phi) * sin(phi) dphi,
 * entirely in tensors so the gradient flows from the asymmetry back to the NN
 * CFF parameters. The phi integral uses MathIntegratorModuleTorch with DEXP —
 * the same integrator the scalar DVCSAluMinusSin1Phi uses.
 *
 * Mirrors the scalar class hierarchy: derives from DVCSAluMinusTorch (the
 * pointwise asymmetry layer) and reuses its aLUTensor(), exactly as the scalar
 * DVCSAluMinusSin1Phi derives from DVCSAluMinus and reuses computeObservable().
 * The pointwise scalar wrapper (computeObservable) is inherited unchanged — it
 * calls computeTensor() virtually, which resolves to the override below.
 */
class DVCSAluMinusSin1PhiTorch: public DVCSAluMinusTorch,
        public MathIntegratorModuleTorch {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    DVCSAluMinusSin1PhiTorch(const std::string& className);
    virtual ~DVCSAluMinusSin1PhiTorch();

    virtual DVCSAluMinusSin1PhiTorch* clone() const override;

protected:

    DVCSAluMinusSin1PhiTorch(const DVCSAluMinusSin1PhiTorch& other);

    /**
     * Differentiable A_LU^{sin1phi} at the given kinematics. Implemented as a
     * thin N=1 wrapper around computeTensorImplBatch() -- single-point "is"
     * batch-with-N=1, not a separately-maintained implementation (the
     * vect_optionA design decision: one implementation, verified once against
     * the pre-batching values, rather than two siblings cross-checked against
     * each other). Overrides the pointwise ObservableTorch hook from
     * DVCSAluMinusTorch; the scalar drop-in contract (computeObservable()
     * wrapping computeTensor().item()) is unaffected -- it calls this method
     * exactly as before.
     * @return 0-d torch::Tensor, grad-connected to the NN parameters.
     */
    torch::Tensor computeTensorImpl(
            const PARTONS::DVCSObservableKinematic& kinematic) override;

    /**
     * Batched (N-point) sibling of computeTensorImpl() -- the real
     * ObservableTorch<K> batched hook implementation (channel-generic
     * List<DVCSObservableKinematic>). Unpacks the list into
     * xB[N]/t[N]/Q2[N]/E[N] tensors (each kinematic's own phi is ignored --
     * this observable integrates over the full phi range regardless, same
     * as computeTensorImpl()) and reduces to the sin(phi) Fourier moment of
     * aLUTensorBatch(), batched over the GL-10 quadrature nodes shared by
     * every data point -- exactly what aLUTensorBatch()/
     * crossSectionTensorBatch() were built for. computeTensorImpl() (above)
     * is a thin N=1 wrapper around this method, not a separate implementation.
     * @return [N] torch::Tensor, grad-connected to the NN parameters.
     */
    torch::Tensor computeTensorImplBatch(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics)
            override;
};

#endif /* DVCS_ALU_MINUS_SIN1PHI_TORCH_H */