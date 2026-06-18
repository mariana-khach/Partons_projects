//
// Created by Mariana Khachatryan on 6/17/26.
//

#ifndef CUSTOM_LOSS_H
#define CUSTOM_LOSS_H

#include <torch/torch.h>

#include <string>
#include <vector>

#include "NNFit/CFF_NN_Fit.h"                                  // CFFNNModel
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h" // DVCSObservableTorch alias

class DVCSObservableServiceTorch; // fwd decl (held only as a pointer)

/**
 * @class CustomLossImpl
 *
 * @brief chi^2 loss on a DVCS observable, trained directly on data.
 *
 * Formula:
 *   chi^2 = sum_{i=1}^{n} ( O(NN(x_i)) - y_i )^2 / sigma_i^2
 *
 * where O is the observable evaluated through the differentiable PARTONS-tensor
 * module chain (DVCSCFFNNTorch -> DVCSProcessBMJ12Torch -> the wired observable),
 * driven by DVCSObservableServiceTorch::computeSingleKinematicTorch — exactly the
 * wiring of CFF_NN_Fitter::observ_calc_torch(). With the default wiring the
 * observable is A_LU^{sin1phi} (DVCSAluMinusSin1PhiTorch), but the loss is kept
 * general: the full kinematics (xB, t, Q2, E, phi) are passed per point, so it
 * also serves phi-dependent observables. (For the sin1phi moment, phi is
 * integrated out and the supplied value is irrelevant.)
 *
 * The autograd graph runs from the returned chi^2 back to the NN parameters, so
 * the network is trained on the measured observable rather than on CFF labels.
 *
 * torch::nn::Module subclass (wrapped by TORCH_MODULE), instantiated and called
 * like a PyTorch loss module:
 *   CustomLoss loss(net, {"ImH"}, xMin, xMax);
 *   torch::Tensor chi2 = loss(X, E, phi, y_obs, sigma);
 */
class CustomLossImpl : public torch::nn::Module {

public:

    /**
     * Build and wire the *Torch module chain once (same as observ_calc_torch()),
     * injecting the network whose CFFs parametrize the observable.
     *
     * @param net          NN providing the CFFs. Shared by handle — the optimizer
     *                     updates its parameters in place, so the chain always
     *                     sees the current weights.
     * @param outputLayer  CFF output-neuron names (e.g. {"ImH"}).
     * @param xMin,xMax    Optional per-feature min-max scaling for the NN inputs
     *                     (must match what observ_calc* use). Undefined = raw.
     */
    CustomLossImpl(CFFNNModel net, const std::vector<std::string>& outputLayer,
            const torch::Tensor& xMin = {}, const torch::Tensor& xMax = {});

    /**
     * chi^2 over all rows. Each row builds a DVCSObservableKinematic from the full
     * (xB, t, Q2, E, phi) and the predicted observable is compared to its measured
     * value.
     *
     * @param X      [N,3] raw kinematics (xB, t, Q2).
     * @param E      [N]   beam energy per point.
     * @param phi    [N]   azimuthal angle per point (used by phi-dependent
     *                     observables; ignored by the sin1phi moment).
     * @param y_obs  [N]   measured observable value.
     * @param sigma  [N]   per-point uncertainty.
     * @return 0-d torch::Tensor chi^2, grad-connected to the NN parameters.
     */
    torch::Tensor forward(const torch::Tensor& X, const torch::Tensor& E,
            const torch::Tensor& phi, const torch::Tensor& y_obs,
            const torch::Tensor& sigma);

private:

    DVCSObservableServiceTorch* m_pServiceTorch = nullptr; ///< Torch-aware service.
    DVCSObservableTorch*        m_pObsTorch     = nullptr; ///< Wired tensor observable.
};

TORCH_MODULE(CustomLoss);

#endif /* CUSTOM_LOSS_H */