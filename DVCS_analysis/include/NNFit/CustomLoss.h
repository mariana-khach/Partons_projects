//
// Created by Mariana Khachatryan on 6/17/26.
//

#ifndef CUSTOM_LOSS_H
#define CUSTOM_LOSS_H

#include <partons/beans/List.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "NNFit/CFF_NN_Fit.h"                                  // CFFNNModel
#include "NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h" // DVCSObservableTorch alias

class DVCSObservableServiceTorch; // fwd decl (held only as a pointer)

/**
 * @class CustomLossImpl
 *
 * @brief Reduced chi^2 loss on a DVCS observable, trained directly on data.
 *
 * Formula (normalize=true, the default):
 *   chi^2/n = (1/n) sum_{i=1}^{n} ( O(NN(x_i)) - y_i )^2 / sigma_i^2
 *
 * Normalized by n (matches Gepard's CustomLoss/CustomLoss_vectloss, which use
 * torch.mean rather than a raw sum) so the loss is comparable across splits of
 * different size (train vs val) and reads as a standard reduced-chi^2
 * diagnostic (~1 = good fit). normalize=false reverts to the raw sum (the
 * original, pre-normalization loss) — kept only for controlled A/B comparisons
 * against the reduced-chi^2 default, not for normal training.
 *
 * where O is the observable evaluated through the differentiable PARTONS-tensor
 * module chain (DVCSCFFNNTorch -> DVCSProcessBMJ12Torch -> the wired observable),
 * driven by DVCSObservableServiceTorch::computeManyKinematicTorch — the batched
 * (Option A: channel-generic List<K>) sibling of the single-point wiring used
 * by CFF_NN_Fitter::observ_calc_torch(). All N rows are evaluated in one batched
 * call instead of a per-row loop (the 2026-09 #3 speedup, vect_optionA design:
 * the kinematics List is built once per fit -- see fit_once() -- not rebuilt
 * every epoch). With the default wiring the observable is A_LU^{sin1phi}
 * (DVCSAluMinusSin1PhiTorch), but the loss is kept general: the full kinematics
 * (xB, t, Q2, E, phi) are carried by each list element, so it also serves
 * phi-dependent observables. (For the sin1phi moment, phi is integrated out
 * and the supplied value is irrelevant.)
 *
 * The autograd graph runs from the returned chi^2 back to the NN parameters, so
 * the network is trained on the measured observable rather than on CFF labels.
 *
 * torch::nn::Module subclass (wrapped by TORCH_MODULE), instantiated and called
 * like a PyTorch loss module:
 *   CustomLoss loss(net, {"ImH"}, xMin, xMax);
 *   torch::Tensor chi2 = loss(kinematics, y_obs, sigma);
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
     * @param xPow         Power applied to xB as CFF = xB^xPow * NNet_output
     *                     (must match what observ_calc and predict() use).
     * @param normalize    true (default): return chi^2/n. false: return the
     *                     raw chi^2 sum (pre-normalization behavior) — only
     *                     for controlled A/B comparisons.
     */
    CustomLossImpl(CFFNNModel net, const std::vector<std::string>& outputLayer,
            const torch::Tensor& xMin = {}, const torch::Tensor& xMax = {},
            double xPow = 0.0, bool normalize = true);

    /**
     * chi^2 over all rows, evaluated in one batched call through the
     * channel-generic List<DVCSObservableKinematic> (Option A) batched chain.
     * The list is expected to be built once by the caller (e.g. once per
     * fit_once() call, reused across every epoch) rather than rebuilt per
     * forward() call -- see fit_once()'s doc comment.
     *
     * @param kinematics [N] observable kinematics (xB, t, Q2, E, phi per point).
     * @param y_obs      [N] measured observable value.
     * @param sigma      [N] per-point uncertainty.
     * @return 0-d torch::Tensor chi^2, grad-connected to the NN parameters.
     */
    torch::Tensor forward(
            const PARTONS::List<PARTONS::DVCSObservableKinematic>& kinematics,
            const torch::Tensor& y_obs, const torch::Tensor& sigma);

private:

    DVCSObservableServiceTorch* m_pServiceTorch = nullptr; ///< Torch-aware service.
    DVCSObservableTorch*        m_pObsTorch     = nullptr; ///< Wired tensor observable.
    bool m_normalize = true; ///< true: return chi^2/n. false: raw chi^2 sum.
};

TORCH_MODULE(CustomLoss);

#endif /* CUSTOM_LOSS_H */