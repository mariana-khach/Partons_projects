//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef PROCESS_MODULE_TORCH_H
#define PROCESS_MODULE_TORCH_H

/**
 * @class ProcessModuleTorch
 *
 * @brief Channel-agnostic skeleton for a differentiable (libtorch) process module.
 *
 * Tensor counterpart of PARTONS' ProcessModule<KinematicType, ResultType>. As in
 * the scalar tree, the cross-section API is NOT declared here: its
 * parametrization is channel-specific (DVCS uses sigma(lambda, charge, phi); TCS
 * and DVMP differ), so the tensor cross-section virtuals live on the per-channel
 * subclass (e.g. DVCSProcessModuleTorch), exactly as PARTONS puts
 * compute(lambda, charge, target, ...) on DVCSProcessModule rather than the
 * generic ProcessModule.
 */
template <typename KinematicType>
class ProcessModuleTorch {

public:

    virtual ~ProcessModuleTorch() = default;
};

#endif /* PROCESS_MODULE_TORCH_H */