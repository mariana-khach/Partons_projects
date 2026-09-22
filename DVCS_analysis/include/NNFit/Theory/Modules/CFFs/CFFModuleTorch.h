//
// Created by Mariana Khachatryan on 9/21/26.
//

#ifndef CFF_MODULE_TORCH_H
#define CFF_MODULE_TORCH_H

/**
 * @class CFFModuleTorch
 *
 * @brief Channel-agnostic base of the CFF link of the tensor chain: twin of
 * PARTONS' ConvolCoeffFunctionModule<KinematicType, ResultType>.
 *
 * Completes the parallel between the torch chain and the scalar one. Each link
 * now has a generic template with a channel class under it, mirroring PARTONS:
 *
 *   Observable<K,R>              <->  ObservableTorch<K>
 *     DVCSObservable                    DVCSObservableTorch (alias)
 *   ProcessModule<K,R>           <->  ProcessModuleTorch<K>
 *     DVCSProcessModule                 DVCSProcessModuleTorch
 *   ConvolCoeffFunctionModule<K,R> <->  this
 *     DVCSConvolCoeffFunctionModule     DVCSCFFModuleTorch
 *
 * As with ProcessModuleTorch<K>, this carries no API of its own, and for the
 * same two reasons. First, the lifecycle a PARTONS module needs (clone,
 * configure, run, registration) is inherited by the CONCRETE classes from
 * their PARTONS twin, so the torch side never has to restate it -- which is
 * why these torch generics are thin where PARTONS' are load-bearing. Second,
 * what a CFF module computes is channel-specific: the four CFFs H, E, Ht, Et
 * suit DVCS (and TCS), but not DVMP's transition CFFs, so the struct and the
 * compute signature belong one level down, in the channel class -- exactly
 * where PARTONS puts setXiConverterModule rather than in ProcessModule<K,R>.
 *
 * ResultType is absent because it collapses: every tensor method returns
 * torch::Tensor. Only KinematicType is templated.
 *
 * Note which kinematics the parameter names. PARTONS' scalar CFF module is
 * templated on the CCF kinematics (xi, t, Q2, muF2, muR2), because the process
 * module converts before calling it. The torch chain defers that conversion to
 * the CFF source instead -- the network wants xB directly, and the scalar
 * adapter runs the xi-converter and scales modules itself -- so what flows
 * through this link is OBSERVABLE-level kinematics, and the channel classes
 * instantiate it accordingly.
 *
 * Inherits nothing from PARTONS, deliberately: a concrete class pairs its
 * PARTONS module with this mixin, so a PARTONS base here would be inherited
 * twice; and an implementation is not required to be a PARTONS module at all
 * (DVCSCFFScalarTorch is not one).
 */
template <typename KinematicType>
class CFFModuleTorch {

public:

    virtual ~CFFModuleTorch() = default;
};

#endif /* CFF_MODULE_TORCH_H */