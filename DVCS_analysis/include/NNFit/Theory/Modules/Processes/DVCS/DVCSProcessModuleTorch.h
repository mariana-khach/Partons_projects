//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef DVCS_PROCESS_MODULE_TORCH_H
#define DVCS_PROCESS_MODULE_TORCH_H

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/process/VCSSubProcessType.h>
#include <torch/torch.h>

#include "NNFit/Theory/Modules/Processes/ProcessModuleTorch.h"

/**
 * @class DVCSProcessModuleTorch
 *
 * @brief DVCS channel layer of the tensor process interface — the tensor twin of
 *        PARTONS::DVCSProcessModule.
 *
 * Batched (N-point) API only — the single-kinematic prepare/assemble split was
 * removed once the only live observable leaf (DVCSAluMinusSin1PhiTorch) moved
 * to routing every call, including N=1, through the batched entry points:
 *  - crossSectionBHTensorBatch / crossSectionVCSTensorBatch /
 *    crossSectionInterfTensorBatch are the overridable sub-process atoms, the
 *    tensor siblings of CrossSectionBH / CrossSectionVCS / CrossSectionInterf.
 *    They assume the phi-independent setup has run for the current batch.
 *  - crossSectionTensorBatch() is the template method that runs the setup
 *    once (via prepareTensorBatch()) and sums the selected sub-processes, the
 *    tensor sibling of DVCSProcessModule::compute(..., VCSSubProcessType) =
 *    Sum(CrossSection*).
 *
 * A tensor observable holds the attached process through this base pointer and
 * dispatches virtually, so any concrete DVCS tensor process is a drop-in.
 */
class DVCSCFFModuleTorch;

class DVCSProcessModuleTorch
        : public ProcessModuleTorch<PARTONS::DVCSObservableKinematic> {

public:

    virtual ~DVCSProcessModuleTorch() = default;

    /**
     * Drive the tensor chain from a CFF source that is not the wired PARTONS
     * convol-coeff module, and takes precedence over it. Needed because a
     * DVCSCFFModuleTorch implementation is not required to be a PARTONS module
     * at all -- DVCSCFFScalarTorch, which adapts a scalar CFF model for
     * validation, has no classId and cannot go through
     * setConvolCoeffFunctionModule(). Non-owning; pass nullptr to fall back to
     * the wired module. The usual case (DVCSCFFNNTorch, which IS both) leaves
     * this unset and is found by cross-cast.
     */
    void setCFFModuleTorch(DVCSCFFModuleTorch* pCFFTorch) {
        m_pCFFTorch = pCFFTorch;
    }


    /**
     * Prepare the phi-independent quantities once for N kinematic points at
     * once (BMJ12 derived quantities, angular coefficients, and one batched
     * NN forward for the CFFs). After this, the lightweight
     * crossSectionTensorBatch(lambda, charge, phi) overloads may be called
     * repeatedly — e.g. once per beam helicity — without redoing the
     * (helicity-independent) setup.
     */
    void prepareTensorBatch(const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E) {
        setupKinematicsTorchBatch(xB, t, Q2, E);
        m_preparedBatch = true;
    }

    /**
     * Total unpolarized-target DVCS cross section sigma(lambda, phi), batched
     * over N data points x M phi nodes -- lightweight, assumes
     * prepareTensorBatch() already cached the phi-independent setup.
     */
    torch::Tensor crossSectionTensorBatch(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) {
        return crossSectionTensorBatch(beamHelicity, beamCharge, phi,
                PARTONS::VCSSubProcessType::ALL);
    }

    /** Lightweight selectable batched assemble (assumes prepareTensorBatch() ran). */
    torch::Tensor crossSectionTensorBatch(double beamHelicity, double beamCharge,
            const torch::Tensor& phi, PARTONS::VCSSubProcessType::Type processType) {

        if (!m_preparedBatch) {
            throw ElemUtils::CustomException("DVCSProcessModuleTorch", __func__,
                    "crossSectionTensorBatch() called before "
                    "prepareTensorBatch(); no phi-independent setup is cached.");
        }

        torch::Tensor sigma;
        bool any = false;

        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::DVCS) {
            torch::Tensor v = crossSectionVCSTensorBatch(beamHelicity, beamCharge, phi);
            sigma = any ? sigma + v : v;
            any = true;
        }
        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::BH) {
            torch::Tensor v = crossSectionBHTensorBatch(beamHelicity, beamCharge, phi);
            sigma = any ? sigma + v : v;
            any = true;
        }
        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::INT) {
            torch::Tensor v = crossSectionInterfTensorBatch(beamHelicity, beamCharge, phi);
            sigma = any ? sigma + v : v;
            any = true;
        }

        return sigma;
    }

    /** Batched Bethe-Heitler sub-process sigma_BH(phi), [N,M]. */
    virtual torch::Tensor crossSectionBHTensorBatch(double beamHelicity,
            double beamCharge, const torch::Tensor& phi) = 0;

    /** Batched pure-DVCS (VCS) sub-process sigma_VCS(phi), [N,M]. */
    virtual torch::Tensor crossSectionVCSTensorBatch(double beamHelicity,
            double beamCharge, const torch::Tensor& phi) = 0;

    /** Batched interference sub-process sigma_I(phi), [N,M]. */
    virtual torch::Tensor crossSectionInterfTensorBatch(double beamHelicity,
            double beamCharge, const torch::Tensor& phi) = 0;

protected:

    /**
     * Batched (N-point) sibling of setupKinematicsTorch: the BMJ12 derived
     * quantities and angular coefficients as [N]-tensor arithmetic, plus one
     * batched NN forward for the CFFs. Called once via prepareTensorBatch()
     * before the batched sub-process atoms.
     */
    virtual void setupKinematicsTorchBatch(const torch::Tensor& xB,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& E) = 0;

    /// Optional CFF source overriding the wired convol-coeff module (non-owning).
    DVCSCFFModuleTorch* m_pCFFTorch = nullptr;

    /// Set by prepareTensorBatch(); gates the lightweight batched assemble overloads.
    bool m_preparedBatch = false;
};

#endif /* DVCS_PROCESS_MODULE_TORCH_H */
