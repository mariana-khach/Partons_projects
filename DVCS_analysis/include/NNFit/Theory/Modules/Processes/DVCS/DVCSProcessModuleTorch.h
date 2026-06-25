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
 * Mirrors the scalar granularity exactly:
 *  - crossSectionBHTensor / crossSectionVCSTensor / crossSectionInterfTensor are
 *    the overridable sub-process atoms, the tensor siblings of
 *    CrossSectionBH / CrossSectionVCS / CrossSectionInterf. They assume the
 *    phi-independent setup has run.
 *  - crossSectionTensor() is the template method that runs the setup once and
 *    sums the selected sub-processes, the tensor sibling of
 *    DVCSProcessModule::compute(..., VCSSubProcessType) = Sum(CrossSection*).
 *
 * A tensor observable holds the attached process through this base pointer and
 * dispatches virtually, so any concrete DVCS tensor process is a drop-in.
 */
class DVCSProcessModuleTorch
        : public ProcessModuleTorch<PARTONS::DVCSObservableKinematic> {

public:

    virtual ~DVCSProcessModuleTorch() = default;

    /**
     * Prepare the phi-independent quantities once for the given kinematics
     * (BMJ12 derived quantities, angular coefficients, and one NN forward for
     * the CFFs). Tensor sibling of DVCSProcessModule::setKinematics + CFF
     * forward. After this, the lightweight crossSectionTensor(lambda, charge,
     * phi) overloads may be called repeatedly — e.g. once per beam helicity —
     * without redoing the (helicity-independent) setup.
     */
    void prepareTensor(const PARTONS::DVCSObservableKinematic& kinematic) {
        setupKinematicsTorch(kinematic);
        m_prepared = true;
    }

    /**
     * Total unpolarized-target DVCS cross section sigma(lambda, phi), batched
     * over phi (all sub-processes). Self-contained (prepare + assemble) — a
     * drop-in for single calls. Tensor sibling of
     * DVCSProcessModule::compute(lambda, charge, target, kinematic).
     */
    virtual torch::Tensor crossSectionTensor(double beamHelicity,
            double beamCharge,
            const PARTONS::DVCSObservableKinematic& kinematic,
            const torch::Tensor& phi) {
        return crossSectionTensor(beamHelicity, beamCharge, kinematic, phi,
                PARTONS::VCSSubProcessType::ALL);
    }

    /**
     * Selectable sub-process cross section. Self-contained: runs prepareTensor()
     * then assembles the requested sub-processes — exactly as
     * DVCSProcessModule::compute(..., VCSSubProcessType) accumulates
     * CrossSectionVCS/BH/Interf.
     * @param processType ALL / DVCS (=VCS) / BH / INT.
     */
    virtual torch::Tensor crossSectionTensor(double beamHelicity,
            double beamCharge,
            const PARTONS::DVCSObservableKinematic& kinematic,
            const torch::Tensor& phi,
            PARTONS::VCSSubProcessType::Type processType) {
        prepareTensor(kinematic);
        return crossSectionTensor(beamHelicity, beamCharge, phi, processType);
    }

    /**
     * Lightweight cross section sigma(lambda, phi), all sub-processes — assumes
     * prepareTensor() already cached the phi-independent setup. Lets a caller
     * (e.g. the asymmetry) prepare once and assemble per helicity, avoiding the
     * redundant setup the self-contained overload would otherwise repeat.
     */
    torch::Tensor crossSectionTensor(double beamHelicity, double beamCharge,
            const torch::Tensor& phi) {
        return crossSectionTensor(beamHelicity, beamCharge, phi,
                PARTONS::VCSSubProcessType::ALL);
    }

    /** Lightweight selectable assemble (assumes prepareTensor() ran). */
    torch::Tensor crossSectionTensor(double beamHelicity, double beamCharge,
            const torch::Tensor& phi,
            PARTONS::VCSSubProcessType::Type processType) {

        if (!m_prepared) {
            throw ElemUtils::CustomException("DVCSProcessModuleTorch", __func__,
                    "crossSectionTensor(lambda, charge, phi) called before "
                    "prepareTensor(); no phi-independent setup is cached.");
        }

        torch::Tensor sigma;
        bool any = false;

        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::DVCS) {
            torch::Tensor v = crossSectionVCSTensor(beamHelicity, beamCharge, phi);
            sigma = any ? sigma + v : v;
            any = true;
        }
        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::BH) {
            torch::Tensor v = crossSectionBHTensor(beamHelicity, beamCharge, phi);
            sigma = any ? sigma + v : v;
            any = true;
        }
        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::INT) {
            torch::Tensor v = crossSectionInterfTensor(beamHelicity, beamCharge, phi);
            sigma = any ? sigma + v : v;
            any = true;
        }

        return sigma;
    }

    /**
     * Bethe-Heitler sub-process sigma_BH(phi), batched. Assumes the
     * phi-independent setup has run (driven through crossSectionTensor).
     */
    virtual torch::Tensor crossSectionBHTensor(double beamHelicity,
            double beamCharge, const torch::Tensor& phi) = 0;

    /** Pure-DVCS (VCS) sub-process sigma_VCS(phi), batched. */
    virtual torch::Tensor crossSectionVCSTensor(double beamHelicity,
            double beamCharge, const torch::Tensor& phi) = 0;

    /** Interference sub-process sigma_I(phi), batched. */
    virtual torch::Tensor crossSectionInterfTensor(double beamHelicity,
            double beamCharge, const torch::Tensor& phi) = 0;

protected:

    /**
     * Prepare the phi-independent quantities for the tensor path (BMJ12 derived
     * quantities, angular coefficients, and one NN forward for the CFFs).
     * Tensor sibling of DVCSProcessModule::setKinematics + CFF forward; called
     * once via prepareTensor() before the sub-process atoms.
     */
    virtual void setupKinematicsTorch(
            const PARTONS::DVCSObservableKinematic& kinematic) = 0;

    /// Set by prepareTensor(); gates the lightweight assemble overloads.
    bool m_prepared = false;
};

#endif /* DVCS_PROCESS_MODULE_TORCH_H */