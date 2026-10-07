//
// Created by Mariana Khachatryan on 6/16/26.
//

#ifndef DVCS_PROCESS_MODULE_TORCH_H
#define DVCS_PROCESS_MODULE_TORCH_H

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/process/VCSSubProcessType.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/utils/type/PhysicalType.h>
#include <partons/utils/type/PhysicalUnit.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionResult.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/List.h>
#include <partons/beans/Scales.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <partons/modules/scales/DVCS/DVCSScalesModule.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterModule.h>
#include <torch/torch.h>

#include <complex>
#include <map>
#include <string>
#include <vector>

#include "NNFit/Theory/Modules/CFFs/DVCS/DVCSConvolCoeffFunctionModuleTorch.h"
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

class DVCSProcessModuleTorch
        : public ProcessModuleTorch<PARTONS::DVCSObservableKinematic> {

public:

    virtual ~DVCSProcessModuleTorch() = default;

    /**
     * The tensor interface of a PARTONS process module, or nullptr when it has
     * none (a plain DVCSProcessBMJ12, BM03, ...). A cross-cast: this mixin and
     * PARTONS::DVCSProcessModule are unrelated bases of one complete object,
     * so static_cast cannot express it.
     *
     * The torch observable leaves branch on this: a torch process runs the
     * tensor chain, anything else runs the leaf's native PARTONS twin.
     */
    static DVCSProcessModuleTorch* tryFrom(PARTONS::DVCSProcessModule* pProc) {
        return dynamic_cast<DVCSProcessModuleTorch*>(pProc);
    }

    /**
     * As tryFrom(), but for the tensor path, which has no fallback: a gradient
     * cannot come out of native double arithmetic, so a non-torch process is
     * an error there.
     * @param className the calling observable, for the message.
     */
    static DVCSProcessModuleTorch& from(PARTONS::DVCSProcessModule* pProc,
            const std::string& className) {
        DVCSProcessModuleTorch* pTorch = tryFrom(pProc);
        if (!pTorch) {
            throw ElemUtils::CustomException(className, __func__,
                    "Tensor path requires a DVCSProcessModuleTorch process module.");
        }
        return *pTorch;
    }


    /**
     * Prepare the phi-independent quantities once for N kinematic points at
     * once (BMJ12 derived quantities, angular coefficients, and one batched
     * NN forward for the CFFs). After this, the assemble-only
     * crossSectionTensorBatch(lambda, charge, phi) overloads may be called
     * repeatedly — e.g. once per beam helicity — without redoing the
     * (helicity-independent) setup.
     *
     * What the split is worth, measured 2026-09-22 at N=16, M=40 on the NN
     * chain (3 runs x 300 reps): prepare 2.49-2.68 ms, one assemble 2.27-2.56
     * ms — a ratio of 1.05-1.12, i.e. they cost the SAME. Earlier notes called
     * the assemble "lightweight"; it is not. It evaluates BH, VCS and
     * interference over the whole [N,M] grid, comparable work to the setup.
     * That parity is exactly why the split pays: every avoided re-preparation
     * costs as much as the call that remains. Per observable evaluation,
     * dropping it would cost +34-36% for an A_LU variant (2 cross sections per
     * prepare) and +62-65% for A_C (4).
     *
     * Those are process-layer figures. An epoch also pays the integrand, the
     * chi^2 and backward(), so the end-to-end training penalty is smaller and
     * was not measured; and the benchmark ran 300 forwards with no backward(),
     * so autograd graphs accumulated — which inflates both sides and leaves
     * the ratio more trustworthy than the absolute milliseconds.
     */
    void prepareTensorBatch(const torch::Tensor& xB, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& E) {
        // Same order as PARTONS' DVCSProcessModule::compute():
        // computeConvolCoeffFunction (generic, here) before initModule (the
        // concrete process's own setup, which reads the stored CFFs).
        m_cffsBatch = computeConvolCoeffFunctionTensorBatch(xB, t, Q2, E);
        setupKinematicsTorchBatch(xB, t, Q2, E);
        m_preparedBatch = true;
    }

    /**
     * Total unpolarized-target DVCS cross section sigma(lambda, phi) in
     * GeV^-2, batched over N data points x M phi nodes -- assemble-only: assumes
     * prepareTensorBatch() already cached the phi-independent setup. Not
     * cheap relative to that setup; see prepareTensorBatch() for the numbers.
     */
    PARTONS::PhysicalType<torch::Tensor> crossSectionTensorBatch(
            double beamHelicity, double beamCharge, const torch::Tensor& phi) {
        return crossSectionTensorBatch(beamHelicity, beamCharge, phi,
                PARTONS::VCSSubProcessType::ALL);
    }

    /** Selectable batched assemble (assumes prepareTensorBatch() ran). */
    PARTONS::PhysicalType<torch::Tensor> crossSectionTensorBatch(
            double beamHelicity, double beamCharge, const torch::Tensor& phi,
            PARTONS::VCSSubProcessType::Type processType) {

        if (!m_preparedBatch) {
            throw ElemUtils::CustomException("DVCSProcessModuleTorch", __func__,
                    "crossSectionTensorBatch() called before "
                    "prepareTensorBatch(); no phi-independent setup is cached.");
        }

        // Accumulated as PhysicalType, exactly as DVCSProcessModule::compute
        // does with its PhysicalType<double> value(0., PhysicalUnit::GEVm2):
        // the += are unit-checked, so a sub-process that returned the wrong
        // unit throws here instead of silently contributing a wrong number.
        PARTONS::PhysicalType<torch::Tensor> sigma;
        bool any = false;
        auto add = [&](const PARTONS::PhysicalType<torch::Tensor>& v) {
            if (any) { sigma = sigma + v; } else { sigma = v; any = true; }
        };

        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::DVCS) {
            add(crossSectionVCSTensorBatch(beamHelicity, beamCharge, phi));
        }
        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::BH) {
            add(crossSectionBHTensorBatch(beamHelicity, beamCharge, phi));
        }
        if (processType == PARTONS::VCSSubProcessType::ALL
                || processType == PARTONS::VCSSubProcessType::INT) {
            add(crossSectionInterfTensorBatch(beamHelicity, beamCharge, phi));
        }

        return sigma;
    }

    /** Batched Bethe-Heitler sub-process sigma_BH(phi), [N,M], GeV^-2. */
    virtual PARTONS::PhysicalType<torch::Tensor> crossSectionBHTensorBatch(
            double beamHelicity, double beamCharge,
            const torch::Tensor& phi) = 0;

    /** Batched pure-DVCS (VCS) sub-process sigma_VCS(phi), [N,M], GeV^-2. */
    virtual PARTONS::PhysicalType<torch::Tensor> crossSectionVCSTensorBatch(
            double beamHelicity, double beamCharge,
            const torch::Tensor& phi) = 0;

    /** Batched interference sub-process sigma_I(phi), [N,M], GeV^-2. */
    virtual PARTONS::PhysicalType<torch::Tensor> crossSectionInterfTensorBatch(
            double beamHelicity, double beamCharge,
            const torch::Tensor& phi) = 0;

protected:

    /**
     * The concrete process's own phi-independent setup over N points -- the
     * tensor twin of PARTONS' initModule() in a concrete process: its derived
     * kinematics and angular coefficients, and whatever it builds from the
     * CFFs, which prepareTensorBatch() has already stored in m_cffsBatch. The
     * generic CFF step is not the process's business, exactly as no PARTONS
     * process calls computeConvolCoeffFunction itself.
     */
    virtual void setupKinematicsTorchBatch(const torch::Tensor& xB,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& E) = 0;

    /**
     * The CFFs at N points, as [N] complex tensors -- the tensor twin of
     * PARTONS' DVCSProcessModule::computeConvolCoeffFunction, and like it
     * generic: identical for every DVCS process, so it lives here rather than
     * in a concrete process. The process converts and the CFF module receives:
     * the xi-converter and scales modules the process is wired with turn
     * (xB, t, Q2, E) into the CCF kinematics (xi, t, Q2, muF2, muR2), once per
     * prepare (N scalar calls).
     *
     * Any CFF module PARTONS accepts is accepted here. One implementing
     * DVCSConvolCoeffFunctionModuleTorch (the network) is asked for tensors
     * directly -- the path that carries a gradient; any other is evaluated per
     * point by scalarCFFsTensorBatch().
     *
     * The modules are read through PARTONS' public getters on the process's
     * own PARTONS base (a cross-cast: this mixin is not a PARTONS class).
     */
    DVCSConvolCoeffFunctionModuleTorch::AllCFFsTensorBatch
    computeConvolCoeffFunctionTensorBatch(const torch::Tensor& xB,
            const torch::Tensor& t, const torch::Tensor& Q2,
            const torch::Tensor& E) {

        PARTONS::DVCSProcessModule* self =
                dynamic_cast<PARTONS::DVCSProcessModule*>(this);
        if (!self) {
            throw ElemUtils::CustomException("DVCSProcessModuleTorch", __func__,
                    "A torch process must also derive from PARTONS::DVCSProcessModule.");
        }
        PARTONS::DVCSConvolCoeffFunctionModule* pCCF =
                self->getConvolCoeffFunctionModule();
        if (!pCCF) {
            throw ElemUtils::CustomException(self->getClassName(), __func__,
                    "No convol-coeff function module set.");
        }

        const int64_t N = xB.size(0);
        std::vector<double> xiVec(N), muF2Vec(N), muR2Vec(N);
        for (int64_t i = 0; i < N; ++i) {
            PARTONS::DVCSObservableKinematic kin(xB[i].item<double>(),
                    t[i].item<double>(), Q2[i].item<double>(),
                    E[i].item<double>(), 0.); // phi is irrelevant to both modules
            xiVec[i] = self->getXiConverterModule()->compute(kin).getValue();
            PARTONS::Scales scale = self->getScaleModule()->compute(kin);
            muF2Vec[i] = scale.getMuF2().getValue();
            muR2Vec[i] = scale.getMuR2().getValue();
        }
        const torch::TensorOptions f64 = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor xi   = torch::tensor(xiVec, f64);
        torch::Tensor muF2 = torch::tensor(muF2Vec, f64);
        torch::Tensor muR2 = torch::tensor(muR2Vec, f64);

        DVCSConvolCoeffFunctionModuleTorch* pTorch =
                dynamic_cast<DVCSConvolCoeffFunctionModuleTorch*>(pCCF);
        return pTorch ? pTorch->computeAllCFFsTensorBatch(xi, t, Q2, muF2, muR2)
                      : scalarCFFsTensorBatch(*pCCF, xi, t, Q2, muF2, muR2);
    }

    /**
     * The CFFs of a SCALAR PARTONS CFF module (DVCSCFFConstant,
     * DVCSCFFStandard, ...) as tensors: one ordinary compute() per point, all
     * four CFFs per call, packed into [N] no-grad complex tensors. This is how
     * a torch process accepts any CFF module PARTONS accepts, as the scalar
     * process does; a module implementing DVCSConvolCoeffFunctionModuleTorch is asked for
     * tensors directly instead. No gradient is lost -- a parametric model has
     * no parameters in the graph.
     *
     * Components the model does not provide come back zero, matching the
     * network's behaviour for CFFs outside its output layer.
     * @param xi,t,Q2,muF2,muR2 [N] CCF kinematics, converted by the process.
     */
    static DVCSConvolCoeffFunctionModuleTorch::AllCFFsTensorBatch scalarCFFsTensorBatch(
            PARTONS::DVCSConvolCoeffFunctionModule& scalarCFF,
            const torch::Tensor& xi, const torch::Tensor& t,
            const torch::Tensor& Q2, const torch::Tensor& muF2,
            const torch::Tensor& muR2) {

        // The four CFFs the BMJ12 tensor layer consumes, in its storage order.
        const PARTONS::GPDType::Type types[4] = { PARTONS::GPDType::H,
                PARTONS::GPDType::E, PARTONS::GPDType::Ht, PARTONS::GPDType::Et };

        PARTONS::List<PARTONS::GPDType> gpdTypes;
        for (int k = 0; k < 4; ++k)
            gpdTypes.add(PARTONS::GPDType(types[k]));

        const int64_t N = xi.size(0);
        std::vector<std::vector<double> > re(4, std::vector<double>(N, 0.));
        std::vector<std::vector<double> > im(4, std::vector<double>(N, 0.));

        for (int64_t i = 0; i < N; ++i) {
            PARTONS::DVCSConvolCoeffFunctionKinematic ccfKin(xi[i].item<double>(),
                    t[i].item<double>(), Q2[i].item<double>(),
                    muF2[i].item<double>(), muR2[i].item<double>());

            PARTONS::DVCSConvolCoeffFunctionResult result =
                    scalarCFF.compute(ccfKin, gpdTypes);

            // Read through the map rather than getResult(), so a CFF the model
            // does not provide stays zero instead of throwing.
            const std::map<PARTONS::GPDType::Type, std::complex<double> >& values =
                    result.getResultsByGpdType();
            for (int k = 0; k < 4; ++k) {
                std::map<PARTONS::GPDType::Type, std::complex<double> >::const_iterator it =
                        values.find(types[k]);
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

        DVCSConvolCoeffFunctionModuleTorch::AllCFFsTensorBatch cffs;
        cffs.H  = cff[0];
        cffs.E  = cff[1];
        cffs.Ht = cff[2];
        cffs.Et = cff[3];
        return cffs;
    }

    /// CFFs of the current batch, set by prepareTensorBatch() before
    /// setupKinematicsTorchBatch() runs -- twin of PARTONS'
    /// DVCSProcessModule::m_dvcsConvolCoeffFunctionResult.
    DVCSConvolCoeffFunctionModuleTorch::AllCFFsTensorBatch m_cffsBatch;

    /// Set by prepareTensorBatch(); gates the assemble-only batched overloads.
    bool m_preparedBatch = false;
};

#endif /* DVCS_PROCESS_MODULE_TORCH_H */
