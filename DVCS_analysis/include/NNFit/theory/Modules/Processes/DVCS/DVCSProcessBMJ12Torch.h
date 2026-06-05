#ifndef DVCS_PROCESS_BMJ12_TORCH_H
#define DVCS_PROCESS_BMJ12_TORCH_H

/**
 * @file DVCSProcessBMJ12Torch.h
 *
 * @brief PARTONS-registered DVCS process module that exposes a
 *        differentiable, torch::Tensor-returning cross-section.
 *
 * Subclasses DVCSProcessBMJ12 so the existing scalar PARTONS pipeline
 * (BH, VCS, Interference cross sections, BMK kinematics, Fourier
 * coefficients) continues to work unchanged through the inherited
 * virtual methods. On top of that, this class adds the new entry
 * point crossSectionAtPhiTensor(phi, beamHelicity), which routes the
 * computation through the Theory::* torch implementations in
 * NNFit/theory/Modules/Processes/DVCS/DVCSAmplitudesBMJ12Torch.h so
 * the autograd graph is preserved from NN weights → CFFs → σ(λ,φ).
 *
 * CFFs are sourced from the attached PARTONS CFF module, which must
 * be a DVCSCFFNNPytorch instance: its computeCFFTensor(gpdType)
 * method is invoked for each of H, E, Ht, Et to pull tensor CFFs
 * without going through the std::complex<double> PARTONS interface.
 */

#include <ElementaryUtils/parameters/Parameters.h>
#include <partons/beans/automation/BaseObjectData.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/process/DVCS/DVCSProcessBMJ12.h>

#include <torch/torch.h>

#include <map>
#include <string>

#include "NNFit/theory/Beans/Obs/DVCS/DVCSKinematicsTorch.h"
#include "NNFit/theory/Modules/Processes/DVCS/DVCSAmplitudesBMJ12Torch.h"

#include <utility>

class DVCSProcessBMJ12Torch : public PARTONS::DVCSProcessBMJ12 {

public:

    static const unsigned int classId; ///< Unique ID for automatic registry.

    /**
     * Constructor.
     * @param className Name of last child class.
     */
    DVCSProcessBMJ12Torch(const std::string& className);

    virtual ~DVCSProcessBMJ12Torch();

    virtual DVCSProcessBMJ12Torch* clone() const;

    virtual void configure(const ElemUtils::Parameters& parameters);
    virtual void resolveObjectDependencies();
    virtual void prepareSubModules(
            const std::map<std::string, PARTONS::BaseObjectData>& subModulesData);

    /**
     * Build the φ-independent BMJ12 Fourier coefficients (VCS² and
     * BH-DVCS interference) by:
     *   1. doing a single NN forward via DVCSCFFNNPytorch::computeAllCFFsTensor(),
     *   2. dressing the raw CFFs with kinematic helicity coefficients
     *      via Theory::computeDressedCFFs,
     *   3. running Theory::computeVCSCoeffs / computeInterfCoeffs.
     *
     * Call once per kinematic point, then reuse the result across all
     * φ and helicity values — this is the entry point that turns the
     * (4 × 20)× redundant NN-forward / dressed-CFF cost into 1×.
     *
     * Prerequisites: kinematics already pushed onto the modules by a
     * preceding setupKinematics() (or scalar compute()); attached CFF
     * module must be a DVCSCFFNNPytorch.
     *
     * @return (VCSCoeffs, InterfCoeffs), both in-graph torch tensors.
     */
    std::pair<Theory::VCSCoeffs, Theory::InterfCoeffs>
        computeFourierCoeffsTensor();

    /**
     * Differentiable counterpart of the scalar BH+VCS+Interference
     * cross-section sequence. Returns the total cross-section
     *
     *     σ(λ, φ) = σ_BH + σ_VCS + σ_Interf
     *
     * at a single azimuthal angle φ as a 0-d torch::Tensor connected
     * to the autograd graph. Now takes pre-computed Fourier coefficients
     * (from computeFourierCoeffsTensor()) so that the φ loop in the
     * upstream observable can call this 20× without redoing the NN
     * forward / dressed-CFF / Fourier-coeff work.
     *
     * @param vcs          VCS² Fourier coefficients (φ-independent).
     * @param interf       BH-DVCS interference coefficients (φ-independent).
     * @param phi          azimuthal angle (rad, Trento convention)
     * @param beamHelicity ±1
     * @return σ(λ,φ) as a 0-d torch::Tensor
     */
    torch::Tensor crossSectionAtPhiTensor(
            const Theory::VCSCoeffs&    vcs,
            const Theory::InterfCoeffs& interf,
            double phi, double beamHelicity);

    /**
     * Push a kinematic point onto the module without running the scalar
     * BMJ12 pipeline. Calls the inherited protected setKinematics() to
     * set (xB, t, Q², E, φ) on the parent's members, then refreshes the
     * cached Theory::DVCSKin used by the tensor path.
     *
     * Intended as a replacement for the scalar pProc->compute(...) call
     * that callers of the tensor path would otherwise need to make
     * purely as a side-effect to initialise kinematic state. Avoids the
     * four NoGrad CFF NN forward passes plus the BH+VCS+Interf scalar
     * arithmetic that compute() would do and discard.
     *
     * @param kinematic (xB, t, Q², E, φ) — φ is stored but not used by
     *                  the tensor path (it is supplied per-call to
     *                  crossSectionAtPhiTensor).
     */
    void setupKinematics(const PARTONS::DVCSObservableKinematic& kinematic);

protected:

    /**
     * Copy constructor.
     * @param other Object to be copied.
     */
    DVCSProcessBMJ12Torch(const DVCSProcessBMJ12Torch& other);

    virtual void initModule();
    virtual void isModuleWellConfigured();

private:

    /**
     * Lazily (re)build the Theory::DVCSKin struct from the parent's
     * current scalar kinematics members (m_xB, m_t, m_Q2, m_E). A
     * cached copy is reused while the kinematics are unchanged so
     * that a φ-scan (e.g. the GL quadrature in the observable) does
     * not pay for the full kinematic setup at every node.
     */
    void buildTorchKinematics();

    Theory::DVCSKin m_torchKin;       ///< Cached φ-independent kinematics.
    bool   m_torchKinInitialized;     ///< Whether m_torchKin is populated.
    double m_lastXB, m_lastT, m_lastQ2, m_lastE; ///< Cached kinematics signature.
};

#endif /* DVCS_PROCESS_BMJ12_TORCH_H */