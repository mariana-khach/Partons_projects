//
// Created by Mariana Khachatryan on 3/25/26.
//

#include "../../include/NNFit/CFF_NN_Fit.h"
#include "NNFit/CustomLoss.h"
#include "NNFit/theory/Modules/CFFs/DVCS/DVCSCFFNNPytorch.h"
#include "NNFit/theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.h"
#include "NNFit/theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.h"

#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/beans/observable/ObservableResult.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinusSin1Phi.h>
#include <partons/modules/process/DVCS/DVCSProcessBMJ12.h>
#include <partons/modules/scales/DVCS/DVCSScalesQ2Multiplier.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterXBToXi.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/ServiceObjectRegistry.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>


CFF_NN_Fitter::CFF_NN_Fitter(const std::string& data_path,
                               float test_fraction,
                               const std::vector<std::string>& output_layer)
    : m_data_path(data_path),
      m_test_fraction(test_fraction),
      m_output_layer(output_layer) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CFF_NN_Fitter::load_data_observable() const {

    std::ifstream file(m_data_path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open data file: " + m_data_path);

    // Skip header row
    std::string line;
    std::getline(file, line);

    // Read rows
    std::vector<std::vector<float>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<float> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, '|'))
            row.push_back(std::stof(token));
        rows.push_back(row);
    }

    int n = static_cast<int>(rows.size());
    if (n == 0)
        throw std::runtime_error("No data rows read from file: " + m_data_path);

    // Expected columns: xB | t | Q2 | E | phi | DVCSAluSinPhi | error
    // (φ is dropped — the observable integrates it out)
    torch::Tensor X     = torch::zeros({n, 3});
    torch::Tensor E     = torch::zeros({n});
    torch::Tensor y_obs = torch::zeros({n});
    torch::Tensor sigma = torch::zeros({n});

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j)
            X[i][j] = rows[i][j];
        E[i]     = rows[i][3];
        y_obs[i] = rows[i][5];
        sigma[i] = rows[i].back();
    }

    return {X, E, y_obs, sigma};
}

void CFF_NN_Fitter::train_nn() {

    // ─── Load observable training data ─────────────────────────────────────
    // Data columns: xB | t | Q² | E | phi | DVCSAluSinPhi | error
    // φ is dropped (integrated out by the observable); y_obs is the
    // observable value; sigma is the uncertainty entering the χ² loss.
    auto [X, E, y_obs, sigma] = load_data_observable();

    int n       = static_cast<int>(X.size(0));
    int n_val   = static_cast<int>(n * m_test_fraction);
    int n_train = n - n_val;

    std::cout << "Loaded " << n << " samples | train: " << n_train
              << " | val: " << n_val << "\n";
    std::cout << "NN output layer (" << m_output_layer.size() << "): ";
    for (const auto& s : m_output_layer) std::cout << s << " ";
    std::cout << "\n\n";

    // Shuffle indices
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});

    auto idx_train = torch::tensor(std::vector<int>(idx.begin(), idx.begin() + n_train));
    auto idx_val   = torch::tensor(std::vector<int>(idx.begin() + n_train, idx.end()));

    torch::Tensor X_train     = X.index_select    (0, idx_train);
    torch::Tensor E_train     = E.index_select    (0, idx_train);
    torch::Tensor y_obs_train = y_obs.index_select(0, idx_train);
    torch::Tensor sigma_train = sigma.index_select(0, idx_train);

    torch::Tensor X_val     = X.index_select    (0, idx_val);
    torch::Tensor E_val     = E.index_select    (0, idx_val);
    torch::Tensor y_obs_val = y_obs.index_select(0, idx_val);
    torch::Tensor sigma_val = sigma.index_select(0, idx_val);

    // No input scaling: the PARTONS DVCSCFFNNPytorch module feeds raw
    // (xB, t, Q²) to the NN. Keeping training inputs raw too ensures the
    // NN sees the same input distribution at train and inference time.

    // ─── Build model ───────────────────────────────────────────────────────
    int n_outputs = static_cast<int>(m_output_layer.size());
    m_net = CFFNNModel(n_outputs);
    std::cout << "Model: 3 -> 6 (Tanh) -> " << n_outputs << "\n\n";

    torch::optim::Adam optimizer(m_net->parameters(),
            torch::optim::AdamOptions(1e-4).weight_decay(1e-3));

    // χ² loss on the observable, evaluated through the PARTONS-tensor
    // module chain. Constructing CustomLoss instantiates the three
    // PARTONS modules and wires them up once; subsequent forward()
    // calls reuse them.
    CustomLoss loss_fn(m_net, m_output_layer);

    // Early stopping parameters
    const int patience       = 200;
    const int max_epochs     = 10000;
    float     best_val_loss  = std::numeric_limits<float>::max();
    int       patience_count = 0;

    const std::string out_dir = "/Users/marianav/Documents/Research/Analysis/GPD_studies/git/Partons/DVCS_analysis/My_Analysis/Partons_output";

    {
        std::ofstream csv_init(out_dir + "/cff_learning_curve.csv", std::ios::trunc);
        if (!csv_init)
            throw std::runtime_error("Cannot open cff_learning_curve.csv for writing in: " + out_dir);
        csv_init << "epoch,train_loss,val_loss\n";
    }

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {

        // Training step
        m_net->train();
        optimizer.zero_grad();
        auto loss_train = loss_fn(X_train, E_train, y_obs_train, sigma_train);
        loss_train.backward();
        optimizer.step();

        // Validation step
        m_net->eval();
        float val_loss;
        {
            torch::NoGradGuard no_grad;
            val_loss = loss_fn(X_val, E_val, y_obs_val, sigma_val).item<float>();
        }

        if (epoch % 2 == 0) {
            std::cout << "Epoch " << std::setw(6) << epoch
                      << " | Train loss: " << std::setw(12) << loss_train.item<float>()
                      << " | Val loss: "   << std::setw(12) << val_loss << "\n";

            std::ofstream csv(out_dir + "/cff_learning_curve.csv", std::ios::app);
            if (!csv)
                throw std::runtime_error("Cannot open cff_learning_curve.csv for appending in: " + out_dir);
            csv << epoch << "," << loss_train.item<float>() << "," << val_loss << "\n";
        }

        // Early stopping check
        if (val_loss < best_val_loss) {
            best_val_loss  = val_loss;
            patience_count = 0;
        } else {
            ++patience_count;
            if (patience_count >= patience) {
                std::cout << "\nEarly stopping at epoch " << epoch
                          << " | Best val loss: " << best_val_loss << "\n";
                return;
            }
        }
    }

    std::cout << "\nTraining complete | Best val loss: " << best_val_loss << "\n";
}

void CFF_NN_Fitter::observ_calc() {

    using namespace PARTONS;

    if (!m_net)
        throw std::runtime_error("Model has not been trained. Call train_nn() first.");

    // Modules
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNPytorch::classId);
    static_cast<DVCSCFFNNPytorch*>(pDVCSCFF)->setModel(m_net, m_output_layer);

    DVCSXiConverterModule* pDVCSXiConverter =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSXiConverterModule(
                    DVCSXiConverterXBToXi::classId);

    DVCSScalesModule* pDVCSScales =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSScalesModule(
                    DVCSScalesQ2Multiplier::classId);

    DVCSProcessModule* pDVCSProcess =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSProcessModule(
                    DVCSProcessBMJ12::classId);

    DVCSObservable* pDVCSObs =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    DVCSAluMinusSin1Phi::classId);

    // pQCD order
    pDVCSCFF->setQCDOrderType(PerturbativeQCDOrderType::LO);

    // Link modules
    pDVCSProcess->setXiConverterModule(pDVCSXiConverter);
    pDVCSProcess->setScaleModule(pDVCSScales);
    pDVCSProcess->setConvolCoeffFunctionModule(pDVCSCFF);
    pDVCSObs->setProcessModule(pDVCSProcess);

    // Service
    DVCSObservableService* pObservableService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();

    // Kinematics
    DVCSObservableKinematic dvcsKinematics(0.2, -0.2, 2., 5.932, 6.);

    // Evaluate
    double result = pObservableService->computeSingleKinematic(
            dvcsKinematics, pDVCSObs).getValue().getValue();

    std::cout << "Observable: " << pDVCSObs->getClassName() << "\n";
    std::cout << "Kinematics: xB=0.2, t=-0.2, Q2=2, E=5.932, phi=6\n";
    std::cout << "DVCSAluMinusSin1Phi = " << result << "\n";
}

void CFF_NN_Fitter::observ_calc_torch() {

    using namespace PARTONS;

    if (!m_net)
        throw std::runtime_error("Model has not been trained. Call train_nn() first.");

    // ─── PARTONS modules (factory) ─────────────────────────────────────────
    // The differentiable pipeline uses the *Torch subclasses of each module
    // so that crossSectionAtPhiTensor / computeTensor stay inside the
    // autograd graph from NN weights to the final asymmetry.
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNPytorch::classId);
    static_cast<DVCSCFFNNPytorch*>(pDVCSCFF)->setModel(m_net, m_output_layer);

    DVCSXiConverterModule* pDVCSXiConverter =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSXiConverterModule(
                    DVCSXiConverterXBToXi::classId);

    DVCSScalesModule* pDVCSScales =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSScalesModule(
                    DVCSScalesQ2Multiplier::classId);

    DVCSProcessModule* pDVCSProcess =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSProcessModule(
                    DVCSProcessBMJ12Torch::classId);

    DVCSObservable* pDVCSObs =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    DVCSAluMinusSin1PhiTorch::classId);

    // pQCD order
    pDVCSCFF->setQCDOrderType(PerturbativeQCDOrderType::LO);

    // Link modules
    pDVCSProcess->setXiConverterModule(pDVCSXiConverter);
    pDVCSProcess->setScaleModule(pDVCSScales);
    pDVCSProcess->setConvolCoeffFunctionModule(pDVCSCFF);
    pDVCSObs->setProcessModule(pDVCSProcess);

    // ─── Kinematics (matches observ_calc()) ────────────────────────────────
    DVCSObservableKinematic dvcsKinematics(0.2, -0.2, 2., 5.932, 6.);

    // ─── Tensor evaluation ─────────────────────────────────────────────────
    // computeTensor() returns a 0-d torch::Tensor connected to the autograd
    // graph; .item<double>() pulls the scalar value out for printing.
    DVCSAluMinusSin1PhiTorch* pTorchObs =
            static_cast<DVCSAluMinusSin1PhiTorch*>(pDVCSObs);
    torch::Tensor result_tensor = pTorchObs->computeTensor(dvcsKinematics);
    const double result = result_tensor.item<double>();

    std::cout << "Observable (torch): " << pDVCSObs->getClassName() << "\n";
    std::cout << "Kinematics: xB=0.2, t=-0.2, Q2=2, E=5.932\n";
    std::cout << "DVCSAluMinusSin1Phi (torch) = " << result << "\n";
}

void CFF_NN_Fitter::observ_calc_torch_via_service() {

    using namespace PARTONS;

    if (!m_net)
        throw std::runtime_error("Model has not been trained. Call train_nn() first.");

    // Same wiring as observ_calc(), but instantiating the *
    // subclasses
    // for the process and observable modules. PARTONS' DVCSObservableService
    // still drives the computation; only the inherited scalar virtuals
    // (CrossSectionBH/VCS/Interf on the process module, MathIntegratorModule
    // φ-integral inside the observable) run. The tensor entry points are not
    // touched. Purpose: verify that the *Torch subclasses remain operable
    // through PARTONS' standard service path.
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNPytorch::classId);
    static_cast<DVCSCFFNNPytorch*>(pDVCSCFF)->setModel(m_net, m_output_layer);

    DVCSXiConverterModule* pDVCSXiConverter =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSXiConverterModule(
                    DVCSXiConverterXBToXi::classId);

    DVCSScalesModule* pDVCSScales =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSScalesModule(
                    DVCSScalesQ2Multiplier::classId);

    DVCSProcessModule* pDVCSProcess =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSProcessModule(
                    DVCSProcessBMJ12Torch::classId);

    DVCSObservable* pDVCSObs =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSObservable(
                    DVCSAluMinusSin1PhiTorch::classId);

    pDVCSCFF->setQCDOrderType(PerturbativeQCDOrderType::LO);

    pDVCSProcess->setXiConverterModule(pDVCSXiConverter);
    pDVCSProcess->setScaleModule(pDVCSScales);
    pDVCSProcess->setConvolCoeffFunctionModule(pDVCSCFF);
    pDVCSObs->setProcessModule(pDVCSProcess);

    DVCSObservableService* pObservableService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();

    DVCSObservableKinematic dvcsKinematics(0.2, -0.2, 2., 5.932, 6.);

    double result = pObservableService->computeSingleKinematic(
            dvcsKinematics, pDVCSObs).getValue().getValue();

    std::cout << "Observable (torch-classes via service): " << pDVCSObs->getClassName() << "\n";
    std::cout << "Kinematics: xB=0.2, t=-0.2, Q2=2, E=5.932, phi=6\n";
    std::cout << "DVCSAluMinusSin1Phi (torch-classes via service) = " << result << "\n";
}