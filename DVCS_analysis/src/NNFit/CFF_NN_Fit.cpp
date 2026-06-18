//
// Created by Mariana Khachatryan on 3/25/26.
//

#include "../../include/NNFit/CFF_NN_Fit.h"
#include "../../include/NNFit/CustomLoss.h"
#include "../../include/NNFit/Theory/Modules/CFFs/DVCS/DVCSCFFNNTorch.h"
#include "../../include/NNFit/Theory/Modules/Obs/DVCS/DVCSObservableTorch.h"
#include "../../include/NNFit/Theory/Modules/Obs/DVCS/DVCSAluMinusSin1PhiTorch.h"
#include "../../include/NNFit/Theory/Modules/Processes/DVCS/DVCSProcessBMJ12Torch.h"
#include "../../include/NNFit/Theory/Modules/Services/DVCS/DVCSObservableServiceTorch.h"

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor> CFF_NN_Fitter::load_data_observable() const {

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

    // Columns: xB | t | Q2 | E | phi | <observable> | error
    torch::Tensor X     = torch::zeros({n, 3});
    torch::Tensor E     = torch::zeros({n});
    torch::Tensor phi   = torch::zeros({n});
    torch::Tensor y_obs = torch::zeros({n});
    torch::Tensor sigma = torch::zeros({n});

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j)
            X[i][j] = rows[i][j];
        E[i]     = rows[i][3];
        phi[i]   = rows[i][4];
        y_obs[i] = rows[i][5];          // observable value
        sigma[i] = rows[i].back();      // last column: "error"
    }

    return {X, E, phi, y_obs, sigma};
}

void CFF_NN_Fitter::train_nn() {

    auto [X, E, phi, y_obs, sigma] = load_data_observable();

    int n       = static_cast<int>(X.size(0));
    int n_val   = static_cast<int>(n * m_test_fraction);
    int n_train = n - n_val;

    std::cout << "Loaded " << n << " samples | train: " << n_train
              << " | val: " << n_val << "\n";
    std::cout << "Output layer (" << m_output_layer.size() << "): ";
    for (const auto& s : m_output_layer) std::cout << s << " ";
    std::cout << "\nTraining directly on observable data (chi^2 loss)\n\n";

    // Shuffle indices
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});

    auto idx_train = torch::tensor(std::vector<int>(idx.begin(), idx.begin() + n_train));
    auto idx_val   = torch::tensor(std::vector<int>(idx.begin() + n_train, idx.end()));

    auto split = [](const torch::Tensor& T, const torch::Tensor& sel) {
        return T.index_select(0, sel);
    };
    torch::Tensor X_train   = split(X, idx_train),     X_val   = split(X, idx_val);
    torch::Tensor E_train   = split(E, idx_train),     E_val   = split(E, idx_val);
    torch::Tensor phi_train = split(phi, idx_train),   phi_val = split(phi, idx_val);
    torch::Tensor y_train   = split(y_obs, idx_train), y_val   = split(y_obs, idx_val);
    torch::Tensor s_train   = split(sigma, idx_train), s_val   = split(sigma, idx_val);

    // Per-feature min-max from the training kinematics. Applied INSIDE the NN
    // module (passed to CustomLoss -> setModel below); X stays RAW here because
    // the observable chain builds its kinematics from raw (xB, t, Q2) and the
    // module scales internally — matching observ_calc*.
    m_X_min = std::get<0>(X_train.min(0));
    m_X_max = std::get<0>(X_train.max(0));

    // Build model
    int n_outputs = static_cast<int>(m_output_layer.size());
    CFFNNModel net(n_outputs);
    std::cout << "Model: 3 -> 6 (ReLU) -> " << n_outputs << "\n\n";

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4).weight_decay(1e-3));

    // chi^2 loss on the observable, evaluated through the differentiable *Torch
    // chain. Shares `net` (optimizer updates propagate); scaling matches observ_calc*.
    CustomLoss loss_fn(net, m_output_layer, m_X_min, m_X_max);

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

        // Training step — chi^2 on the observable
        net->train();
        optimizer.zero_grad();
        torch::Tensor loss_train =
                loss_fn(X_train, E_train, phi_train, y_train, s_train);
        loss_train.backward();
        optimizer.step();

        // Validation step
        float val_loss;
        {
            torch::NoGradGuard no_grad;
            val_loss = loss_fn(X_val, E_val, phi_val, y_val, s_val).item<float>();
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
                m_net = net;
                return;
            }
        }
    }

    std::cout << "\nTraining complete | Best val loss: " << best_val_loss << "\n";
    m_net = net;
}

void CFF_NN_Fitter::predict() {

    if (!m_net)
        throw std::runtime_error("Model has not been trained. Call train_nn() first.");

    using namespace PARTONS;

    auto [X, E, phi, y_obs, sigma] = load_data_observable();
    int n = static_cast<int>(X.size(0));

    // Wire the *Torch chain with the trained model — same as observ_calc_torch().
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNTorch::classId);
    static_cast<DVCSCFFNNTorch*>(pDVCSCFF)->setModel(
            m_net, m_output_layer, m_X_min, m_X_max);

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

    DVCSObservableServiceTorch* pServiceTorch =
            static_cast<DVCSObservableServiceTorch*>(
                    Partons::getInstance()->getServiceObjectRegistry()->get(
                            "DVCSObservableServiceTorch"));
    DVCSObservableTorch* pObsTorch = dynamic_cast<DVCSObservableTorch*>(pDVCSObs);

    // Predicted observable per data point (evaluation only — no autograd needed).
    torch::Tensor y_pred = torch::zeros({n}, torch::kFloat64);
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < n; ++i) {
            DVCSObservableKinematic kin(
                    X[i][0].item<double>(), X[i][1].item<double>(),
                    X[i][2].item<double>(), E[i].item<double>(),
                    phi[i].item<double>());
            y_pred[i] = pServiceTorch->computeSingleKinematicTorch(kin, pObsTorch)
                    .item<double>();
        }
    }

    torch::Tensor y_true = y_obs.to(torch::kFloat64);
    torch::Tensor sig    = sigma.to(torch::kFloat64);

    const std::string out_dir = "/Users/marianav/Documents/Research/Analysis/GPD_studies/git/Partons/DVCS_analysis/My_Analysis/Partons_output";

    // Per-point predicted vs measured observable.
    std::ofstream csv(out_dir + "/obs_prediction.csv", std::ios::trunc);
    if (!csv)
        throw std::runtime_error("Cannot open obs_prediction.csv for writing in: " + out_dir);
    csv << "xB,t,Q2,E,phi,obs_true,obs_pred,error\n";
    for (int i = 0; i < n; ++i) {
        csv << X[i][0].item<float>() << "," << X[i][1].item<float>() << ","
            << X[i][2].item<float>() << "," << E[i].item<float>() << ","
            << phi[i].item<float>()  << "," << y_true[i].item<float>() << ","
            << y_pred[i].item<float>() << "," << sig[i].item<float>() << "\n";
    }
    std::cout << "Predictions written to obs_prediction.csv (" << n << " rows)\n";

    // Goodness of fit on the observable: MSE, R^2, and chi^2 (sigma-weighted).
    float mse    = (y_pred - y_true).pow(2).mean().item<float>();
    float y_mean = y_true.mean().item<float>();
    float ss_res = (y_pred - y_true).pow(2).sum().item<float>();
    float ss_tot = (y_true - y_mean).pow(2).sum().item<float>();
    float r2     = (ss_tot > 0.f) ? 1.f - ss_res / ss_tot : 0.f;
    float chi2   = ((y_pred - y_true) / sig).pow(2).sum().item<float>();

    std::ofstream eval(out_dir + "/obs_model_eval.csv", std::ios::trunc);
    if (!eval)
        throw std::runtime_error("Cannot open obs_model_eval.csv for writing in: " + out_dir);
    eval << "observable,mse,r_squared,chi2\n";
    eval << pDVCSObs->getClassName() << "," << mse << "," << r2 << "," << chi2 << "\n";

    std::cout << pDVCSObs->getClassName() << " | MSE: " << mse
              << " | R²: " << r2 << " | chi2: " << chi2 << "\n";
    std::cout << "Model evaluation written to obs_model_eval.csv\n";
}

void CFF_NN_Fitter::observ_calc() {

    using namespace PARTONS;

    if (!m_net)
        throw std::runtime_error("Model has not been trained. Call train_nn() first.");

    // Modules
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNTorch::classId);
    static_cast<DVCSCFFNNTorch*>(pDVCSCFF)->setModel(
            m_net, m_output_layer, m_X_min, m_X_max);

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

    // Modules — the *Torch chain.
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNTorch::classId);
    static_cast<DVCSCFFNNTorch*>(pDVCSCFF)->setModel(
            m_net, m_output_layer, m_X_min, m_X_max);

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

    // Torch-aware service, fetched by name through the standard registry.
    DVCSObservableServiceTorch* pServiceTorch =
            static_cast<DVCSObservableServiceTorch*>(
                    Partons::getInstance()->getServiceObjectRegistry()->get(
                            "DVCSObservableServiceTorch"));    //get returns ServiceObject* that's why we cast down

    // Cross-cast to the base tensor-observable interface (different base
    // subobject than DVCSObservable*, so dynamic_cast). The service drives any
    // DVCS tensor observable polymorphically through it.
    DVCSObservableTorch* pObsTorch =
            dynamic_cast<DVCSObservableTorch*>(pDVCSObs);

    // Kinematics
    DVCSObservableKinematic dvcsKinematics(0.2, -0.2, 2., 5.932, 6.);

    // Tensor path (autograd preserved); detach for printing/comparison.
    torch::Tensor resultTensor =
            pServiceTorch->computeSingleKinematicTorch(dvcsKinematics, pObsTorch);
    double result = resultTensor.item<double>();

    
    std::cout << "Observable (Torch): " << pDVCSObs->getClassName() << "\n";
    std::cout << "Kinematics: xB=0.2, t=-0.2, Q2=2, E=5.932\n";
    std::cout << "DVCSAluMinusSin1Phi (Torch tensor path) = " << result << "\n";
    std::cout << "  requires_grad = "
            << (resultTensor.requires_grad() ? "true" : "false") << "\n";
}

void CFF_NN_Fitter::observ_calc_torch_scalar() {

    using namespace PARTONS;

    if (!m_net)
        throw std::runtime_error("Model has not been trained. Call train_nn() first.");

    // Modules — the *Torch chain, but driven through the SCALAR pipeline.
    DVCSConvolCoeffFunctionModule* pDVCSCFF =
            Partons::getInstance()->getModuleObjectFactory()->newDVCSConvolCoeffFunctionModule(
                    DVCSCFFNNTorch::classId);
    static_cast<DVCSCFFNNTorch*>(pDVCSCFF)->setModel(
            m_net, m_output_layer, m_X_min, m_X_max);

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

    // Standard scalar service: drives the *Torch modules' scalar virtuals
    // (computeObservable / computeCFF), which wrap the tensor methods under
    // NoGradGuard + .item() — so no autograd graph is built.
    DVCSObservableService* pObservableService =
            Partons::getInstance()->getServiceObjectRegistry()->getDVCSObservableService();

    // Kinematics
    DVCSObservableKinematic dvcsKinematics(0.2, -0.2, 2., 5.932, 6.);

    // Evaluate (scalar return, no gradient)
    double result = pObservableService->computeSingleKinematic(
            dvcsKinematics, pDVCSObs).getValue().getValue();

    std::cout << "Observable (Torch, scalar): " << pDVCSObs->getClassName() << "\n";
    std::cout << "Kinematics: xB=0.2, t=-0.2, Q2=2, E=5.932\n";
    std::cout << "DVCSAluMinusSin1Phi (Torch scalar path) = " << result << "\n";
}