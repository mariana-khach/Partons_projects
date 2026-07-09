//
// Created by Mariana Khachatryan on 3/25/26.
//

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

// Neural network: 3 inputs -> 6 hidden neurons (Tanh) -> n_outputs
struct CFFNNModelImpl : torch::nn::Module {

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    explicit CFFNNModelImpl(int n_outputs) {
        fc1 = register_module("fc1", torch::nn::Linear(3, 6));
        fc2 = register_module("fc2", torch::nn::Linear(6, n_outputs));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        return fc2->forward(x);
    }
};
TORCH_MODULE(CFFNNModel);

class CFF_NN_Fitter {

public:
    explicit CFF_NN_Fitter(
        const std::string& data_path,
        float test_fraction = 0.3f,
        const std::vector<std::string>& output_layer = {
            "ImH", "ReH", "ImE", "ReE", "ImHt", "ReHt", "ImEt", "ReEt"});

    void train_nn();
    void predict();
    void observ_calc();
    void observ_calc_torch();
    void observ_calc_torch_scalar();

private:
    std::string m_data_path;
    float m_test_fraction;
    std::vector<std::string> m_output_layer;
    mutable CFFNNModel m_net{nullptr};
    torch::Tensor m_X_min, m_X_max;  // per-feature min/max from training set
    float m_best_val_loss = -1.f;    // val chi2 of the snapshot stored in m_net

    // Load observable-format CSV: xB|t|Q2|E|phi|<observable>|error.
    // Returns {X[N,3]=(xB,t,Q2), E[N], phi[N], y_obs[N]=col 5, sigma[N]=last col}.
    // Used for training directly on observable data via CustomLoss.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
            torch::Tensor> load_data_observable() const;

    // Export the trained NN (fc1/fc2 weights+biases, min-max scaling, output
    // labels) as JSON, so the exact forward can be reproduced out-of-process
    // (e.g. CFF scans/plots in Python). Written by predict() to cff_model.json.
    void export_model_json(const std::string& path) const;
};
