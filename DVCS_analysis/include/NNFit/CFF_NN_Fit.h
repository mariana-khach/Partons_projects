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

private:
    std::string m_data_path;
    float m_test_fraction;
    std::vector<std::string> m_output_layer;
    mutable CFFNNModel m_net{nullptr};
    torch::Tensor m_X_min, m_X_max;  // per-feature min/max from training set

    // Load pipe-separated CSV; returns {X, y, sigma}.
    // Features: first 3 columns (xB, t, Q2).
    // Labels:   columns whose header matches names in m_output_layer.
    // Sigma:    last column titled "error".
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> load_data() const;
};
