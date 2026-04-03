//
// Created by Mariana Khachatryan on 3/25/26.
//

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

// Neural network: 3 inputs -> 2 hidden neurons (tanh) -> n_outputs
struct NNModelImpl : torch::nn::Module {

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    explicit NNModelImpl(int n_outputs) {
        fc1 = register_module("fc1", torch::nn::Linear(3, 2));
        fc2 = register_module("fc2", torch::nn::Linear(2, n_outputs));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        return fc2->forward(x);
    }
};
TORCH_MODULE(NNModel);

class NN_Fitter {

public:
    explicit NN_Fitter(
        const std::string& data_path,
        float test_fraction = 0.3f,
        const std::vector<std::string>& output_layer = {
            "ImH", "ReH", "ImE", "ReE", "ImHt", "ReHt", "ImEt", "ReEt"});

    void train_nn();
    void predict();

private:
    std::string m_data_path;
    float m_test_fraction;
    std::vector<std::string> m_output_layer;
    mutable NNModel m_net{nullptr};
    torch::Tensor m_X_min, m_X_max;  // per-feature min/max from training set

    // Load pipe-separated CSV; maps header names to feature/label tensors.
    // Features: first 3 columns (xB, t, Q2).
    // Labels:   columns whose header matches names in m_output_layer.
    std::pair<torch::Tensor, torch::Tensor> load_data() const;
};