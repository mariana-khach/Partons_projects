//
// Created by Mariana Khachatryan on 3/25/26.
//

#include "../../include/NNFit/NN_Fit.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>


NN_Fitter::NN_Fitter(const std::string& data_path,
                     float test_fraction,
                     const std::vector<std::string>& output_layer)
    : m_data_path(data_path),
      m_test_fraction(test_fraction),
      m_output_layer(output_layer) {}

std::pair<torch::Tensor, torch::Tensor> NN_Fitter::load_data() const {

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

    int n          = static_cast<int>(rows.size());
    int n_outputs  = static_cast<int>(m_output_layer.size());

    torch::Tensor X = torch::zeros({n, 3});
    torch::Tensor y = torch::zeros({n, n_outputs});

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j)
            X[i][j] = rows[i][j];
        for (int k = 0; k < n_outputs; ++k)
            y[i][k] = rows[i][5 + k];
    }

    return {X, y};
}

void NN_Fitter::train_nn() {

    auto [X, y] = load_data();

    int n       = static_cast<int>(X.size(0));
    int n_val   = static_cast<int>(n * m_test_fraction);
    int n_train = n - n_val;

    std::cout << "Loaded " << n << " samples | train: " << n_train
              << " | val: " << n_val << "\n";
    std::cout << "Output layer (" << m_output_layer.size() << "): ";
    for (const auto& s : m_output_layer) std::cout << s << " ";
    std::cout << "\n\n";

    // Shuffle indices
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});

    auto idx_train = torch::tensor(std::vector<int>(idx.begin(), idx.begin() + n_train));
    auto idx_val   = torch::tensor(std::vector<int>(idx.begin() + n_train, idx.end()));

    torch::Tensor X_train = X.index_select(0, idx_train);
    torch::Tensor y_train = y.index_select(0, idx_train);
    torch::Tensor X_val   = X.index_select(0, idx_val);
    torch::Tensor y_val   = y.index_select(0, idx_val);

    // Min-max scaling: fit on training set, apply to both splits
    m_X_min = std::get<0>(X_train.min(0));
    m_X_max = std::get<0>(X_train.max(0));
    auto denom  = (m_X_max - m_X_min).clamp_min(1e-8f);
    X_train = (X_train - m_X_min) / denom;
    X_val   = (X_val   - m_X_min) / denom;

    // Build model
    int n_outputs = static_cast<int>(m_output_layer.size());
    NNModel net(n_outputs);
    std::cout << "Model: 3 -> 3 (Tanh) -> " << n_outputs << "\n\n";

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4).weight_decay(1e-3));
    auto loss_fn = torch::nn::MSELoss();

    // Early stopping parameters
    const int patience       = 800;
    const int max_epochs     = 20000;
    float     best_val_loss  = std::numeric_limits<float>::max();
    int       patience_count = 0;

    const std::string out_dir = "/Users/marianav/Documents/Research/Analysis/GPD_studies/git/Partons/DVCS_analysis/My_Analysis/Partons_output";

    {
        std::ofstream csv_init(out_dir + "/learning_curve.csv", std::ios::trunc);
        if (!csv_init)
            throw std::runtime_error("Cannot open learning_curve.csv for writing in: " + out_dir);
        csv_init << "epoch,train_loss,val_loss\n";
    }

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {

        // Training step
        net->train();
        optimizer.zero_grad();
        auto loss_train = loss_fn(net->forward(X_train), y_train);
        loss_train.backward();
        optimizer.step();

        // Validation step
        net->eval();
        float val_loss;
        {
            torch::NoGradGuard no_grad;
            val_loss = loss_fn(net->forward(X_val), y_val).item<float>();
        }

        if (epoch % 2 == 0) {
            std::cout << "Epoch " << std::setw(6) << epoch
                      << " | Train loss: " << std::setw(12) << loss_train.item<float>()
                      << " | Val loss: "   << std::setw(12) << val_loss << "\n";

            std::ofstream csv(out_dir + "/learning_curve.csv", std::ios::app);
            if (!csv)
                throw std::runtime_error("Cannot open learning_curve.csv for appending in: " + out_dir);
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

void NN_Fitter::predict() {

    if (!m_net)
        throw std::runtime_error("Model has not been trained. Call train_nn() first.");

    auto [X, y] = load_data();
    int n        = static_cast<int>(X.size(0));
    int n_outputs = static_cast<int>(m_output_layer.size());

    // Apply the same min-max scaling used during training
    auto denom = (m_X_max - m_X_min).clamp_min(1e-8f);
    X = (X - m_X_min) / denom;

    torch::Tensor y_pred;
    {
        torch::NoGradGuard no_grad;
        m_net->eval();
        y_pred = m_net->forward(X);
    }

    const std::string out_dir = "/Users/marianav/Documents/Research/Analysis/GPD_studies/git/Partons/DVCS_analysis/My_Analysis/Partons_output";
    std::ofstream csv(out_dir + "/obs_prediction.csv", std::ios::trunc);
    if (!csv)
        throw std::runtime_error("Cannot open obs_prediction.csv for writing in: " + out_dir);

    // Header
    for (int k = 0; k < n_outputs; ++k)
        csv << m_output_layer[k] << "_true,";
    for (int k = 0; k < n_outputs; ++k) {
        csv << m_output_layer[k] << "_pred";
        if (k < n_outputs - 1) csv << ",";
    }
    csv << "\n";

    // Rows
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n_outputs; ++k)
            csv << y[i][k].item<float>() << ",";
        for (int k = 0; k < n_outputs; ++k) {
            csv << y_pred[i][k].item<float>();
            if (k < n_outputs - 1) csv << ",";
        }
        csv << "\n";
    }

    std::cout << "Predictions written to obs_prediction.csv (" << n << " rows)\n";

    // Compute R² and MSE per output and write obs_predict_model_eval.csv
    std::ofstream eval(out_dir + "/obs_predict_model_eval.csv", std::ios::trunc);
    if (!eval)
        throw std::runtime_error("Cannot open obs_predict_model_eval.csv for writing in: " + out_dir);

    eval << "output,mse,r_squared\n";
    for (int k = 0; k < n_outputs; ++k) {
        auto  y_k    = y.select(1, k);
        auto  yp_k   = y_pred.select(1, k);
        float mse    = (y_k - yp_k).pow(2).mean().item<float>();
        float y_mean = y_k.mean().item<float>();
        float ss_res = (y_k - yp_k).pow(2).sum().item<float>();
        float ss_tot = (y_k - y_mean).pow(2).sum().item<float>();
        float r2     = (ss_tot > 0.f) ? 1.f - ss_res / ss_tot : 0.f;
        eval << m_output_layer[k] << "," << mse << "," << r2 << "\n";
        std::cout << m_output_layer[k] << " | MSE: " << mse << " | R²: " << r2 << "\n";
    }
    std::cout << "Model evaluation written to obs_predict_model_eval.csv\n";
}