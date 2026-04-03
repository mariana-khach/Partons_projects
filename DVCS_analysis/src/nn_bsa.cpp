#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

// Neural network: 5 inputs -> 6 hidden neurons (ReLU) -> 1 output
struct NetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    NetImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(5, 6));
        fc2 = register_module("fc2", torch::nn::Linear(6, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
};
TORCH_MODULE(Net);

// Load pipe-separated CSV: skip header row
// Returns (features tensor [N,5], labels tensor [N,1])
std::pair<torch::Tensor, torch::Tensor> load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << path << "\n";
        std::exit(1);
    }

    std::string line;
    std::vector<std::vector<float>> rows;

    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<float> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, '|')) {
            row.push_back(std::stof(token));
        }
        if (row.size() >= 6) rows.push_back(row);
    }

    int n = static_cast<int>(rows.size());
    torch::Tensor X = torch::zeros({n, 5});
    torch::Tensor y = torch::zeros({n, 1});

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 5; ++j) {
            X[i][j] = rows[i][j];
        }
        y[i][0] = rows[i][5]; // label (ALU)
    }
    return {X, y};
}

int main() {
    auto [X, y] = load_csv(
        "/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_input/BSA_CLAS_15_KK_format_ALU.csv");

    std::cout << "Loaded " << X.size(0) << " samples, "
              << X.size(1) << " features.\n\n";

    // Build model
    Net net;
    std::cout << "Model: 5 -> 6 (ReLU) -> 1\n\n";

    torch::optim::Adam optimizer(net->parameters(),
                                 torch::optim::AdamOptions(1e-3));
    auto loss_fn = torch::nn::MSELoss();

    for (int epoch = 1; epoch <= 2000; ++epoch) {
        optimizer.zero_grad();
        auto pred = net->forward(X);
        auto loss = loss_fn(pred, y);
        loss.backward();
        optimizer.step();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | Loss: " << loss.item<float>() << "\n";
        }
    }

    std::cout << "\n"
              << std::left
              << std::setw(12) << "Predicted"
              << std::setw(12) << "Label"
              << std::setw(12) << "Residual" << "\n"
              << std::string(36, '-') << "\n";

    auto pred = net->forward(X);
    for (int i = 0; i < X.size(0); ++i) {
        float p = pred[i][0].item<float>();
        float l = y[i][0].item<float>();
        std::cout << std::fixed << std::setprecision(6)
                  << std::setw(12) << p
                  << std::setw(12) << l
                  << std::setw(12) << (p - l) << "\n";
    }

    return 0;
}
