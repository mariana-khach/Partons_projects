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

// Result of one fit attempt (central or replica): the trained net plus the
// per-feature min-max scaling fit on that attempt's own training split.
struct TrainedModel {
    CFFNNModel    net{nullptr};
    torch::Tensor X_min, X_max;
    float         best_val_loss = -1.f;  // reduced chi^2 (chi^2/n_val) of the snapshot
};

class CFF_NN_Fitter {

public:
    explicit CFF_NN_Fitter(
        const std::string& data_path,
        float test_fraction = 0.3f,
        const std::vector<std::string>& output_layer = {
            "ImH", "ReH", "ImE", "ReE", "ImHt", "ReHt", "ImEt", "ReEt"},
        double x_pow = 0.0);

    void train_nn();
    void predict();
    void observ_calc();
    void observ_calc_torch();
    void observ_calc_torch_scalar();

    // Train n_replicas independent fits to Monte-Carlo-smeared pseudodata
    // (y_smeared = y_obs + N(0, sigma), same formula/independence as Gepard's
    // datasets_replica_vectloss: fresh smear + fresh train/val split + fresh
    // weights + fresh optimizer per replica). A replica whose validation loss
    // (reduced chi^2 = chi^2/n_val) is NaN/Inf, or still above hopeless_val_loss
    // at any epoch multiple of hopeless_check_epoch, is discarded and fully
    // redrawn (not just weight-reinit), up to max_retries_per_replica times;
    // the last attempt is kept with a warning if still hopeless after that
    // many retries. Populates m_replicas.
    //
    // hopeless_val_loss is a reduced-chi^2 (chi^2/n_val) threshold, re-checked
    // every hopeless_check_epoch epochs (periodic, not a single checkpoint —
    // matches Gepard's train_net_vectloss, which re-checks its hopeless
    // condition every validation batch rather than once).
    //
    // base_seed: 0 (default) = each attempt draws its smear/split/init from
    // std::random_device (true independence, the normal production mode).
    // Nonzero = deterministic: attempt (r, attempt) uses seed
    // base_seed + r*100 + attempt, so re-running with the same base_seed
    // reproduces byte-identical smear/split/initial-weights per replica — used
    // for controlled A/B comparisons (e.g. normalize_loss on vs off) where the
    // two runs must start from identical conditions and differ only in what's
    // being compared.
    //
    // normalize_loss: forwarded to CustomLoss (true = chi^2/n, the default;
    // false = raw chi^2 sum, kept only for the A/B comparison above).
    void train_replicas(int n_replicas = 10, int max_retries_per_replica = 5,
            float hopeless_val_loss = 100.f, int hopeless_check_epoch = 200,
            unsigned base_seed = 0, bool normalize_loss = true);

    // Export each trained replica (fc1/fc2 weights+biases, min-max scaling,
    // output labels) as <name_prefix><NN>.json in out_dir, same format as
    // export_model_json()'s cff_model.json for the central fit. name_prefix
    // defaults to "cff_model_replica_"; override for a distinct export set
    // (e.g. "cff_model_replica_origloss_" for the normalize_loss=false A/B run).
    void export_replicas(const std::string& out_dir,
            const std::string& name_prefix = "cff_model_replica_") const;

private:
    std::string m_data_path;
    float m_test_fraction;
    std::vector<std::string> m_output_layer;
    double m_xPow;  // CFF = xB^m_xPow * NNet_output (set once, shared by train/predict/eval)
    CFFNNModel m_net{nullptr};
    torch::Tensor m_X_min, m_X_max;  // per-feature min/max from training set
    float m_best_val_loss = -1.f;    // reduced val chi2 (chi2/n_val) of the snapshot stored in m_net
    std::vector<TrainedModel> m_replicas;

    // Load observable-format CSV: xB|t|Q2|E|phi|<observable>|error.
    // Returns {X[N,3]=(xB,t,Q2), E[N], phi[N], y_obs[N]=col 5, sigma[N]=last col}.
    // Used for training directly on observable data via CustomLoss.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
            torch::Tensor> load_data_observable() const;

    // One fully independent fit attempt: if smear, draws
    // y_used = y_obs + N(0, sigma) before the train/val split; otherwise
    // fits the raw y_obs (the central-fit case). Everything else (shuffle,
    // split, per-feature min-max on the training split, fresh net/optimizer,
    // chi^2 early-stopping loop, best-val snapshot) is today's train_nn()
    // logic, operating on locals only (no member state touched). Aborts
    // early (hopeless=true in the result) on a NaN/Inf validation loss, or if
    // val_loss is still above hopeless_val_loss at any epoch multiple of
    // hopeless_check_epoch (periodic check; hopeless_check_epoch<=0 disables
    // it, used by train_nn()'s central fit). learning_curve_path may be "" to
    // skip writing a CSV. seed fixes torch::manual_seed() (weight init +
    // MC smear draw) and the train/val shuffle RNG, for reproducibility.
    // normalize_loss is forwarded to CustomLoss (see train_replicas()).
    struct FitOutcome { TrainedModel model; bool hopeless; };
    FitOutcome fit_once(const torch::Tensor& X, const torch::Tensor& E,
            const torch::Tensor& phi, const torch::Tensor& y_obs,
            const torch::Tensor& sigma, bool smear,
            const std::string& learning_curve_path,
            float hopeless_val_loss, int hopeless_check_epoch,
            unsigned seed, bool normalize_loss = true) const;

    // Export the trained NN (fc1/fc2 weights+biases, min-max scaling, output
    // labels) as JSON, so the exact forward can be reproduced out-of-process
    // (e.g. CFF scans/plots in Python). Written by predict() to cff_model.json.
    void export_model_json(const std::string& path) const;

    // Same export, for an explicit model rather than the stored m_net/m_X_min/
    // m_X_max/m_best_val_loss (used by export_model_json() above for the
    // central fit, and by export_replicas() for each trained replica).
    void export_model_json(const std::string& path, const CFFNNModel& net,
            const torch::Tensor& xMin, const torch::Tensor& xMax,
            float bestValLoss) const;
};
