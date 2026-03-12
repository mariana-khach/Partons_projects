#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TStyle.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

void hist_from_csv() {
    const char* filename = "/Users/marianav/Documents/Research/Analysis/GPD_studies/Data/Partons_output/dvcs_DVCSAluSinPhi_ANN_replicas.csv";

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: could not open file:\n" << filename << std::endl;
        return;
    }

    std::vector<double> values;
    std::string line;

    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;

        // Read first column from CSV line
        if (std::getline(ss, cell, ',')) {
            try {
                double val = std::stod(cell);
                values.push_back(val);
            } catch (...) {
                // Skip header or non-numeric lines
                continue;
            }
        }
    }

    infile.close();

    if (values.empty()) {
        std::cerr << "No numeric values found in file." << std::endl;
        return;
    }

    double minVal = *std::min_element(values.begin(), values.end());
    double maxVal = *std::max_element(values.begin(), values.end());

    int nbins = 50;
    TH1D* h = new TH1D("h", "Histogram of values from CSV;Value;Counts",
                       nbins, minVal, maxVal);

    for (double v : values) {
        h->Fill(v);
    }

    TCanvas* c1 = new TCanvas("c1", "CSV Histogram", 800, 600);
    gStyle->SetOptStat(1110);
    h->Rebin(2);
    h->Draw();
    c1->Update();
}
