#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

class LinearRegressor
{
private:
    std::vector<float> w;
    std::vector<float> grad;


    const float lambda;
    const float learningRate;
    const float epsilon;
    const std::uint32_t D;

public:
    LinearRegressor(const std::uint32_t size, 
                    const std::uint32_t featureSize,
                    float lam = 0.05, 
                    float lr= 0.01, 
                    float eps = 1e-3);

    ~LinearRegressor() = default;

    void fit(const std::vector<float> &X,
         const std::vector<float> &y,
         const float& learningRate, 
         const uint32_t& nEpochs,
        const uint32_t nSamples);

    void predict(const std::vector<float> X&);
};

LinearRegressor::LinearRegressor(const std::uint32_t size, 
                                const std::uint32_t featureSize,
                                float lam,
                                float lr,
                                float eps)
:   lambda{lam},
    learningRate{lr},
    epsilon{eps},
    D{featureSize},

    //Initialize design matrix and target vector
    w{std::vector<float>(D)}
{
    //Initializes random device
    std::random_device rnd_device;

    //Initializes the engine
    std::mt19937 mersenne_engine {rnd_device()};

    //Initializing distirbution
    std::uniform_real_distribution<float> dist {0, 1};

    //Generator function as a lambda function for std::generate
    auto gen = [&](){
        return dist(mersenne_engine);
    };

    //Generating weights
    std::generate(this->w.begin(), this->w.end(), gen);
}
