#include <torch/torch.h>
#include <GigaLearnCPP/PPO/GAE.h>
#include <cmath>
#include <iostream>

// The same implementation used by PPO minibatches, not a reimplementation.
torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor masks, bool maskEntropy);

int main() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    auto logits = torch::tensor({{0.0f, -20.0f}, {0.0f, 0.0f}}, options);
    auto probs = torch::softmax(logits, -1);
    auto masks = torch::tensor({{1, 0}, {1, 1}}, torch::kInt32);
    auto entropy = ComputeEntropy(probs, masks, true);
    if (!std::isfinite(entropy.item<float>()) || std::abs(entropy.item<float>() - 0.5f) > 1e-5f)
        return 1;
    entropy.backward();
    if (!torch::isfinite(logits.grad()).all().item<bool>()) return 2;
    auto singleton = ComputeEntropy(torch::ones({1, 1}), torch::ones({1, 1}), false);
    if (singleton.item<float>() != 0.0f) return 3;

    auto rews = torch::tensor({2.0f, -1.0f});
    auto terminals = torch::tensor({0, 1}, torch::kInt8);
    auto values = torch::zeros({2});
    torch::Tensor advantages, targets, returns;
    float clipped = -1;
    GGL::GAE::Compute(rews, terminals, values, {}, advantages, targets, returns, clipped,
                      1.f, 1.f, 0.f, 1.f);
    if (std::abs(clipped) > 1e-6f || !torch::allclose(returns, torch::tensor({1.0f, -1.0f}))) return 4;
    GGL::GAE::Compute(rews, terminals, values, {}, advantages, targets, returns, clipped,
                      1.f, 1.f, 1.f, 1.f);
    if (std::abs(clipped - 1.f / 3.f) > 1e-6f) return 5;
    bool rejected = false;
    try {
        GGL::GAE::Compute(rews, torch::zeros({2}, torch::kInt8), values, {}, advantages, targets, returns, clipped);
    } catch (const c10::Error&) { rejected = true; }
    if (!rejected) return 6;
    rejected = false;
    try {
        GGL::GAE::Compute(rews, terminals, torch::zeros({1}), {}, advantages, targets, returns, clipped);
    } catch (const c10::Error&) { rejected = true; }
    if (!rejected) return 7;
    std::cout << "PPO entropy and GAE regressions passed\n";
}
