import torch
import numpy as np

print("=" * 60)
print("TASK 1: ADAM BY HAND")
print("=" * 60)

# Hyperparameters
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
lr = 0.01
w0 = 0.5

# Gradients
gradients = [0.1, -0.05, 0.2, -0.1, 0.15]

# --- Hand computation ---
print("\n--- HAND COMPUTATION ---")
print(f"{'Step':<5} {'g':<10} {'m':<12} {'v':<14} {'m_hat':<12} {'v_hat':<14} {'w':<12}")
print("-" * 80)

m = 0.0
v = 0.0
w = w0

for t, g in enumerate(gradients, start=1):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    print(f"{t:<5} {g:<10.4f} {m:<12.6f} {v:<14.8f} {m_hat:<12.6f} {v_hat:<14.8f} {w:<12.6f}")

hand_result = w
print(f"\nFinal weight (hand): {hand_result:.8f}")

# --- PyTorch verification ---
print("\n--- PYTORCH VERIFICATION ---")
w_torch = torch.tensor([w0], requires_grad=True, dtype=torch.float64)
optimizer = torch.optim.Adam([w_torch], lr=lr, betas=(beta1, beta2), eps=eps)

pytorch_weights = [w0]
for t, g in enumerate(gradients, start=1):
    optimizer.zero_grad()
    w_torch.grad = torch.tensor([g], dtype=torch.float64)
    optimizer.step()
    pytorch_weights.append(w_torch.item())
    print(f"Step {t}: w = {w_torch.item():.8f}")

pytorch_result = w_torch.item()
print(f"\nFinal weight (PyTorch): {pytorch_result:.8f}")
print(f"Difference: {abs(hand_result - pytorch_result):.10f}")

if abs(hand_result - pytorch_result) < 1e-6:
    print("✅ Hand computation matches PyTorch!")
else:
    print("⚠️ Mismatch — check your computation!")
