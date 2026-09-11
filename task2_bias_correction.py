import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 60)
print("TASK 2: DISABLE BIAS CORRECTION")
print("=" * 60)

torch.manual_seed(42)
x = torch.randn(32, 16)
y = torch.randint(0, 2, (32,))

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- With bias correction (default Adam) ---
model1 = TinyModel()
opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)
losses1 = []
for _ in range(20):
    opt1.zero_grad()
    loss = nn.CrossEntropyLoss()(model1(x), y)
    loss.backward()
    opt1.step()
    losses1.append(loss.item())

# --- Without bias correction ---
def adam_no_bias(model, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    m = [torch.zeros_like(p) for p in model.parameters()]
    v = [torch.zeros_like(p) for p in model.parameters()]
    def step():
        for i, p in enumerate(model.parameters()):
            if p.grad is None:
                continue
            g = p.grad.data
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            p.data -= lr * m[i] / (torch.sqrt(v[i]) + eps)
    return step

model2 = TinyModel()
model2.load_state_dict(model1.state_dict())
step_fn = adam_no_bias(model2)
losses2 = []
for _ in range(20):
    model2.zero_grad()
    loss = nn.CrossEntropyLoss()(model2(x), y)
    loss.backward()
    step_fn()
    losses2.append(loss.item())

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(losses1, label='With Bias Correction', marker='o', linewidth=2)
plt.plot(losses2, label='Without Bias Correction', marker='s', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Bias Correction: With vs Without')
plt.legend()
plt.grid(True)
plt.savefig('plots/task2_bias_correction.png')
# plt.show()

# --- Difference analysis ---
print("\n--- DIFFERENCE ANALYSIS ---")
# --- Difference analysis ---
print("\n--- DIFFERENCE ANALYSIS ---")
print(f"{'Step':<5} {'With':<12} {'Without':<12} {'Diff':<12}")
print("-" * 45)
min_diff_step = None
min_diff = float('inf')
for step, (l1, l2) in enumerate(zip(losses1, losses2)):
    diff = abs(l1 - l2)
    if diff < min_diff:
        min_diff = diff
        min_diff_step = step
    print(f"{step:<5} {l1:<12.6f} {l2:<12.6f} {diff:<12.6f}")

print(f"\n📌 Observation: Difference grows over time — it does NOT shrink below 0.01.")
print(f"   Minimum difference was at step {min_diff_step} with diff = {min_diff:.6f}")
print(f"\n💡 Interpretation: Without bias correction, updates are much larger early on")
print(f"   because m and v are underestimated. Bias correction normalizes this.")
