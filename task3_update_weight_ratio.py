import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 60)
print("TASK 3: UPDATE-TO-WEIGHT RATIO")
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

model = TinyModel()

# Warmup + Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Warmup scheduler
def warmup_lr(step, warmup_steps=20, base_lr=0.01):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr

ratios = {name: [] for name, _ in model.named_parameters()}

for step in range(100):
    # Apply warmup LR
    lr = warmup_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    
    # Save weights before update
    old_weights = {name: p.data.clone() for name, p in model.named_parameters()}
    
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    # Compute update-to-weight ratio for each layer
    for name, p in model.named_parameters():
        update = p.data - old_weights[name]
        w_norm = old_weights[name].norm().item()
        u_norm = update.norm().item()
        ratio = u_norm / (w_norm + 1e-8)
        ratios[name].append(ratio)

# --- Plot ---
plt.figure(figsize=(12, 6))
for name, r in ratios.items():
    plt.plot(r, label=name, linewidth=2)
plt.axvline(x=20, color='red', linestyle='--', label='Warmup ends (step 20)')
plt.xlabel('Step')
plt.ylabel('Update-to-Weight Ratio')
plt.title('Update-to-Weight Ratio Per Layer')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.savefig('plots/task3_update_weight_ratio.png')
print("\n✅ Plot saved: plots/task3_update_weight_ratio.png")

# --- Find warmup stop point ---
print("\n--- WARMUP ANALYSIS ---")
for name, r in ratios.items():
    # Ratio stabilizes after warmup — look for when it stops changing significantly
    for i in range(5, len(r)):
        if abs(r[i] - r[i-1]) < 0.01 * r[i-1]:
            print(f"{name}: Warmup stops changing ratio at step {i}")
            break
    else:
        print(f"{name}: Ratio continues changing through step {len(r)-1}")

# --- Show first few steps ---
print("\n--- FIRST 25 STEPS ---")
print(f"{'Step':<5} {'fc1.weight':<15} {'fc1.bias':<15} {'fc2.weight':<15} {'fc2.bias':<15}")
print("-" * 65)
for step in range(25):
    row = f"{step:<5}"
    for name in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']:
        row += f" {ratios[name][step]:<15.6f}"
    print(row)
