import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("TASK 5: LEARNING RATE SWEEP (Extended)")
print("=" * 60)

widths = [256, 512, 1024]
lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0]
results = {}

torch.manual_seed(42)
x = torch.randn(64, 16)
y = torch.randint(0, 2, (64,))

for width in widths:
    results[width] = []
    for lr in lrs:
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, width)
                self.fc2 = nn.Linear(width, 2)
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))
        
        model = Model()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
        losses = []
        for step in range(100):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        
        final_loss = np.mean(losses[-10:])
        results[width].append(final_loss)
        print(f"Width {width}, LR {lr:.0e}: Loss = {final_loss:.6f}")

# --- Plot ---
plt.figure(figsize=(10, 6))
for width in widths:
    plt.plot(lrs, results[width], marker='o', label=f'Width {width}', linewidth=2)
    min_idx = np.argmin(results[width])
    plt.scatter(lrs[min_idx], results[width][min_idx], s=200,
                facecolors='none', edgecolors='red', linewidth=2, zorder=5)

plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Sweep by Width (Extended)')
plt.legend()
plt.grid(True)
plt.savefig('plots/task5_lr_sweep.png')
print("\n✅ Plot saved: plots/task5_lr_sweep.png")

print("\n--- MINIMA ---")
for width in widths:
    min_idx = np.argmin(results[width])
    print(f"Width {width}: Best LR = {lrs[min_idx]:.0e}, Loss = {results[width][min_idx]:.6f}")

# --- Task 6: Predict LR for width 4096 ---
print("\n" + "=" * 60)
print("TASK 6: PREDICT LR FOR WIDTH 4096")
print("=" * 60)

best_lrs = [lrs[np.argmin(results[w])] for w in widths]
log_widths = np.log(widths)
log_lrs = np.log(best_lrs)

b, log_a = np.polyfit(log_widths, log_lrs, 1)
a = np.exp(log_a)
predicted_lr = a * (4096 ** b)

print(f"Power law fit: lr = {a:.4e} * width^{b:.4f}")
print(f"\nPredicted LR for width 4096: {predicted_lr:.2e}")
print(f"\nConfidence: Medium")
print("Reasoning:")
print("  - Only 3 data points for extrapolation")
print("  - All 3 minima are at the edge of the tested range, which means")
print("    the true minimum may be outside the range")
print("  - Power law assumption may not hold at larger widths")
print(f"\nI would use: {predicted_lr:.2e}")
