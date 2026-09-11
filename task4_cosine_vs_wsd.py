import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

print("=" * 60)
print("TASK 4: COSINE vs WSD")
print("=" * 60)

torch.manual_seed(42)
x = torch.randn(64, 16)
y = torch.randint(0, 2, (64,))

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- Cosine Schedule ---
def cosine_schedule(step, max_steps=300, warmup=20, base_lr=0.01):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# --- WSD Schedule ---
def wsd_schedule(step, max_steps=300, warmup=20, stable=200, base_lr=0.01):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    elif step < stable:
        return base_lr
    else:
        progress = (step - stable) / (max_steps - stable)
        return base_lr * (1 - progress)

# --- Train with Cosine ---
model_cosine = TinyModel()
opt_cosine = torch.optim.Adam(model_cosine.parameters())
losses_cosine = []
for step in range(300):
    lr = cosine_schedule(step)
    for pg in opt_cosine.param_groups:
        pg['lr'] = lr
    opt_cosine.zero_grad()
    loss = nn.CrossEntropyLoss()(model_cosine(x), y)
    loss.backward()
    opt_cosine.step()
    losses_cosine.append(loss.item())

# --- Train with WSD ---
model_wsd = TinyModel()
opt_wsd = torch.optim.Adam(model_wsd.parameters())
losses_wsd = []
for step in range(300):
    lr = wsd_schedule(step)
    for pg in opt_wsd.param_groups:
        pg['lr'] = lr
    opt_wsd.zero_grad()
    loss = nn.CrossEntropyLoss()(model_wsd(x), y)
    loss.backward()
    opt_wsd.step()
    losses_wsd.append(loss.item())

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(losses_cosine, label='Cosine', linewidth=2)
plt.plot(losses_wsd, label='WSD', linewidth=2)
plt.axvline(x=200, color='red', linestyle='--', label='Stop at step 200')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Cosine vs WSD')
plt.legend()
plt.grid(True)
plt.savefig('plots/task4_cosine_vs_wsd.png')
print("✅ Plot saved: plots/task4_cosine_vs_wsd.png")

print(f"\n--- RESULTS AT STEP 200 ---")
print(f"Cosine loss at step 200: {losses_cosine[200]:.6f}")
print(f"WSD loss at step 200:    {losses_wsd[200]:.6f}")
print(f"\nI would keep: {'Cosine' if losses_cosine[200] < losses_wsd[200] else 'WSD'}")
