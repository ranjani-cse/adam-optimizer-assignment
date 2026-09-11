import numpy as np

print("=" * 60)
print("TASK 6: PREDICT LR FOR WIDTH 4096")
print("=" * 60)

# Best LRs from Task 5 (extended sweep)
widths = [256, 512, 1024]
best_lrs = [1e-1, 3e-1, 1e-1]

print("\nInputs from Task 5:")
for w, lr in zip(widths, best_lrs):
    print(f"  Width {w}: Best LR = {lr:.0e}")

# Power law fit: lr = a * width^b
log_widths = np.log(widths)
log_lrs = np.log(best_lrs)

b, log_a = np.polyfit(log_widths, log_lrs, 1)
a = np.exp(log_a)

print(f"\nPower law fit: lr = {a:.4e} * width^{b:.4f}")

# Predict for width 4096
predicted_lr = a * (4096 ** b)
print(f"\nPredicted LR for width 4096: {predicted_lr:.2e}")

# Confidence and reasoning
print(f"\nConfidence: Medium")
print("Reasoning:")
print("  - Only 3 data points for extrapolation")
print("  - All 3 minima cluster in a narrow range (1e-1 to 3e-1)")
print("  - Power law fit gives an exponent of ~0 (flat curve)")
print("  - True minimum may shift at larger widths")
print(f"\nI would use: {predicted_lr:.2e}")
