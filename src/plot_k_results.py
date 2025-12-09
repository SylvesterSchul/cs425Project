import matplotlib.pyplot as plt
import numpy as np
import os

results = {
    10: 1.017, 20: 0.984, 30: 0.963, 40: 0.949, 50: 0.939,
    60: 0.933, 70: 0.927, 80: 0.922, 90: 0.918, 100: 0.915,
    110: 0.912, 120: 0.910, 130: 0.908, 140: 0.906, 150: 0.905, 160: 0.903
}

k_values = sorted(results.keys())
rmse_values = [results[k] for k in k_values]
selected_k = 120

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(k_values, rmse_values, 'b-o', linewidth=2.5, markersize=10, label='RMSE')
ax.set_xlabel('k (Number of Neighbors)', fontsize=14, fontweight='bold')
ax.set_ylabel('RMSE (Root Mean Squared Error)', fontsize=14, fontweight='bold')
ax.set_title('K-Value Optimization: User-Based Collaborative Filtering\nMovieLens 1M Dataset', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(k_values)

for k, rmse in results.items():
    ax.annotate(f'{rmse:.3f}', xy=(k, rmse), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.legend(fontsize=12, loc='upper right')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/complete_k_value_analysis.png', dpi=300, bbox_inches='tight')

print("="*80)
print("K-VALUE ANALYSIS")
print("="*80)
print(f"\n{'k':>5} | {'RMSE':>8} | {'Improvement':>12}")
print("-" * 35)

prev_rmse = None
for k in k_values:
    rmse = results[k]
    improvement = "-" if prev_rmse is None else f"-{prev_rmse - rmse:.3f}"
    marker = " ← SELECTED" if k == selected_k else ""
    print(f"{k:>5} | {rmse:>8.3f} | {improvement:>12}{marker}")
    prev_rmse = rmse

print("\n" + "="*80)
print(f"Selected k: {selected_k} (RMSE={results[selected_k]:.3f})")
print(f"Total improvement from k=10: {results[10] - results[selected_k]:.3f}")
print("="*80)
print("\n✓ Saved graph: results/complete_k_value_analysis.png\n✓ Analysis complete!")
plt.show()