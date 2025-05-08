import os, matplotlib.pyplot as plt, numpy as np, seaborn as sns
os.makedirs('figures/deception', exist_ok=True)
sns.set_style('whitegrid')
q = [100, 500, 1000, 2000, 5000, 10000]
u = [0.2, 0.45, 0.6, 0.75, 0.85, 0.9]
d = [0.15, 0.3, 0.4, 0.45, 0.5, 0.52]
b = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(q, u, 'o-', color='firebrick', lw=2, label='Undefended Model')
ax1.plot(q, d, 'o-', color='royalblue', lw=2, label='HoneypotNet Defended')
ax1.set_xlabel('Number of Queries'), ax1.set_ylabel('Attack Success Rate')
ax1.set_ylim(0, 1), ax1.set_xscale('log')
ax2 = ax1.twinx()
ax2.plot(q, b, 'o--', color='forestgreen', lw=2, label='Backdoor Detection Rate')
ax2.set_ylabel('Detection Rate', color='forestgreen')
ax2.tick_params(axis='y', colors='forestgreen'), ax2.set_ylim(0, 1)
ax1.annotate('Protection Gap', xy=(5000, 0.4), xytext=(3000, 0.3),
            arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
ax1.set_title('Model Extraction Attack Success vs. Defense Effectiveness')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
plt.savefig('figures/deception/attack_success_comparison.png', dpi=300, bbox_inches='tight')
print("✅ HoneypotNet defense simulation completed!")