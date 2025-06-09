import sys, os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component.fun_Baseline import baseline_als as als
from component.fun_Baseline import oals_baseline as oals

# Daten laden
chromatogram = np.load('U:/Documents/Masterarbeit/TestSet/Test.npy', allow_pickle=True).item()
name, chrom = list(chromatogram.items())[0]  # nur ein Beispiel

# Chromatogramm: Form (T, N), T = Scans, N = M/z-Kanäle
# nur die ersten 1000 Scans verwenden, um die Performance zu verbessern
# chrom = chrom[:2000, :]

# Composite Score
def composite_score(signal, baseline, peak_mask):
    corrected = signal - baseline
    l1 = np.sum(np.abs(corrected))
    flatness = np.var(np.diff(baseline, n=2))
    aub = np.sum(np.abs(baseline[~peak_mask]))  # außerhalb Peaks
    return 0.5 * l1 + 0.3 * flatness + 0.2 * aub

# Peak-Maske erstellen: Schwellenwert auf Summensignal
summed = np.sum(chrom, axis=1)
threshold = np.percentile(summed, 80)
peak_mask = summed > threshold  # True: Peak-Bereich

# ALS Grid-Search
als_params = {
    'lam': [ 5e5, 1e6, 5e6, 1e7, 5e7, 1e8],
    'p': [0.005, 0.001, 0.0005, 0.0001]
}
best_als_score = float('inf')
best_als = None

for lam, p in tqdm(product(als_params['lam'], als_params['p']), desc='ALS Grid Search'):
    baseline = np.zeros_like(chrom)
    score = 0
    for j in range(chrom.shape[1]):
        sig = chrom[:, j]
        base = als(sig, lam=lam, p=p, niter=10)
        baseline[:, j] = base
        score += composite_score(sig, base, peak_mask)
    score /= chrom.shape[1]
    if score < best_als_score:
        best_als_score = score
        best_als = (lam, p, baseline)

# OALS Grid-Search
oals_params = {
    'lam': [1e9, 5e9, 1e10, 5e10, 1e11],
    'p': [0.7, 0.9, 1.2, 1.5, 2.0, 3, 5, 7, 10],
    'alpha_factor': [ 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0]
}
best_oals_score = float('inf')
best_oals = None

for lam, p, alpha in tqdm(product(oals_params['lam'], oals_params['p'], oals_params['alpha_factor']), desc='OALS Grid Search'):
    baseline = np.zeros_like(chrom)
    score = 0
    for j in range(chrom.shape[1]):
        sig = chrom[:, j]
        base = oals(sig, lam=lam, p=p, alpha_factor=alpha, epsilon=1e-6, maxiter=50)
        baseline[:, j] = base
        score += composite_score(sig, base, peak_mask)
    score /= chrom.shape[1]
    if score < best_oals_score:
        best_oals_score = score
        best_oals = (lam, p, alpha, baseline)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(np.sum(chrom, axis=1), label='Original', alpha=0.5)
plt.plot(np.sum(chrom - best_als[2], axis=1), label=f'Best ALS (λ={best_als[0]:.0e}, p={best_als[1]})')
plt.plot(np.sum(chrom - best_oals[3], axis=1), label=f'Best OALS (λ={best_oals[0]:.0e}, p={best_oals[1]}, α={best_oals[2]})', linestyle='--')
plt.title(f'Optimierte Baseline-Korrektur – {name}')
plt.xlabel('Scan Index')
plt.ylabel('Summierte Intensität')
plt.legend()
plt.tight_layout()
plt.show()

# Ergebnisse ausgeben
print("Best ALS Params:")
print(f"  λ = {best_als[0]}, p = {best_als[1]}, Composite Score = {best_als_score:.4f}")
print("Best OALS Params:")
print(f"  λ = {best_oals[0]}, p = {best_oals[1]}, α = {best_oals[2]}, Composite Score = {best_oals_score:.4f}")
