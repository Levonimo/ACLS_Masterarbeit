import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component.fun_Baseline import baseline_als as als
from component.fun_Baseline import oals_baseline as oals
from component.fun_Baseline import oals_baseline as arPLS
import matplotlib.pyplot as plt
import seaborn as sns

# import npy file as chromatogram
import numpy as np
import os

chromatogram = np.load('U:/Documents/Masterarbeit/TestSet/Test.npy', allow_pickle=True).item()


chromatogram_als = dict()
chromatogram_oals = dict()
chromatogram_arpls = dict()






for name, chrom in chromatogram.items():
    # Baseline correction for each intensity channel (level)
    corrected = np.zeros_like(chrom)
    for j in range(chrom.shape[1]):
        signal = chrom[:, j]
        base = als(signal, lam=1e6, p=0.001, niter=10)
        corrected[:, j] = signal - base
    chromatogram_als[name] = corrected

    corrected = np.zeros_like(chrom)
    for j in range(chrom.shape[1]):
        signal = chrom[:, j]
        base = oals(signal, lam=1e10, p=5, alpha_factor=3, epsilon=1e-6, maxiter=30)
        corrected[:, j] = signal - base
    chromatogram_oals[name] = corrected

    corrected = np.zeros_like(chrom)
    for j in range(chrom.shape[1]):
        signal = chrom[:, j]
        base = arPLS(signal, lam=2e8, p=0.7)
        corrected[:, j] = signal - base
    chromatogram_arpls[name] = corrected




# als_params = {
#     'lam': [1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8],
#     # 'p': [0.7, 0.9, 1.2, 1.5, 2.0, 3, 5, 7, 10],
#     'p' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
#     'niter': [10, 20, 30, 40, 50]
# }

# for name, chrom in chromatogram.items():
#     plt.figure(figsize=(10, 6))
#     for i in als_params['niter']:
#         corrected = np.zeros_like(chrom)
#         for j in range(chrom.shape[1]):
#             signal = chrom[:, j]
#             base = als(signal, lam=1e6, p=0.001, niter=int(i))
#             corrected[:, j] = signal - base
#         chromatogram_oals[name] = corrected

#         plt.plot(np.sum(chromatogram_oals[name], axis=1), label=f'OALS (niter={i})', linestyle='--')
#     plt.plot(np.sum(chromatogram[name], axis=1), label=f'Original', alpha=0.5)
#     plt.xlabel('Scan Index')    
#     plt.ylabel('Intensity')
#     # remove right and top spines
#     sns.despine(top=True, right=True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     break


# plot the results as overlays for each name

    plt.figure(figsize=(10, 6))
    plt.plot(np.sum(chromatogram_als[name], axis=1), label=f'ALS')
    plt.plot(np.sum(chromatogram_oals[name], axis=1), label=f'OALS')
    plt.plot(np.sum(chromatogram_arpls[name], axis=1), label=f'arPLS')

    plt.plot(np.sum(chromatogram[name], axis=1), label=f'Original', alpha=0.5)
    # plt.title('Baseline Correction Comparison')
    plt.xlabel('Scan Index')    
    plt.ylabel('Intensity')
    # remove right and top spines
    sns.despine(top=True, right=True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    break