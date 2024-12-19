import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'component')))
from Warping import correlation_optimized_warping

# =========================================================================================================
# Import the data
#warped = np.load('./Outputs/warped_chromatograms.npy', allow_pickle=True).item()
# print(warped)
unwarped = np.load('./Outputs/unwarped_chromatograms.npy', allow_pickle=True).item()
# print(unwarped)
targets = np.load('./Outputs/selected_target.npy', allow_pickle=True)
# print(targets)
rt = np.load('./Outputs/retention_time.npy', allow_pickle=True)
mz = np.arange(35, 401, 1)

# select file to plot
file1 = targets[8]
file2 = targets[3]


# =========================================================================================================
# Plot Chromatogram in 1D and 2D
# Top plot is 2D and bottom plot is 1D
# 1D plot is the sum of the 2D plot

# fig = plt.figure(figsize=(10, 10))
# fig.suptitle(f'Chromatogram {file1}')
# ax1 = fig.add_subplot(211)
# X, Y = np.meshgrid(rt, mz, indexing='ij')
# ax1.pcolormesh(X, Y, np.log10(unwarped[file1]), cmap='inferno')
# #ax1.imshow(np.log10(unwarped[file1]), aspect='auto', cmap='viridis', extent=[rt[0], rt[-1], mz[0], mz[-1]])
# ax1.set_xlabel('Retention Time')
# ax1.set_ylabel('m/z')

# ax2 = fig.add_subplot(212)
# ax2.plot(rt, np.sum(unwarped[file1], axis=1))
# ax2.set_xlim([rt[0], rt[-1]])
# ax2.set_xlabel('Retention Time')
# ax2.set_ylabel('Intensity')

# plt.tight_layout()
# plt.savefig(f'./Outputs/Chromatogram_{file1}.png')
# plt.show()

# =========================================================================================================
# Plot Correlation between two chromatograms

# fig = plt.figure(figsize=(10, 10))
# fig.suptitle(f'Correlation between {file1} and {file2}')
# X, Y = np.meshgrid(rt, rt, indexing='ij')
# ax = fig.add_subplot(111)
# ax.pcolormesh(X, Y, np.log10(np.dot(unwarped[file1], unwarped[file2].T)), cmap='inferno')
# ax.set_xlabel(f'Retention Time {file1}')
# ax.set_ylabel(f'Retention Time {file2}')

# plt.tight_layout()
# plt.savefig(f'./Outputs/Correlation_{file1}_{file2}.png')
# plt.show()


# =========================================================================================================
# Plot Wapring 

warped_target, warp_path, hole_score = correlation_optimized_warping(unwarped[file1], unwarped[file2])

# plot the reference and warped target and the reference and unwarped target
fig = plt.figure(figsize=(10, 8))
fig.suptitle(f'Warping between {file1} and {file2}')

# Sum the reference and target signals along the second axis for plotting
reference_sum = np.sum(unwarped[file1], axis=1)
target_sum = np.sum(unwarped[file2], axis=1)
warped_target_sum = np.sum(warped_target, axis=1)

# Create a figure with two subplots
axs = fig.subplots(2, 1)

# Plot the original data
axs[0].plot(rt, reference_sum, label='Reference')
axs[0].plot(rt, target_sum, label='Target')
axs[0].set_xlabel("Retention Time")
axs[0].set_ylabel("Intensity")
axs[0].legend()

# Plot the warped data
axs[1].plot(rt, reference_sum, label='Reference')
axs[1].plot(rt, warped_target_sum, label='Warped Target')
axs[1].set_xlabel("Retention Time")
axs[1].set_ylabel("Intensity")
axs[1].legend()

plt.tight_layout()
plt.savefig(f'./Outputs/Warping_{file1}_{file2}.png')
plt.show()


# =========================================================================================================
# 