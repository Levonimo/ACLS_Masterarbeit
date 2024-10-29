import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def correlation_optimized_warping(reference_2D: np.ndarray, target_2D: np.ndarray, slack: int = 14, segments: int = 100) -> tuple[np.ndarray, np.ndarray]:
    if reference.ndim != 1:
        # Sum the reference signal along the second axis
        reference_1D = np.sum(reference, axis=1)

    if target.ndim != 1:
        # Sum the target signal along the second axis
        target_1D = np.sum(target, axis=1)

    ref_length = len(reference_1D)
    tar_length = len(target_1D)
    if ref_length != tar_length:
        raise ValueError("Reference and target signals must have the same length.")
    
    # Define segment lengths
    segment_length = ref_length // segments

    # Initialize the warping path with straight-line mapping
    warp_path = np.linspace(0, tar_length - 1, ref_length).astype(int)

    # Warping function
    def warp_segment(start_ref, end_ref, start_tar, end_tar):
        x_ref = np.arange(start_ref, end_ref + 1)
        x_tar = np.linspace(start_tar, end_tar, end_ref - start_ref + 1).astype(int)
        return x_tar

    # Iterate over segments
    for i in range(segments):
        start_ref = i * segment_length
        end_ref = (i + 1) * segment_length if i < segments - 1 else ref_length - 1

        best_score = -np.inf
        best_segment = np.linspace(0, tar_length - 1, ref_length).astype(int)

        # Try different warping options within the slack range
        for shift in range(-slack, slack + 1):
            start_tar = max(0, start_ref + shift)
            end_tar = min(tar_length - 1, start_ref + segment_length + shift)

            if end_tar - start_tar + 1 < segment_length:
                continue

            warped_segment = warp_segment(start_ref, end_ref, start_tar, end_tar)

            # Calculate the correlation score
            score = np.corrcoef(reference_1D[start_ref:end_ref + 1], target_1D[warped_segment])[0, 1]
            # Calculate the correlation matrix for the 2D data in the segment
            corr_matrix = np.dot(reference_2D[start_ref:end_ref + 1, :], target_2D[warped_segment, :].T)
            
            # Find Index of Maximum Value in the Correlation Matrix for refference and target
            max_index = np.argmax(corr_matrix)
            max_index = np.unravel_index(max_index, corr_matrix.shape)

            if abs(max_index[0] - max_index[1]) < 2:
                score = score + 0.1

            '''
            # Plot the correlation matrix and the reference and target signals in one figure
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))

            # Plot the correlation matrix
            axs[0].imshow(corr_matrix, aspect='auto', cmap='plasma', origin='lower')
            axs[0].set_title('Correlation Matrix')
            axs[0].set_xlabel('Target Index')
            axs[0].set_ylabel('Reference Index')
            fig.text(0, 1.05, f'Max Index: {max_index}', transform=axs[0].transAxes, ha='center')
            fig.text(0.1, 1.05, f'Score: {np.round(score,6)}', transform=axs[1].transAxes, ha='center')
            # Add two diogonal line forming an x
            axs[0].plot(np.arange(segment_length), np.arange(segment_length), 'w--')
            axs[0].plot(np.arange(segment_length), segment_length - np.arange(segment_length) - 1, 'w--')
            #fig.colorbar(axs[0].images[0], ax=axs[0])

            # Plot the reference and target signals
            axs[1].plot(reference_1D[start_ref:end_ref + 1], label='Reference')
            axs[1].plot(target_1D[warped_segment], label='Target')
            axs[1].plot(np.diff(reference_1D[start_ref:end_ref + 1]), label='Target Diff')
            axs[1].plot(savgol_filter(np.diff(np.diff(reference_1D[start_ref:end_ref + 1]))*100, 5 ,3), label='Target Diff')
            axs[1].set_title('Reference and Target Signals')
            axs[1].set_xlabel('Index')
            axs[1].set_ylabel('Intensity')
            axs[1].legend()
            
            plt.tight_layout()
            plt.show()
            '''

            if score > best_score:
                best_score = score
                best_segment = warped_segment

        warp_path[start_ref:end_ref + 1] = best_segment

    # Warp the 2D target signal
    warped_target = target_2D[warp_path, :]

    # Calulate the correlation score for the hole chromatogram
    hole_score = np.corrcoef(np.sum(reference_2D, axis=1), np.sum(warped_target, axis=1))[0, 1]

    return warped_target, warp_path, hole_score


# Example usage

#PATH = "F:/Documents/MasterArbeit/Data"
#PATH = "D:/OneDrive - ZHAW/Masterarbeit/Data"
PATH = 'C:/Users/wilv/OneDrive - ZHAW (1)/Masterarbeit/Data'

import master_class as mc
Data = mc.DataPreparation(PATH)
data_files = Data.get_file_names()
Chromatograms = Data.get_list_of_chromatograms('Chromatograms', file_list=data_files)
rt = Data.get_retention_time()
# Example usage
# Create example 2D GC-MS data


reference = Chromatograms[data_files[7]]
target = Chromatograms[data_files[8]]

#reference = np.sin(np.linspace(0, 4 * np.pi, 100))[:, None] + np.random.normal(0, 0.1, (100, 50))
#target = np.sin(np.linspace(0, 4 * np.pi, 110))[:, None] + np.random.normal(0, 0.1, (110, 50))



# Perform an optimization for the correlation optimized warping with slack and segments parameters
def objective(x):
    slack, segments = x
    _, _, hole_score = correlation_optimized_warping(reference, target, slack=int(slack), segments=int(segments))
    return hole_score

# set the bounds for the slack and segments parameters and permute the values
bounds = [(10, 30), (50, 200)]
x0 = np.random.randint(6, 20), np.random.randint(50, 150)
'''
iter = 200

for i in range(iter):
    x = np.random.randint(6, 20), np.random.randint(50, 150)
    score = objective(x)
    print(f"Iteration {i + 1}/{iter} - Score: {score} - Parameters: {x}")
    if score > best_score:
        best_score = score
        best_params = x

print(f"Best Score: {best_score}")
print(f"Best Parameters: {best_params}")
'''
best_score = -np.inf
best_params = None
score_matrix = np.zeros((bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0]))
for i in range(bounds[0][0], bounds[0][1]):
    for j in range(bounds[1][0], bounds[1][1]):
        score = objective([i, j])
        score_matrix[i - bounds[0][0], j - bounds[1][0]] = score
        if score > best_score:
            best_score = score
            best_params = [i, j]
        print(f"Score: {score} - Parameters: {i, j} - Best Score: {best_score} - Best Parameters: {best_params}")


# Plot the score matrix
plt.imshow(score_matrix, aspect='auto', cmap='plasma', origin='lower')
plt.axes().set_xticks(np.arange(bounds[1][1] - bounds[1][0]))
plt.axes().set_yticks(np.arange(bounds[0][1] - bounds[0][0]))
plt.colorbar()
plt.xlabel('Segments')
plt.ylabel('Slack')
plt.title('Score Matrix')
plt.show()
#best_params = [19, 68]



warped_target, warp_path, _ = correlation_optimized_warping(reference, target, slack=best_params[0], segments=best_params[1])

# Print the results
print("Warping Path:", warp_path)
print("Warped Target Shape:", warped_target.shape)



# Plot the original and warped data in one figure

# Sum the reference and target signals along the second axis for plotting
reference_sum = np.sum(reference, axis=1)
target_sum = np.sum(target, axis=1)
warped_target_sum = np.sum(warped_target, axis=1)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot the original data
axs[0].plot(rt, reference_sum, label='Reference')
axs[0].plot(rt, target_sum, label='Target')
axs[0].set_xlabel("Retention Time")
axs[0].set_ylabel("Intensity")
axs[0].set_title("Original Data")
axs[0].legend()

# Plot the warped data
axs[1].plot(rt, reference_sum, label='Reference')
axs[1].plot(rt, warped_target_sum, label='Warped Target')
axs[1].set_xlabel("Retention Time")
axs[1].set_ylabel("Intensity")
axs[1].set_title("Warped Data")
axs[1].legend()

plt.tight_layout()
plt.show()