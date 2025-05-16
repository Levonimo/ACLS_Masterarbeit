import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def correlation_optimized_warping(reference_2D: np.ndarray, target_2D: np.ndarray, slack: int = 14, segments: int = 100) -> tuple[np.ndarray, np.ndarray]:
    if reference_2D.ndim != 1:
        # Sum the reference signal along the second axis
        reference_1D = np.sum(reference_2D, axis=1)
    else:
        reference_1D = copy(reference_2D)

    if target_2D.ndim != 1:
        # Sum the target signal along the second axis
        target_1D = np.sum(target_2D, axis=1)
    else:
        target_1D = copy(target_2D)


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

            
            # # Plot the correlation matrix and the reference and target signals in one figure
            # fig, axs = plt.subplots(2, 1, figsize=(10, 8))

            # # Plot the correlation matrix
            # axs[0].imshow(corr_matrix, aspect='auto', cmap='plasma', origin='lower')
            # axs[0].set_title('Correlation Matrix')
            # axs[0].set_xlabel('Target Index')
            # axs[0].set_ylabel('Reference Index')
            # fig.text(0, 1.05, f'Max Index: {max_index}', transform=axs[0].transAxes, ha='center')
            # fig.text(0.1, 1.05, f'Score: {np.round(score,6)}', transform=axs[1].transAxes, ha='center')
            # # Add two diogonal line forming an x
            # axs[0].plot(np.arange(segment_length), np.arange(segment_length), 'w--')
            # axs[0].plot(np.arange(segment_length), segment_length - np.arange(segment_length) - 1, 'w--')
            # #fig.colorbar(axs[0].images[0], ax=axs[0])

            # # Plot the reference and target signals
            # axs[1].plot(reference_1D[start_ref:end_ref + 1], label='Reference')
            # axs[1].plot(target_1D[warped_segment], label='Target')
            # axs[1].set_title('Reference and Target Signals')
            # axs[1].set_xlabel('Index')
            # axs[1].set_ylabel('Intensity')
            # axs[1].legend()
            
            # plt.tight_layout()
            # plt.show()
            # # pause process until user closes the plot
            # plt.pause(0.51)

            if score > best_score:
                best_score = score
                best_segment = warped_segment

        warp_path[start_ref:end_ref + 1] = best_segment

    # Warp the 2D target signal
    warped_target = target_2D[warp_path, :]

    # Calulate the correlation score for the hole chromatogram
    hole_score = np.corrcoef(np.sum(reference_2D, axis=1), np.sum(warped_target, axis=1))[0, 1]

    return warped_target, warp_path, hole_score