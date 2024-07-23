import master_class as mc
import numpy as np
import matplotlib.pyplot as plt


Data = mc.Data_Preparation("F:/Documents/MasterArbeit/Data")
data_files = Data.get_name_mzml_files()
Chromatograms = Data.get_list_of_chromatograms('F:/Documents/MasterArbeit/Data/Chromatograms.npy', Type = 'FromNPY')
rt = Data.get_retention_time()
# Example usage
# Create example 2D GC-MS data


reference = Chromatograms[data_files[7]]
target = Chromatograms[data_files[8]]



def correlation_optimized_warping(reference, target, segment_length=10, slack=1):

    # calculate similarity between the reference and target signal by using dot product
    # of each normalized spectra of the reference and target chromatogramm

    # Normalize the reference and target signals spectra-wise
    norm_reference = reference.T - np.mean(reference, axis=1)
    norm_reference = norm_reference.T / np.linalg.norm(norm_reference, axis=1)
    norm_target = target.T - np.mean(target, axis=1)
    norm_target = norm_target.T / np.linalg.norm(norm_target, axis=1)

    print(norm_reference.shape)
    print(norm_target.shape)

    # Calculate the dot product between the reference and target signals
    similarity = np.dot(norm_reference, norm_target.T)

    print(similarity.shape)

    reference = np.sum(reference, axis=1)
    target = np.sum(target, axis=1)


    ref_length = len(reference)
    tar_length = len(target)

    # Number of segments
    num_segments = ref_length // segment_length
    if ref_length % segment_length != 0:
        num_segments += 1

    warp_path = []

    for seg in range(num_segments):
        start_ref = seg * segment_length
        end_ref = min((seg + 1) * segment_length, ref_length)

        start_tar = max(0, start_ref - slack)
        end_tar = min(tar_length, end_ref + slack)

        ref_segment = reference[start_ref:end_ref]
        tar_segment = target[start_tar:end_tar]

        # Initialize cost matrix for the current segment
        segment_cost_matrix = np.full((len(ref_segment) + 1, len(tar_segment) + 1), np.inf)
        segment_cost_matrix[0, 0] = 0

        # Initialize path matrix for the current segment
        segment_path_matrix = np.zeros((len(ref_segment) + 1, len(tar_segment) + 1), dtype=int)

        # Compute cost matrix using dynamic programming within the segment
        for i in range(1, len(ref_segment) + 1):
            for j in range(1, len(tar_segment) + 1):
                cost = np.abs(ref_segment[i - 1] - tar_segment[j - 1])
                min_cost = min(segment_cost_matrix[i - 1, j], segment_cost_matrix[i, j - 1],
                               segment_cost_matrix[i - 1, j - 1])
                segment_cost_matrix[i, j] = cost + min_cost

                if min_cost == segment_cost_matrix[i - 1, j]:
                    segment_path_matrix[i, j] = 1  # From above
                elif min_cost == segment_cost_matrix[i, j - 1]:
                    segment_path_matrix[i, j] = 2  # From left
                else:
                    segment_path_matrix[i, j] = 3  # From diagonal

        # Backtrack to find the optimal warping path within the segment
        segment_warp_path = []
        i, j = len(ref_segment), len(tar_segment)
        while i > 0 or j > 0:
            segment_warp_path.append((start_ref + i - 1, start_tar + j - 1))
            if segment_path_matrix[i, j] == 1:
                i -= 1
            elif segment_path_matrix[i, j] == 2:
                j -= 1
            else:
                i -= 1
                j -= 1
        segment_warp_path.reverse()

        warp_path.extend(segment_warp_path)

    warp_path = np.array(warp_path)
    warped_target = target[warp_path[:, 1]]

    # Plot the warping path plus the similarity matrix as a heatmap
    plt.figure(figsize=(10, 6))

    #plt.imshow(similarity[3000:3700,3000:3700], aspect='auto', cmap='plasma', origin='lower', extent=(2300,2900, 2300,2900))
    #plt.plot(warp_path[3000:3700, 0], warp_path[3000:3700, 1], marker='o', markersize=

    plt.imshow(np.log10(similarity), aspect='auto', cmap='plasma', origin='lower', extent=(0, ref_length, 0, tar_length))
    plt.plot(warp_path[:, 0], warp_path[:, 1], color='black')
    plt.title('Optimized Warping Path')
    plt.xlabel('Index of Reference Signal')
    plt.ylabel('Index of Target Signal')
    plt.grid(True)
    plt.show()

    return warped_target, warp_path





warped_target, warp_path = correlation_optimized_warping(reference, target)

# Print the results
print("Warping Path:", warp_path)
print("Warped Target:", warped_target)

reference = np.sum(reference, axis=1)
target = np.sum(target, axis=1)

plt.plot(rt[3200:3500],reference[3200:3500])
plt.plot(rt[3200:3500],target[3200:3500])
plt.xlabel("Retention Time")
plt.ylabel("Intensity")
plt.title("Original Data")
plt.show()


plt.plot(rt[3200:3500],reference[3200:3500])
plt.plot(rt[3200:3500],warped_target[3200:3500])
plt.xlabel("Retention Time")
plt.ylabel("Intensity")
plt.title("Warped Data")
plt.show()