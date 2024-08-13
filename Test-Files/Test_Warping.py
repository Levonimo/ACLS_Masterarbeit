import numpy as np
import Archiv.master_class_alt as mc


def correlation_optimized_warping_2d(reference, target, slack=10, segments=200):
    ref_length = reference.shape[0]
    tar_length = target.shape[0]

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
        # Initialize best_segment to a straight-line mapping as a fallback
        best_segment = np.linspace(start_ref, end_ref, end_ref - start_ref + 1).astype(int)

        # Try different warping options within the slack range
        for shift in range(-slack, slack + 1):
            start_tar = max(0, start_ref + shift)
            end_tar = min(tar_length - 1, end_ref + shift)

            if end_tar - start_tar + 1 < segment_length:
                continue

            warped_segment = warp_segment(start_ref, end_ref, start_tar, end_tar)

            # Calculate the correlation score for each column
            scores = []
            for col in range(reference.shape[1]):
                score = np.corrcoef(reference[start_ref:end_ref + 1, col], target[warped_segment, col])[0, 1]
                scores.append(score)

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_segment = warped_segment

        warp_path[start_ref:end_ref + 1] = best_segment

    # Warp the target signal
    warped_target = target[warp_path, :]

    return warped_target, warp_path


Data = mc.Data_Preparation("F:/Documents/MasterArbeit/Data")
data_files = Data.get_name_mzml_files()
Chromatograms = Data.get_list_of_chromatograms('F:/Documents/MasterArbeit/Data/Chromatograms.npy', Type = 'FromNPY')
rt = Data.get_retention_time()
# Example usage
# Create example 2D GC-MS data


reference = Chromatograms[data_files[7]].T
target = Chromatograms[data_files[8]].T

#reference = np.sin(np.linspace(0, 4 * np.pi, 100))[:, None] + np.random.normal(0, 0.1, (100, 50))
#target = np.sin(np.linspace(0, 4 * np.pi, 110))[:, None] + np.random.normal(0, 0.1, (110, 50))

warped_target, warp_path = correlation_optimized_warping_2d(reference, target)

# Print the results
print("Warping Path:", warp_path)
print("Warped Target Shape:", warped_target.shape)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(rt,np.sum(reference.T, axis = 1))
plt.plot(rt,np.sum(target.T, axis = 1))
plt.xlabel("Retention Time")
plt.ylabel("M/Z")
plt.title("Original Data")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(rt,np.sum(reference.T, axis = 1))
plt.plot(rt,np.sum(warped_target.T, axis = 1))
plt.xlabel("Retention Time")
plt.ylabel("M/Z")
plt.title("Warped Data")
plt.show()