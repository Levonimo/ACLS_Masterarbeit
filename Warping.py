import numpy as np
#import master_class as mc


def correlation_optimized_warping(reference, target, slack=14, segments=100):
    ref_length = len(reference)
    tar_length = len(target)

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
            score = np.corrcoef(reference[start_ref:end_ref + 1], target[warped_segment])[0, 1]

            if score > best_score:
                best_score = score
                best_segment = warped_segment

        warp_path[start_ref:end_ref + 1] = best_segment

    # Warp the target signal
    warped_target = target[warp_path]

    return warped_target, warp_path


# Example usage
'''
#PATH = "F:/Documents/MasterArbeit/Data"
PATH = "D:/OneDrive - ZHAW/Masterarbeit/Data"
Data = mc.DataPreparation(PATH)
data_files = Data.get_name_mzml_files()
Chromatograms = Data.get_list_of_chromatograms(PATH+'/Chromatograms.npy', source_type = 'FromNPY')
rt = Data.get_retention_time()
# Example usage
# Create example 2D GC-MS data


reference = Chromatograms[data_files[7]]
target = Chromatograms[data_files[8]]
reference = np.sum(reference, axis = 1)
target = np.sum(target, axis = 1)
#reference = np.sin(np.linspace(0, 4 * np.pi, 100))[:, None] + np.random.normal(0, 0.1, (100, 50))
#target = np.sin(np.linspace(0, 4 * np.pi, 110))[:, None] + np.random.normal(0, 0.1, (110, 50))

warped_target, warp_path = correlation_optimized_warping(reference, target)

# Print the results
print("Warping Path:", warp_path)
print("Warped Target Shape:", warped_target.shape)


import matplotlib.pyplot as plt


plt.plot(rt,reference)
plt.plot(rt,target)
plt.xlabel("Retention Time")
plt.ylabel("Intensity")
plt.title("Original Data")
plt.show()


plt.plot(rt,reference)
plt.plot(rt,warped_target)
plt.xlabel("Retention Time")
plt.ylabel("Intensity")
plt.title("Warped Data")
plt.show()


'''