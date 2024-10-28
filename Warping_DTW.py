import numpy as np


def dynamic_time_warping(reference, target):
    reference_1d = np.sum(reference, axis=1)
    target_1d = np.sum(target, axis=1)

    ref_length = len(reference_1d)
    tar_length = len(target_1d)

    # Initialize the cost matrix
    cost_matrix = np.zeros((ref_length, tar_length))

    # Fill the first row and column
    cost_matrix[0, 0] = abs(reference_1d[0] - target_1d[0])
    for i in range(1, ref_length):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + abs(reference_1d[i] - target_1d[0])

    for j in range(1, tar_length):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + abs(reference_1d[0] - target_1d[j])
    
    # Fill the rest of the matrix
    for i in range(1, ref_length):
        for j in range(1, tar_length):
            cost_matrix[i, j] = abs(reference_1d[i] - target_1d[j]) + min(
                cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]
            )
    
    # Find the optimal path
    i, j = ref_length - 1, tar_length - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev_i, prev_j = min(
                [(i - 1, j), (i, j - 1), (i - 1, j - 1)],
                key=lambda x: cost_matrix[x[0], x[1]]
            )
            i, j = prev_i, prev_j
        path.append((i, j))
    
    path.reverse()

    # Warp the target signal
    warped_target = np.zeros_like(target)
    for i, j in path:
        warped_target[i] = target[j]

    # the cost of the optimal path
    cost = cost_matrix[-1, -1]

    return warped_target, path, cost
    





import master_class as mc

PATH = "F:/Documents/MasterArbeit/Data"
#PATH = "D:/OneDrive - ZHAW/Masterarbeit/Data"
#PATH = 'C:/Users/wilv/OneDrive - ZHAW (1)/Masterarbeit/Data'


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

