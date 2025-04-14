import matplotlib.pyplot as plt
import pandas as pd
import os

# Plotting function
def plot_chromatogram(data, retention_time, font_size, font_family, label, color):
    # Filter data to retention time window
    filtered = data[(data['time'] >= retention_time[0]) & (data['time'] <= retention_time[1])]
    plt.plot(filtered['time'], filtered['intensity'], label=label, color=color)
    return filtered  # Rückgabe für Y-Limit-Berechnung

# Parameters
parameters = dict(
    data_folder = "F:/Dokumente/Masterarbeit/CSV_all_Sample",
    batch_numbers = ["A1"],
    sample_types = ["all"],
    coloring_schema = "Sample",
    retention_time = [9.9, 10.8],
    image_height = 7,
    image_width = 8,
    dpi = 300,
    font_size = 10,
    font_family = "Arial",
    output_folder = "F:/Dokumente/Masterarbeit/Masterarbeit_Latex/Abbildungen/Results/Indirect",
    output_file_name = "Chrom_Auschnit_A1_Benzen_Silicium2.png",
)

color_schema = {
    "Batch": {
        "A1": "blue", "A2": "red", "B1": "green", "C1": "orange", "C2": "black"
    },
    "Sample": {
        "OOO": "black", "SOO": "red", "SOL": "darkred", "SGO": "green", "SGL": "darkgreen", "FFF": "blue",
    }
}

# Plotting setup
plt.figure(figsize=(parameters['image_width'], parameters['image_height']), dpi=parameters['dpi'])

# Sammle Daten
data_files = {}
used_labels = set()
filtered_all = []

for file in os.listdir(parameters["data_folder"]):
    if file.endswith(".TXT"):
        parts = file.rstrip('.TXT').split("_")
        batch, sample = parts[1], parts[3]

        batch_valid = ("all" in parameters["batch_numbers"]) or (batch in parameters["batch_numbers"])
        sample_valid = ("all" in parameters["sample_types"]) or (sample in parameters["sample_types"])

        if batch_valid and sample_valid:
            path = os.path.join(parameters["data_folder"], file)
            data_raw = pd.read_csv(path, sep=";", header=0, comment='#')
            data = pd.DataFrame({
                'time': data_raw.iloc[:, 1],
                'intensity': data_raw.iloc[:, 2]
            })
            label_key = sample if parameters["coloring_schema"] == "Sample" else batch
            color = color_schema[parameters["coloring_schema"]].get(label_key, 'grey')
            label = label_key if label_key not in used_labels else None
            used_labels.add(label_key)

            filtered = plot_chromatogram(data, parameters["retention_time"], parameters["font_size"], parameters["font_family"], label, color)
            filtered_all.append(filtered)

# Berechne globales y-Limit basierend auf allen gefilterten Daten
all_filtered_concat = pd.concat(filtered_all)
y_min = all_filtered_concat['intensity'].min()
y_max = all_filtered_concat['intensity'].max()
y_range = y_max - y_min
if y_range == 0:
    y_range = y_max * 0.1
plt.ylim(y_min - 0.02 * y_range, y_max + 0.05 * y_range)

# Layout und Speichern
plt.xlim(parameters["retention_time"])
plt.xlabel('Time / min', fontsize=parameters["font_size"], fontfamily=parameters["font_family"])
plt.ylabel('Intensity', fontsize=parameters["font_size"], fontfamily=parameters["font_family"])
plt.tick_params(labelsize=parameters["font_size"])
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
os.makedirs(parameters["output_folder"], exist_ok=True)
plt.savefig(os.path.join(parameters["output_folder"], parameters["output_file_name"]))
plt.close()