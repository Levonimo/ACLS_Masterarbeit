import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Parameters
parameters = dict(
    data_folder = "U:/Documents/Masterarbeit/250311_PositivKontrolle_PinkyMaden",
    retention_time = [],
    tick_interval = 2,
    y_max = "default", #3*10**7,  "default" or a specific value 
    image_height = 3,
    image_width = 8, # 12 cm = textwidth
    dpi = 300,
    font_size = 9,
    font_family = "Arial",
    output_folder = "U:/Documents/Masterarbeit/250311_PositivKontrolle_PinkyMaden",
    output_file_name = "SPME_Chromatogramm_PinkyMaden.png"
)


# Plot all TXT files in  the data folder SEP= ";", with the parameters defined above
def plot_chromatograms(data_folder, retention_time, tick_interval, y_max, image_height, image_width, dpi, font_size, font_family, output_folder, output_file_name):
    plt.figure(figsize=(image_width, image_height), dpi=dpi)
    plt.rcParams.update({'font.size': font_size, 'font.family': font_family})

    for file in os.listdir(data_folder):
        if file.endswith(".TXT"):
            path = os.path.join(data_folder, file)
            data_raw = pd.read_csv(path, sep=";", header=1, comment='#')
            data_raw.columns = ['Index','time', 'intensity']
            
            # Filter data to retention time window
            filtered = data_raw[(data_raw['time'] >= retention_time[0]) & (data_raw['time'] <= retention_time[1])]
            plt.plot(filtered['time'], filtered['intensity'], label=file[:-4], linewidth=0.5)

    plt.xlabel('Retention Time / min')
    plt.ylabel('Intensity')
    plt.xlim(retention_time)
    
    if y_max == "default":
        y_max = filtered['intensity'].max() * 1.1  # 10% margin
    plt.ylim(-100000, y_max)
    
    plt.xticks(np.arange(retention_time[0], retention_time[1] + tick_interval, tick_interval))
    plt.grid(False)
    # remove axis spines top and right
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # remove legend frame
    plt.legend(frameon=False, loc='upper right', fontsize='small')
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, output_file_name)
    plt.savefig(output_path)
    plt.show()

# Call the plotting function with the parameters
plot_chromatograms(
    data_folder=parameters['data_folder'],
    retention_time=[4.5, 28.5],  # Adjusted retention time for the SPME data
    tick_interval=parameters['tick_interval'],
    y_max=parameters['y_max'],
    image_height=parameters['image_height'],
    image_width=parameters['image_width'],
    dpi=parameters['dpi'],
    font_size=parameters['font_size'],
    font_family=parameters['font_family'],
    output_folder=parameters['output_folder'],
    output_file_name=parameters['output_file_name']
)

