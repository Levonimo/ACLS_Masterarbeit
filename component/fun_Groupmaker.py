import numpy as np
from collections import Counter
from collections import defaultdict




# filenames = np.load('./Outputs/selected_target.npy', allow_pickle=True)

# filenames = ['001_A1_1_OOO', '002_A1_2_SOO', '003_A1_3_SGO', '004_A1_4_SOL',
#  '005_A1_5_SGL', '006_A1_6_SGL', '007_A1_0_FFF', '008_A1_1_OOO',
#  '009_A1_2_SOO', '010_A1_3_SGO', '011_A2_4_SOL', '012_A1_5_SGL',
#  '013_A1_6_SGL', '014_A2_0_FFF', '015_A1_1_OOO', '016_A1_2_SOO',
#  '017_A1_3_SGO', '018_A1_4_SOL', '019_A1_5_SGL', '020_A1_6_SGL',
#  '021_A1_0_FFF', '022_A1_1_OOO', '023_B1_2_SOO', '024_B1_3_SGO',
#  '025_A1_4_SOL', '026_B2_5_SGL', '027_B2_6_SGL']




def detect_delimiter(filenames):
    delimiters = Counter()
    for filename in filenames:
        for char in filename:
            if not char.isalnum():
                delimiters[char] += 1
    most_common_delimiter = delimiters.most_common(1)
    return most_common_delimiter[0][0] if most_common_delimiter else None


def GroupMaker(filenames, keyword = 'Parts'):
    delimiter = detect_delimiter(filenames)
    component_values_set = defaultdict(set)
    component_values_list = defaultdict(list)
    for filename in filenames:
        parts = filename.split(delimiter)
        for idx, part in enumerate(parts):
            component_values_set[idx].add(part)
            component_values_list[idx].append(part)
    
    # Exclude groups that have the same amount of objects as the input
    filtered_sets = {idx: values for idx, values in component_values_set.items() 
                     if len(values) != len(filenames)}
    filtered_lists = {idx: values for idx, values in component_values_list.items() 
                      if len(set(values)) != len(filenames)}
    # Create a mapping of filenames to their parts
    filename_parts = {filename: filename.split(delimiter) for filename in filenames}

    
    if keyword is 'Parts':
        return filtered_sets, filename_parts
    elif keyword is 'GroupList':
        return filtered_sets, filtered_lists
    else:
        return filtered_sets, filename_parts


