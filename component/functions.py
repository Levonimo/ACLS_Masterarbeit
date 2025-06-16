"""Utility functions for color handling and group file operations."""

import colorsys
import os

def generate_colors(n: int) -> list:
    """Generate a list of visually distinct hex color codes.

    -------
    Parameter:
        n : int --> Number of colors to generate

    Output:
        colors : list --> Generated hex color strings
    """
    colors = []
    for i in range(n):
        hue = i / n             # Hue-Wert zwischen 0 und 1
        lightness = 0.5         # Konstanter Lightness-Wert
        saturation = 0.9        # Konstante Sättigung
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Umwandlung in einen Hex-Code:
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def assign_colors(elements: list) -> dict:
    """Assign a unique hex color to each element.

    -------
    Parameter:
        elements : list --> Elements that require colors

    Output:
        color_mapping : dict --> Mapping of element to color string
    """
    n = len(elements)
    color_list = generate_colors(n)
    color_mapping = {elem: color_list[i] for i, elem in enumerate(elements)}
    return color_mapping

def SaveGroups(groups: dict, filename: str) -> None:
    """Write groups to a meta text file.

    -------
    Parameter:
        groups : dict --> Mapping from group index to filenames
        filename : str --> Directory containing a ``meta`` folder
    """
    with open(os.path.join(filename, 'meta', 'Groups.txt'), 'w') as f:
        for _, value in groups.items():
            for item in value:
                f.write(f'{item} ')
            f.write('\n')

def LoadGroups(filename: str) -> dict:
    """Load group definitions from ``Groups.txt``.

    -------
    Parameter:
        filename : str --> Directory containing ``meta/Groups.txt``

    Output:
        groups : dict --> Mapping of group index to filenames
    """
    groups = {}
    with open(os.path.join(filename, 'meta', 'Groups.txt'), 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            groups[i] = line.split()
    return groups

def LoadSingleGroup(filename: str, group: int) -> list:
    """Return filenames belonging to one group.

    -------
    Parameter:
        filename : str --> Directory containing ``meta/Groups.txt``
        group : int --> Group index to load

    Output:
        entries : list --> Filenames in the group
    """
    with open(os.path.join(filename, 'meta', 'Groups.txt'), 'r') as f:
        lines = f.readlines()
        return lines[group].split()

