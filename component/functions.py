import colorsys
import os

def generate_colors(n):
    """
    Generiert n verschiedene Farben als Hex-Codes, basierend auf gleichmäßig verteilten Hue-Werten.
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

def assign_colors(elements):
    """
    Weist jedem Element in der Liste 'elements' eine Farbe (als Hex-Code) zu.
    """
    n = len(elements)
    color_list = generate_colors(n)
    color_mapping = {elem: color_list[i] for i, elem in enumerate(elements)}
    return color_mapping

def SaveGroups(groups: dict, filename: str) -> None:
    """
    Speichert die Gruppenzuordnung in einer Textdatei.
    """
    with open(os.path.join(filename, 'meta', 'Groups.txt'), 'w') as f:
        for _, value in groups.items():
            for item in value:
                f.write(f'{item} ')
            f.write('\n')

def LoadGroups(filename: str) -> dict:
    """
    Lädt die Gruppenzuordnung aus einer Textdatei.
    """
    groups = {}
    with open(os.path.join(filename, 'meta', 'Groups.txt'), 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            groups[i] = line.split()
    return groups

def LoadSingleGroup(filename: str, group: int) -> list:
    """
    Lädt eine einzelne Gruppe aus einer Textdatei.
    """
    with open(os.path.join(filename, 'meta', 'Groups.txt'), 'r') as f:
        lines = f.readlines()
        return lines[group].split()

