import pandas as pd
from scipy.stats import mannwhitneyu

# Excel-Datei einlesen, Sheet2 verwenden
df = pd.read_excel('U:/Documents/Masterarbeit/Flächenunterschiede EIC.xlsx', sheet_name='PJ_Arrow')

# Nullwerte entfernen, falls vorhanden
df = df[df["Area (m/z 133)"] > 0]

# Gruppen trennen
larva = df[df["Was"] == "Larva"]["Area (m/z 133)"]
water_blank = df[df["Was"] == "WaterBlank"]["Area (m/z 133)"]

# Mann-Whitney-U-Test (nicht-parametrisch)
stat, p = mannwhitneyu(larva, water_blank, alternative='two-sided')

# Ergebnis ausgeben
print(f"Mann-Whitney-U-Test: Statistik = {stat:.2f}, p-Wert = {p:.4f}")
if p < 0.05:
    print("Ergebnis: Signifikanter Unterschied zwischen den Gruppen.")
else:
    print("Ergebnis: Kein signifikanter Unterschied zwischen den Gruppen.")
