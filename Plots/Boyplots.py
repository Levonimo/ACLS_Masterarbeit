import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Excel-Datei laden (ersetze 'deine_datei.xlsx' durch den Dateinamen)
df = pd.read_excel('U:/Documents/Masterarbeit/Flächenunterschiede EIC.xlsx', sheet_name='PJ_Arrow_EIC57_Alkan')


# Plot erstellen
plt.figure(figsize=(5, 3))
ax = sns.boxplot(x="Was", y="Area (m/z 57)", data=df, palette="Set2")

# Design-Anpassungen
ax.set_ylabel("Area")
ax.set_ylim(bottom=0)

# remove category title
ax.set_xlabel("")

# Entferne obere und rechte Rahmenlinie
sns.despine(top=True, right=True)
# Nur horizontale Gridlines
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.linewidth': 1})




plt.show()
