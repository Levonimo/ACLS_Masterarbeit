import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_compound_presence_matrix(csv_directory, output_file=None):
    """
    Liest alle CSV-Dateien in einem Verzeichnis ein und erstellt eine Matrix, 
    die zeigt, welche Verbindungen in welchen Proben vorkommen.
    
    Args:
        csv_directory (str): Pfad zum Verzeichnis mit den CSV-Dateien
        output_file (str, optional): Pfad für die Ausgabedatei der Konfusionsmatrix
        
    Returns:
        pd.DataFrame: DataFrame mit der Präsenzmatrix (1=vorhanden, 0=nicht vorhanden)
    """
    print(f"Suche CSV-Dateien in: {csv_directory}")
    
    # Alle CSV-Dateien im angegebenen Verzeichnis finden
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    if not csv_files:
        print(f"Keine CSV-Dateien in {csv_directory} gefunden.")
        return None
    
    print(f"Gefundene CSV-Dateien: {len(csv_files)}")
    
    # Dictionary zum Speichern aller gefundenen Verbindungen und in welchen Dateien sie vorkommen
    all_compounds = set()
    file_compounds = {}
    
    # Jede CSV-Datei einlesen und Verbindungen extrahieren
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        print(f"Verarbeite: {file_name}")
        
        try:
            # CSV-Datei einlesen
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
            except UnicodeDecodeError:
                # Versuchen mit einer anderen Kodierung
                df = pd.read_csv(csv_file, encoding="latin1")
            
            # Prüfen, ob die Spalte "Compound Name" existiert
            if "Compound Name" not in df.columns:
                print(f"WARNUNG: Datei {file_name} enthält keine Spalte 'Compound Name'. Überspringe...")
                continue
            
            # Verbindungen extrahieren und zum Set hinzufügen
            compounds = set(df["Compound Name"])
            all_compounds.update(compounds)
            
            # Speichern, welche Verbindungen in dieser Datei vorkommen
            # Speichern, welche Verbindungen in dieser Datei vorkommen –
            # dabei .csv entfernen und die ersten beiden Teile (bei Unterstrich-Split) weglassen
            base = os.path.splitext(file_name)[0]
            parts = base.split('_')[2:]
            file_key = '_'.join(parts)
            file_compounds[file_key] = compounds
            print(f"  Gefundene Verbindungen: {len(compounds)}")
            
        except Exception as e:
            print(f"Fehler beim Einlesen von {file_name}: {e}")
    
    if not all_compounds:
        print("Keine Verbindungen gefunden. Analyse wird beendet.")
        return None
    
    print(f"\nInsgesamt wurden {len(all_compounds)} einzigartige Verbindungen gefunden.")
    
    # DataFrame für die Präsenzmatrix erstellen
    # (Zeilen = Verbindungen, Spalten = Dateien)
    matrix_data = {}
    
    for file_name, compounds in file_compounds.items():
        # Für jede Datei eine Spalte erstellen mit 1 (vorhanden) oder 0 (nicht vorhanden)
        matrix_data[file_name] = [1 if compound in compounds else 0 for compound in all_compounds]

    # transform matrix_data to a DataFrame
    matrix_data = pd.DataFrame(matrix_data, index=list(all_compounds))
    # check get columns index of matrix_data that contains "SOL" or "SGL" in col names
    col_index = matrix_data.columns[matrix_data.columns.str.contains("SOL|SGL")]
    col_index_reverse = matrix_data.columns[~matrix_data.columns.str.contains("SOL|SGL")]
    # check if each line of matrix_data[colindex] contains 1
    # if yes, set all values of this line to 2
    for line in matrix_data.iterrows():
        flag = False
        if line[1][col_index].sum() == len(col_index) and line[1][col_index_reverse].sum() == 0:
            flag = True

        if flag:  
            # set all values of this line with in the col_index to 2
            matrix_data.loc[line[0], col_index] = 1.1      
        

    #  for file_name, compounds in file_compounds.items():
    #     for file_name, compounds in file_compounds.items():
    #         # Wenn der Dateiname "SGL" oder "SOL" enthält, mit 2 markieren, ansonsten mit 1
    #         if "SGL" in file_name or "SOL" in file_name:
    #             matrix_data[file_name] = [2 if compound in compounds else 0 for compound in all_compounds]
    #         else:
    #             matrix_data[file_name] = [1 if compound in compounds else 0 for compound in all_compounds]

    # change value of compounds to 2 if both appear only in samples which contain SGL or SOL
    

    # DataFrame erstellen
    presence_matrix = pd.DataFrame(matrix_data, index=list(all_compounds))

    # Sortiere nach den letzten 3 Zeichen des file_names
    presence_matrix = presence_matrix.reindex(sorted(presence_matrix.columns, key=lambda x: x[-3:]), axis=1)

    # Entferne Stoffe die TMS, silyl, silox, silanol enthalten
    presence_matrix = presence_matrix[~presence_matrix.index.str.contains("TMS|silyl|silox|silanol|Chrom", case=False)]
    # Sortiere nach den Verbindungen
    presence_matrix = presence_matrix.reindex(sorted(presence_matrix.index), axis=0)
    # Konfusionsmatrix speichern, wenn ein Ausgabepfad angegeben wurde
    if output_file:
        presence_matrix.to_csv(output_file)
        print(f"Präsenzmatrix wurde in {output_file} gespeichert.")
    
    return presence_matrix

def visualize_presence_matrix(presence_matrix, output_image=None):
    """
    Erstellt eine Heatmap zur Visualisierung der Verbindungspräsenz.
    
    Args:
        presence_matrix (pd.DataFrame): DataFrame mit der Präsenzmatrix
        output_image (str, optional): Pfad zum Speichern der Grafik
        
    Returns:
        None
    """
    if presence_matrix is None or presence_matrix.empty:
        print("Keine Daten für die Visualisierung verfügbar.")
        return
    
    # filter out compounds which only appear in one sample
    # presence_matrix = presence_matrix.loc[presence_matrix.sum(axis=1) > 1]

    # Größe der Grafik basierend auf der Anzahl der Verbindungen anpassen
    plt.figure(figsize=(12, min(40, max(10, len(presence_matrix) / 5))))

    color = "CMRmap_r"
    max_value = 1.1
    
    # Heatmap erstellen
    ax = sns.heatmap(presence_matrix, cmap=color, cbar=False, vmax=max_value)
    
    # plt.title("Compound appearance in samples")
    plt.xlabel("Samples")
    plt.ylabel("Compounds")
    
    # Y-Achsen-Beschriftungen lesbar machen
    if len(presence_matrix) > 50:
        # Bei zu vielen Verbindungen nur jede n-te Verbindung beschriften
        n = max(1, len(presence_matrix) // 50)
        visible_labels = [i for i in range(len(presence_matrix)) if i % n == 0]
        ax.set_yticks(visible_labels)
        ax.set_yticklabels([presence_matrix.index[i] for i in visible_labels])
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    
    plt.tight_layout()
    
    # Grafik speichern, wenn ein Ausgabepfad angegeben wurde
    if output_image:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"Visualisierung wurde in {output_image} gespeichert.")
    
    plt.show()

def create_compound_summary(presence_matrix):
    """
    Erstellt eine Zusammenfassung, die zeigt, in wie vielen und welchen Proben
    jede Verbindung vorkommt.
    
    Args:
        presence_matrix (pd.DataFrame): DataFrame mit der Präsenzmatrix
        
    Returns:
        pd.DataFrame: DataFrame mit der Zusammenfassung
    """
    if presence_matrix is None or presence_matrix.empty:
        print("Keine Daten für die Zusammenfassung verfügbar.")
        return None
    
    # Zählen, in wie vielen Proben jede Verbindung vorkommt
    presence_count = presence_matrix.sum(axis=1)
    
    # Liste der Proben, in denen jede Verbindung vorkommt
    presence_in = presence_matrix.apply(
        lambda row: ', '.join(presence_matrix.columns[row == 1]), axis=1
    )
    
    # Zusammenfassung erstellen
    summary = pd.DataFrame({
        'Anzahl Proben': presence_count,
        'In Proben': presence_in
    })
    
    # Nach Anzahl der Proben absteigend sortieren
    summary = summary.sort_values('Anzahl Proben', ascending=False)
    
    return summary

def main():
    """
    Hauptfunktion zum Ausführen des Scripts.
    """
    print("CSV-Konfusionsmatrix-Generator")
    print("===========================\n")
    
    # Benutzereingabe für das Verzeichnis mit den CSV-Dateien
    csv_directory = input("Bitte geben Sie den Pfad zum Verzeichnis mit den CSV-Dateien ein: ")
    
    if not os.path.exists(csv_directory) or not os.path.isdir(csv_directory):
        print(f"Fehler: Das Verzeichnis {csv_directory} existiert nicht!")
        return
    
    # Ausgabeverzeichnis erstellen oder nutzen
    output_dir = os.path.join(csv_directory, "konfusionsmatrix_ergebnisse")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Ausgabedateien definieren
    output_csv = os.path.join(output_dir, "verbindungen_praesenzmatrix.csv")
    output_summary = os.path.join(output_dir, "verbindungen_zusammenfassung.csv")
    output_image = os.path.join(output_dir, "verbindungen_heatmap.png")
    
    # Präsenzmatrix erstellen
    presence_matrix = create_compound_presence_matrix(csv_directory, output_csv)
    
    if presence_matrix is None:
        print("Keine Präsenzmatrix erstellt. Script wird beendet.")
        return
    
    # Zusammenfassung erstellen und speichern
    summary = create_compound_summary(presence_matrix)
    if summary is not None:
        summary.to_csv(output_summary)
        print(f"Zusammenfassung wurde in {output_summary} gespeichert.")
        
        # Die ersten 10 Zeilen der Zusammenfassung anzeigen
        print("\nZusammenfassung der häufigsten Verbindungen:")
        print(summary.head(10))
    
    # Visualisierung erstellen
    try:
        visualize_presence_matrix(presence_matrix, output_image)
    except Exception as e:
        print(f"Fehler bei der Erstellung der Visualisierung: {e}")
    
    print("\nVerarbeitung abgeschlossen!")
    print(f"Alle Ergebnisse wurden im Verzeichnis {output_dir} gespeichert.")

if __name__ == "__main__":
    main()