import pandas as pd
import os
from pathlib import Path

def analyze_csv(input_file, filter_terms_or=None, filter_terms_and=None, output_dir=None):
    """
    Liest die CSV-Datei ein, filtert nach Sample Name und 
    findet gemeinsame Verbindungen ohne separate Dateien zu erstellen.
    
    Args:
        input_file (str): Pfad zur Eingabedatei
        filter_terms_or (list): Liste von Begriffen, von denen mindestens einer im Sample Name
                                enthalten sein muss (OR-Verknüpfung)
        filter_terms_and (list): Liste von Begriffen, die alle im Sample Name enthalten sein müssen (AND-Verknüpfung)
        output_dir (str): Verzeichnis für die Ausgabedatei
    
    Returns:
        dict: Dictionary mit gefilterten DataFrames
        pd.DataFrame: DataFrame mit gemeinsamen Verbindungen
        str: Filter-Suffix für Dateinamen
    """
    print(f"Lese Datei ein: {input_file}")
    # CSV-Datei einlesen
    try:
        df = pd.read_csv(input_file, sep=",", encoding="utf-8")
    except UnicodeDecodeError:
        # Versuchen mit einer anderen Kodierung
        df = pd.read_csv(input_file, sep=",", encoding="latin1")
    except Exception as e:
        print(f"Fehler beim Einlesen der Datei: {e}")
        return {}, pd.DataFrame(), ""
    
    # Kontrollieren, ob alle erwarteten Spalten vorhanden sind
    expected_columns = [
        "Component RT", "Compound Name", "Match Factor", 
        "Formula", "Component Area", "Library RI", "Sample Name"
    ]
    
    # Spaltenprüfung und ggf. Bereinigung von Leerzeichen
    actual_columns = df.columns.tolist()
    columns_mapping = {}
    
    for expected in expected_columns:
        matches = [col for col in actual_columns if col.strip() == expected]
        if matches:
            if matches[0] != expected:  # Wenn Leerzeichen im Namen vorhanden sind
                columns_mapping[matches[0]] = expected
        else:
            print(f"Warnung: Spalte '{expected}' nicht gefunden!")
    
    # Spaltennamen bereinigen, falls nötig
    if columns_mapping:
        df = df.rename(columns=columns_mapping)
    
    # Überprüfen, ob Sample Name in den Spalten vorhanden ist
    if "Sample Name" not in df.columns:
        print("Fehler: Die Spalte 'Sample Name' wurde nicht gefunden!")
        return {}, pd.DataFrame(), ""
    
    # Nach Sample Name gruppieren
    sample_names = df["Sample Name"].unique()
    print(f"Gefundene Sample Names: {len(sample_names)}")
    
    # Hilfsfunktion zum Prüfen, ob ein Begriff im Sample Name als Wort enthalten ist
    def term_matches_as_word(term, sample_name):
        sample_name_str = str(sample_name).lower()
        term_lower = term.lower()
        return (term_lower == sample_name_str or                       # Exakte Übereinstimmung
                f"_{term_lower}_" in f"_{sample_name_str}_" or         # Als Teil mit Unterstrich umgeben
                f" {term_lower} " in f" {sample_name_str} " or         # Als Teil mit Leerzeichen umgeben
                sample_name_str.endswith(f"_{term_lower}") or          # Endet mit _TERM
                sample_name_str.endswith(term_lower) or                # Endet mit TERM
                sample_name_str.startswith(f"{term_lower}_"))          # Beginnt mit TERM_
    
    # Erstelle Filter-Suffix für Dateinamen
    filter_suffix = ""
    if filter_terms_or and len(filter_terms_or) > 0:
        filter_suffix += "_OR_" + "_".join(filter_terms_or)
    if filter_terms_and and len(filter_terms_and) > 0:
        filter_suffix += "_AND_" + "_".join(filter_terms_and)
    
    # Filtern der Sample Names für OR-Bedingung
    if filter_terms_or and len(filter_terms_or) > 0:
        filtered_samples = []
        for sample_name in sample_names:
            # Prüfen, ob mindestens einer der Begriffe als exaktes Wort im Sample Name enthalten ist
            match_found = any(term_matches_as_word(term, sample_name) for term in filter_terms_or)
            if match_found:
                filtered_samples.append(sample_name)
        
        sample_names = filtered_samples
        print(f"Proben nach OR-Filterung: {len(sample_names)}")
        if len(sample_names) == 0:
            print("Warnung: Keine Proben entsprechen den OR-Filterkriterien!")
            return {}, pd.DataFrame(), filter_suffix
    
    # Filtern der verbleibenden Sample Names für AND-Bedingung
    if filter_terms_and and len(filter_terms_and) > 0:
        filtered_samples = []
        for sample_name in sample_names:
            # Prüfen, ob ALLE Begriffe als exakte Wörter im Sample Name enthalten sind
            if all(term_matches_as_word(term, sample_name) for term in filter_terms_and):
                filtered_samples.append(sample_name)
        
        sample_names = filtered_samples
        print(f"Proben nach AND-Filterung: {len(sample_names)}")
        if len(sample_names) == 0:
            print("Warnung: Keine Proben entsprechen den AND-Filterkriterien!")
            return {}, pd.DataFrame(), filter_suffix
    
    # Dictionary für gefilterte DataFrames erstellen
    filtered_dfs = {}
    compounds_sets = []
    
    for sample_name in sample_names:
        # Daten für diese Probe extrahieren
        sample_df = df[df["Sample Name"] == sample_name]
        
        # Nach Component RT sortieren
        sample_df_sorted = sample_df.sort_values(by="Component RT")
        
        # Ergebnisse speichern
        filtered_dfs[sample_name] = sample_df_sorted
        
        # Verbindungsnamen für diese Probe extrahieren
        compounds = set(sample_df_sorted["Compound Name"])
        compounds_sets.append(compounds)
        
        print(f"Probe: {sample_name} mit {len(sample_df_sorted)} Einträgen und {len(compounds)} Verbindungen")
    
    # Wenn keine Proben gefunden wurden
    if not filtered_dfs:
        print("Keine Proben wurden gefunden oder gefiltert. Analyse wird beendet.")
        return {}, pd.DataFrame(), filter_suffix
    
    # Gemeinsame Verbindungen finden (Schnittmenge)
    common_compounds = set.intersection(*compounds_sets)
    print(f"\nGemeinsame Verbindungen in allen Proben: {len(common_compounds)}")
    
    if not common_compounds:
        print("Keine gemeinsamen Verbindungen gefunden!")
        return filtered_dfs, pd.DataFrame(), filter_suffix
    
    # Erstelle einen DataFrame mit allen Informationen zu den gemeinsamen Verbindungen
    # von der ersten Probe
    first_sample = list(filtered_dfs.keys())[0]
    result_df = filtered_dfs[first_sample][filtered_dfs[first_sample]["Compound Name"].isin(common_compounds)].copy()
    
    # Füge eine Spalte hinzu, die bestätigt, dass diese Verbindung in allen Proben vorhanden ist
    result_df["In_All_Samples"] = True
    
    return filtered_dfs, result_df, filter_suffix

def save_results(filtered_dfs, common_df, input_file, filter_suffix, save_samples=True, output_dir=None):
    """
    Speichert die Ergebnisse in CSV-Dateien.
    
    Args:
        filtered_dfs (dict): Dictionary mit gefilterten DataFrames
        common_df (DataFrame): DataFrame mit gemeinsamen Verbindungen
        input_file (str): Pfad zur Eingabedatei
        filter_suffix (str): Filter-Suffix für Dateinamen
        save_samples (bool): Ob die einzelnen Probendateien gespeichert werden sollen
        output_dir (str): Verzeichnis für die Ausgabedateien
    
    Returns:
        str: Pfad zur Datei mit gemeinsamen Verbindungen, falls erstellt
    """
    # Wenn kein output_dir spezifiziert ist, im gleichen Ordner wie die Eingabedatei erstellen
    if not output_dir:
        input_dir = os.path.dirname(os.path.abspath(input_file))
        file_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(input_dir, f"{file_basename}_analysierte_proben{filter_suffix}")
    
    
    
    # Speichere gefilterte Proben, wenn gewünscht
    if save_samples:
        # Verzeichnis erstellen, falls es nicht existiert
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Ausgabeverzeichnis: {output_dir}")
        for sample_name, df in filtered_dfs.items():
            # Dateinamen generieren (ungültige Zeichen entfernen)
            safe_name = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in str(sample_name))
            
            # Füge Filter-Suffix zum Dateinamen hinzu
            output_file = os.path.join(output_dir, f"{safe_name}{filter_suffix}.csv")
            
            # Datei speichern
            df.to_csv(output_file, index=False)
            print(f"Datei erstellt: {output_file} mit {len(df)} Einträgen")
    
    
    
    common_output = ""
    if not common_df.empty:
        if save_samples:
            common_output = os.path.join(output_dir, f"gemeinsame_verbindungen{filter_suffix}.csv")
        else:
            common_output = os.path.join(input_dir, f"gemeinsame_verbindungen{filter_suffix}.csv")
        common_df.to_csv(common_output, index=False)
        print(f"\nGemeinsame Verbindungen wurden in {common_output} gespeichert.")
    else:
        print("\nKeine gemeinsamen Verbindungen gefunden, daher wurde keine Ausgabedatei erstellt.")
    
    return common_output

def main():
    """
    Hauptfunktion zum Ausführen des Scripts.
    """
    # Benutzereingabe für die Eingabedatei
    #input_file = input("Bitte geben Sie den Pfad zur CSV-Datei ein: ")
    input_file = "U:/Documents/Masterarbeit/UA_Results/UA_Deconvolution.csv"
    
    if not os.path.exists(input_file):
        print(f"Fehler: Die Datei {input_file} existiert nicht!")
        return
    
    # Frage den Benutzer, ob er die einzelnen Proben-Ergebnisse speichern möchte
    save_option = input("Möchten Sie die einzelnen Proben-Ergebnisse in Dateien speichern? (j/n): ").strip().lower()
    save_samples_flag = save_option in ['j', 'ja', 'y', 'yes', '1', 'true']
    # Die gemeinsamen Verbindungen werden immer gespeichert
    
    # Filter-Begriffe für OR-Verknüpfung abfragen
    filter_input_or = input("Geben Sie Begriffe an, von denen MINDESTENS EINER im Sample Name enthalten sein soll (durch Komma getrennt, leer lassen für alle): ")
    filter_terms_or = [term.strip() for term in filter_input_or.split(",")] if filter_input_or.strip() else None
    
    if filter_terms_or:
        print(f"OR-Filter (mindestens einer muss enthalten sein): {filter_terms_or}")
    
    # Filter-Begriffe für AND-Verknüpfung abfragen
    filter_input_and = input("Geben Sie Begriffe an, die ALLE im Sample Name enthalten sein müssen (durch Komma getrennt, leer lassen für keine AND-Bedingung): ")
    filter_terms_and = [term.strip() for term in filter_input_and.split(",")] if filter_input_and.strip() else None
    
    if filter_terms_and:
        print(f"AND-Filter (alle müssen enthalten sein): {filter_terms_and}")
    
    # CSV-Datei analysieren
    filtered_dfs, common_df, filter_suffix = analyze_csv(input_file, filter_terms_or, filter_terms_and)
    
    if not filtered_dfs:
        print("Keine Daten wurden gefiltert. Script wird beendet.")
        return
    
    # Ergebnisse speichern - gemeinsame Verbindungen immer, einzelne Proben optional
    common_output_path = save_results(
        filtered_dfs, 
        common_df, 
        input_file, 
        filter_suffix, 
        save_samples=save_samples_flag
    )
    
    # Zusammenfassung anzeigen
    if not save_samples_flag:
        print("\nEinzelne Proben-Ergebnisse wurden nicht gespeichert.")
    
    if common_df.empty:
        print("\nKeine gemeinsamen Verbindungen gefunden.")
    else:
        print(f"\nGemeinsame Verbindungen: {len(common_df)} Einträge")
        
    print("\nVerarbeitung abgeschlossen!")

if __name__ == "__main__":
    main()