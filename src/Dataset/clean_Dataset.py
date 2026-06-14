import json
import os

fichier_corrompu = r"c:\Users\MSI\Desktop\Fine_Tunning\smart_plate_nutrition_dataset_complete_PRO.jsonl"
fichier_propre = r"c:\Users\MSI\Desktop\Fine_Tunning\smart_plate_nutrition_dataset_clean.jsonl"

mots_interdits = [
    "useRalative", "lndustrial", "processing-factory", "isometric", 
    "handsome-young-man", "turbo-", "Câmplate"
]

lignes_valides = 0
lignes_supprimees = 0

if not os.path.exists(fichier_corrompu):
    print("file does not exist. Please check the path and try again.")
    exit()

with open(fichier_corrompu, "r", encoding="utf-8") as f_in, \
     open(fichier_propre, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        if not line.strip():
            continue
        
        # Verification 
        contient_toxine = any(mot in line for mot in mots_interdits)
        
        if contient_toxine:
            lignes_supprimees += 1
        else:
            f_out.write(line)
            lignes_valides += 1

print("\n🧹 --- REPORT OF CLEANING PROCESS ---")
print(f"✅ Lines kept : {lignes_valides}")
print(f"🗑️ Corrupted lines removed : {lignes_supprimees}")
print(f"💾 New clean file created : {fichier_propre}\n")