import json

cheking_file = r"c:\Users\MSI\Desktop\Fine_Tunning\smart_plate_nutrition_dataset_complete_PRO.jsonl"
fr_count = 0
en_count = 0

try:
    with open(cheking_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                system_content = data["messages"][0]["content"]
                if "AI assistant" in system_content or "English" in system_content or "You are the" in system_content:
                    en_count += 1
                else:
                    fr_count += 1
    print("\n📊 --- SCORE DU DATASET ---")
    print(f"🇫🇷 Examples in French : {fr_count}")
    print(f"🇬🇧 Examples in English  : {en_count}")
    print(f"📈 Percentage of English : {round((en_count / (fr_count + en_count)) * 100, 2)}%\n")
except FileNotFoundError:
    print("file does not exist. Please check the path and try again.")