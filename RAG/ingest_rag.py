import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1) Handle absolute paths dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")

JSON_PATH = os.path.join(DATA_DIR, "nutrition_Mix.json")
EXCEL_PATH = os.path.join(DATA_DIR, "nutrition.xlsx")

# Paths for saving the output index files
INDEX_SAVE_PATH = os.path.join(SCRIPT_DIR, "nutrition_index.faiss")
DOCS_SAVE_PATH = os.path.join(SCRIPT_DIR, "nutrition_docs.json")

print("📥 Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

documents = []

# 2) Process the JSON file
try:
    print(f"📖 Looking for JSON at: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    for food_id, info in json_data.items():
        food_name = food_id.replace("_", " ")
        text_doc = (
            f"Food item: {food_name}. "
            f"Average nutritional values: "
            f"{info.get('calories', 0)} kcal, "
            f"Proteins: {info.get('proteines_g', 0)}g, "
            f"Carbohydrates: {info.get('glucides_g', 0)}g, "
            f"Fats: {info.get('lipides_g', 0)}g, "
            f"Fiber: {info.get('fibres_g', 0)}g."
        )
        documents.append(text_doc)
    print(f"✅ JSON processed: {len(json_data)} food items added.")
except FileNotFoundError:
    print(f"❌ JSON file not found at expected path: {JSON_PATH}")
except Exception as e:
    print(f"⚠️ Error reading JSON file: {e}")


# 3) Process the Excel file
try:
    print(f"📖 Looking for Excel at: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
    
    for _, row in df.iterrows():
        name = str(row['name'])
        serving = str(row.get('serving_size', '100 g'))
        calories = row.get('calories', 0)
        fat = row.get('total_fat', '0g')
        sodium = row.get('sodium', '0mg')
        
        text_doc = (
            f"Food item: {name}. "
            f"Nutritional values per {serving}: "
            f"Energy: {calories} kcal, "
            f"Total Fat: {fat}, "
            f"Sodium: {sodium}."
        )
        documents.append(text_doc)
    print(f"✅ Excel processed: {len(df)} food items added.")
except FileNotFoundError:
    print(f"❌ Excel file not found at expected path: {EXCEL_PATH}")
except Exception as e:
    print(f"⚠️ Error reading Excel file: {e}")


# 4) Vectorize and generate the local FAISS index
if documents:
    print(f"⚡ Vectorizing {len(documents)} documents...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, INDEX_SAVE_PATH)
    with open(DOCS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
        
    print(f"🎉 Local RAG knowledge base successfully generated at:\n👉 {INDEX_SAVE_PATH}")
else:
    print("❌ No documents could be extracted. Index generation aborted.")