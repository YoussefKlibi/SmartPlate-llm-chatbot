import os
import json
import re
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from groq import Groq
import uvicorn

app = FastAPI(title="SmartPlate Hybrid RAG AI API")

# ─────────────────────────────────────────────
# 🔑 Clés API (Récupérées depuis Render)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# ─────────────────────────────────────────────
# 1) Initialisation RAG (FAISS + Embeddings)
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(SCRIPT_DIR, "RAG/nutrition_index.faiss")
DOCS_PATH = os.path.join(SCRIPT_DIR, "RAG/nutrition_docs.json")

print("📥 Loading RAG embedding model...")
rag_embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

try:
    rag_index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        rag_documents = json.load(f)
    print("🔍 Local RAG loaded successfully!")
except Exception as e:
    rag_index = None
    rag_documents = []
    print(f"⚠️ Warning loading RAG: {e}")

# ─────────────────────────────────────────────
# 2) Routeur Sémantique
# ─────────────────────────────────────────────
CHITCHAT_EXAMPLES = [
    "hello", "hi", "good morning", "bonjour", "salut", "ça va", "how are you",
    "who are you", "what is your name", "qui es tu", "comment tu t'appelles"
]

NUTRITION_EXAMPLES = [
    "how many calories in an egg", "how can i gain weight", "calories dans un makroudh",
    "combien de proteines dans le poulet", "perdre du poids", "gagner du muscle", "macronutrients"
]

CHITCHAT_EMBS = rag_embedding_model.encode(CHITCHAT_EXAMPLES)
NUTRITION_EMBS = rag_embedding_model.encode(NUTRITION_EXAMPLES)
CHITCHAT_EMBS = CHITCHAT_EMBS / np.linalg.norm(CHITCHAT_EMBS, axis=1, keepdims=True)
NUTRITION_EMBS = NUTRITION_EMBS / np.linalg.norm(NUTRITION_EMBS, axis=1, keepdims=True)

def extract_food_keywords(query: str) -> str:
    cleaned = query.lower()
    stop_phrases = ["is it good to", "how many calories in", "calories in", "qu me dire", "quels sont les"]
    for phrase in stop_phrases:
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r'[?.,!;:()"\']', ' ', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip() or query.lower()

def web_search(query: str) -> str:
    if not tavily_client:
        return ""
    try:
        keywords = extract_food_keywords(query)
        response = tavily_client.search(query=keywords, max_results=3)
        results = response.get("results", [])
        if results:
            context = "[SEARCH RESULTS]\n"
            for i, r in enumerate(results, 1):
                context += f"Source [{i}]: {r['title']} - {r['content']}\n"
            return context
    except Exception as e:
        print(f"❌ Tavily search failed: {e}")
    return ""

# ─────────────────────────────────────────────
# 3) Endpoint API
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_message: str
    context_rag: str = ""
    lang: str = "auto"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Routeur Sémantique
        query_emb = rag_embedding_model.encode([request.user_message])
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        max_chitchat = float(np.max(np.dot(CHITCHAT_EMBS, query_emb.T)))
        max_nutrition = float(np.max(np.dot(NUTRITION_EMBS, query_emb.T)))
        
        routing_mode = "CHITCHAT" if max_chitchat > max_nutrition else "NUTRITION"
        
        # Mode Chitchat direct
        if routing_mode == "CHITCHAT":
            sys_prompt = "Tu es l'assistant SmartPlate, une IA experte en nutrition. Réponds de manière polie, naturelle et concise."
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": request.user_message}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return {"response": completion.choices[0].message.content, "routing_mode": "CHITCHAT"}

        # Mode Nutrition (RAG / Web)
        retrieved_context = ""
        has_local_match = False
        if rag_index is not None and len(rag_documents) > 0:
            faiss_query = extract_food_keywords(request.user_message)
            query_vector = rag_embedding_model.encode([faiss_query]).astype('float32')
            distances, indices = rag_index.search(query_vector, k=2)
            matched = [rag_documents[idx] for idx in indices[0] if idx < len(rag_documents) and distances[0][0] <= 20.0]
            if matched:
                retrieved_context = "\n".join(matched)
                has_local_match = True

        web_indicators = ["nouveau", "nouveauté", "2026", "prix", "actualité", "restaurant"]
        needs_web = any(w in request.user_message.lower() for w in web_indicators)

        final_context = retrieved_context if (has_local_match and not needs_web) else web_search(request.user_message)

        SYS_PROMPT = (
            "Tu es l'assistant SmartPlate, un expert en nutrition.\n"
            "Réponds avec précision, concision et professionnalisme.\n"
            "Si un contexte est fourni, utilise-le pour appuyer ta réponse.\n"
            "Utilise des listes à puces si la réponse contient plusieurs points."
        )

        user_content = f"Contexte :\n{final_context}\n\nQuestion : {request.user_message}" if final_context else request.user_message

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=300,
            temperature=0.5
        )

        return {
            "response": completion.choices[0].message.content,
            "routing_mode": "NUTRITION",
            "injected_context": final_context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)