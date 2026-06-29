import warnings
import logging
import os

# 1) Silence all system environment and python warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# 2) Intercept and silence ALL warnings before they reach the console
def silence_warnings(message, category, filename, lineno, file=None, line=None):
    pass
warnings.showwarning = silence_warnings

# 3) Force AI libraries to only report critical ERRORS
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

import re
import json
import torch
import faiss
import numpy as np
from langdetect import detect
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient 
import uvicorn

# PyTorch patch for fallback float8 support compatibility
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

app = FastAPI(title="SmartPlate Hybrid RAG AI API")


# tavily api key & initialization
TAVILY_API_KEY = ""  # Insère ta clé API Tavily ici
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# 1) rag initialization (local & web)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(SCRIPT_DIR, "RAG/nutrition_index.faiss")
DOCS_PATH = os.path.join(SCRIPT_DIR, "RAG/nutrition_docs.json")

print("📥 Loading RAG embedding model...")
rag_embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

try:
    rag_index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        rag_documents = json.load(f)
    print("🔍 Local RAG database loaded successfully!")
except Exception as e:
    rag_index = None
    rag_documents = []
    print(f"⚠️ Warning: Could not load local RAG files: {e}")


# smart semantic router configuration
CHITCHAT_EXAMPLES = [
    "hello", "hi", "good morning", "bonjour", "salut", "ça va", "how are you",
    "who are you", "what is your name", "what's you're name", "qui es tu", "tu es qui",
    "comment tu t'appelles", "tell me your name", "are you human", "what can you do",
    "good evening", "bonsoir", "test", "hey there", "identity check"
]

NUTRITION_EXAMPLES = [
    "how many calories in an egg", "is it possible to reduce fats while increasing muscles",
    "how can i gain weight", "are eggs healthy these days", "c'est quoi les nouveautés alimentaires",
    "recipe for protein shake", "healthy breakfast options", "low carb diet",
    "calories dans un makroudh", "combien de proteines dans le poulet", "perdre du poids",
    "gagner du muscle", "nutrition facts", "is apple good for weight loss", "macronutrients"
]

print("pre-computing semantic router anchors...")
CHITCHAT_EMBS = rag_embedding_model.encode(CHITCHAT_EXAMPLES)
NUTRITION_EMBS = rag_embedding_model.encode(NUTRITION_EXAMPLES)

# Normalisation pour le calcul de la similarité cosinus (produit scalaire)
CHITCHAT_EMBS = CHITCHAT_EMBS / np.linalg.norm(CHITCHAT_EMBS, axis=1, keepdims=True)
NUTRITION_EMBS = NUTRITION_EMBS / np.linalg.norm(NUTRITION_EMBS, axis=1, keepdims=True)


# Fonction utilitaire pour isoler la partie "aliment" des phrases complexes
def extract_food_keywords(query: str) -> str:
    cleaned = query.lower()
    stop_phrases = [
        "is it good to", "is a choice to", "reduce weight", "weight loss", "good for",
        "how many calories in", "calories in", "what is the nutrition of", "nutrition facts for",
        "y a-t-il eu des", "y a-t-il eu", "est-ce qu'il y a", "est-ce que", 
        "sais-tu si", "peux-tu me dire", "quels sont les", "quelles sont les",
        "au cours des derniers mois", "récemment", "s'il vous plaît", "au cours de",
        "c'est quoi les", "c'est quoi", "qu'est-ce que les", "qu'est ce que", 
        "connais-tu les", "peux-tu m'expliquer", "est-ce bon pour", "est-ce conseillé"
    ]
    for phrase in stop_phrases:
        cleaned = cleaned.replace(phrase, "")
    
    cleaned = re.sub(r'[?.,!;:()"\']', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else query.lower()


# Fonction de recherche Web via Tavily
def web_search(query: str) -> str:
    search_keywords = extract_food_keywords(query)
    print(f" Triggering Tavily Web Search for: '{query}'")
    print(f" keywords used: '{search_keywords}'")

    try:
        response = tavily_client.search(query=search_keywords, max_results=3)
        results = response.get("results", [])
        
        if results:
            context = "[REAL-TIME INTERNET SEARCH RESULTS]\n"
            for i, r in enumerate(results, 1):
                context += f"Web Source [{i}] ({r['url']}): {r['title']} - {r['content']}\n\n"
            return context
        else:
            print("Tavily returned 0 results for these keywords.")
    except Exception as e:
        print(f"Tavily search API call failed: {e}")
        
    return ""


# 2) llm model & quantization configuration
BASE_MODEL_ID    = "Qwen/Qwen3-4B-Base"
ADAPTER_MODEL_ID = "KlibiYoussef/smart-plate-nutrition-adapter"

print(f"Loading weights from Hugging Face: {ADAPTER_MODEL_ID}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)
model.eval()

STOP_WORDS = [
    "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end_of_text|>",       
    "<|extra_0|>", "<|quad_start|>", "<|quad_end|>"   
]

stop_token_ids: list[int] = []
if tokenizer.eos_token_id is not None:
    stop_token_ids.append(tokenizer.eos_token_id)

for sw in STOP_WORDS:
    tid = tokenizer.convert_tokens_to_ids(sw)
    if tid is not None and tid != tokenizer.unk_token_id and tid not in stop_token_ids:
        stop_token_ids.append(tid)


# 3) post-processing utilities
NON_LATIN_PATTERN = re.compile(r'[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0E00-\u0EFF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]')

def clean_response(text: str) -> str:
    # 1) Coupure sur les marqueurs de chat classiques
    hard_stops = [
        "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end_of_text|>",
        "<|extra_0|>", "<|quad_start|>", "<|quad_end|>", "user\n", "assistant\n", 
        "\nuser", "\nassistant", "user:", "assistant:", "คุณuser",   
    ]
    for sw in hard_stops:
        if sw in text:
            text = text.split(sw)[0]

    # 2) security anti-bleeding: immediate truncation if the model recites its instructions
    meta_leaks = [
        "stay focused", "remember your role", "smart plate assistant", 
        "you must always be polite", "warm greeting", "encouraging closing statement"
    ]
    text_lower = text.lower()
    for leak in meta_leaks:
        if leak in text_lower:
            pos = text_lower.find(leak)
            # cut just before the parasite phrase and clean the residual punctuation
            text = text[:pos].rstrip(" !,.;-")
            text_lower = text.lower()

    # 3) alignment of the first valid character
    match_start = re.search(r'[a-zA-ZÀ-ÿ]', text)
    if match_start:
        text = text[match_start.start():]
    else:
        return "I couldn't generate a valid response. Please rephrase."

    # 4) filtering of non-latin characters if accidental
    match_nonlatin = NON_LATIN_PATTERN.search(text)
    if match_nonlatin:
        text = text[:match_nonlatin.start()].rstrip(" .,;:!?")

    # 5) cleaning of multiple line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


# 4) request schema & api termination point
class ChatRequest(BaseModel):
    user_message: str
    context_rag: str = ""
    lang: str = "auto"  


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        retrieved_context = ""
        web_context = ""
        routing_decision = "LOCAL"

        # auto-detection of the base language to guide the system prompts
        langue_cible = request.lang.lower()
        if langue_cible == "auto":
            try:
                langue_cible = "en" if detect(request.user_message) == "en" else "fr"
            except:
                langue_cible = "fr"

        # smart semantic router (via embeddings)
        query_emb = rag_embedding_model.encode([request.user_message])
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # maximum cosine similarity scores calculation
        chitchat_sims = np.dot(CHITCHAT_EMBS, query_emb.T).flatten()
        nutrition_sims = np.dot(NUTRITION_EMBS, query_emb.T).flatten()
        
        max_chitchat = float(np.max(chitchat_sims))
        max_nutrition = float(np.max(nutrition_sims))
        
        # routing decision based on the maximum similarity scores
        if max_chitchat > max_nutrition:
            routing_mode = "CHITCHAT"
        else:
            routing_mode = "NUTRITION"
            
        print(f"semantic router: max chitchat: {max_chitchat:.4f} | max nutrition: {max_nutrition:.4f} -> chosen: {routing_mode}")

        # immediate chitchat processing (without rag, without tavily)
        if routing_mode == "CHITCHAT":
            print("routing engine: mode: chitchat (bypassing rag & web completely)")
            
            current_sys_prompt = (
                "Tu es l'assistant Smart Plate, une IA experte en nutrition. "
                "Présente-toi brièvement ou réponds à la salutation/identité de manière naturelle, polie et très concise."
            ) if langue_cible == "fr" else (
                "You are the Smart Plate assistant, an AI specialized in nutrition. "
                "Briefly introduce yourself or respond to the greeting/identity question in a natural, polite, and very concise manner."
            )
            
            prompt = f"<|im_start|>system\n{current_sys_prompt}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{request.user_message}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=0.4, 
                    eos_token_id=stop_token_ids,
                    pad_token_id=tokenizer.eos_token_id
                )
            raw_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return {
                "response": clean_response(raw_response), 
                "detected_language": langue_cible, 
                "routing_mode": "CHITCHAT", 
                "injected_context": ""
            }

        # local rag execution only if the intention is nutrition
        has_local_match = False
        if rag_index is not None and len(rag_documents) > 0:
            faiss_query = extract_food_keywords(request.user_message)
            print(f"faiss query used: '{faiss_query}'")
            
            query_vector = rag_embedding_model.encode([faiss_query]).astype('float32')
            distances, indices = rag_index.search(query_vector, k=2)
            
            matched_contexts = []
            DISTANCE_THRESHOLD = 20.0
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(rag_documents):
                    doc_text = rag_documents[idx].lower()
                    print(f"faiss debug: index {idx} | brute distance calculated: {dist:.4f}")
                    
                    if dist <= DISTANCE_THRESHOLD:
                        # Double vérification lexicale sémantique
                        if faiss_query in doc_text or any(word in doc_text for word in faiss_query.split() if len(word) > 3):
                            print(f"match validated: '{faiss_query}' found textually at index {idx}")
                            matched_contexts.append(rag_documents[idx])
                            has_local_match = True
                        else:
                            print(f"match rejected: semantic false positive. '{faiss_query}' absent from index {idx}")
            
            if has_local_match:
                retrieved_context = "\n".join(matched_contexts)

        # web explicit need detection
        web_indicators = ["nouveau", "nouveauté", "actualité", "récemment", "2026", "restaurant", "prix", "acheter", "tendance"]
        needs_web_explicit = any(w in request.user_message.lower() for w in web_indicators)

        # routing decision between local and web
        if has_local_match and not needs_web_explicit:
            routing_decision = "LOCAL"
        else:
            routing_decision = "WEB"

        print(f"routing engine: choice: {routing_decision} | local match: {has_local_match} | web explicit: {needs_web_explicit}")

        # collect and structure the chosen rag context
        final_context_blocks = []
        if routing_decision == "LOCAL" and retrieved_context:
            final_context_blocks.append(retrieved_context)
        elif routing_decision == "WEB":
            web_context = web_search(request.user_message) 
            if web_context:
                final_context_blocks.append(web_context)

        request.context_rag = "\n\n".join(final_context_blocks)

        # system instructions for the nutrition data processing
        SYS_FR = (
            "Tu es l'assistant Smart Plate. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.\n"
            "CONSIGNES STRICTES :\n"
            "1) Ne retiens QUE les informations directement liées à la nutrition, l'alimentation et les calories. Élimine tout le reste.\n"
            "2) Reste concis. Fais des phrases courtes.\n"
            "3) Si la réponse contient plusieurs éléments, utilise impérativement une liste à puces.\n"
            "4) Ne commence jamais par des formules de politesse de remplissage."
        )
        
        SYS_EN = (
            "You are the Smart Plate assistant. Answer the user's query using ONLY the facts from the provided context.\n"
            "STRICT RULES:\n"
            "1) Focus EXCLUSIVELY on nutrition, diet, and calorie-related data. Filter out unrelated healthcare or hygiene facts.\n"
            "2) Be concise and write short sentences.\n"
            "3) Always use bullet points if there are multiple developments or facts to list.\n"
            "4) Do not use generic introductory remarks."
        )
        
        current_sys_prompt = SYS_FR if langue_cible == "fr" else SYS_EN

        # format alignment fix
        if request.context_rag:
            user_payload = f"Contexte additionnel :\n{request.context_rag}\n\nQuestion : {request.user_message}"
        else:
            user_payload = request.user_message

        prompt = f"<|im_start|>system\n{current_sys_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_payload}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"

        # tokenisation and execution of the inference
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, 
                do_sample=False,       
                use_cache=True,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15, 
            )

        generated_tokens = outputs[0][prompt_length:]
        raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        clean = clean_response(raw_response)

        return {
            "response": clean, 
            "detected_language": langue_cible,
            "routing_mode": routing_decision,
            "injected_context": request.context_rag
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)