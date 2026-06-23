import warnings
import logging
import os

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

#Silence warnings from libraries like transformers and bitsandbytes
# these warnings polluted the console and are not relevant for the end-user
def silence_warnings(message, category, filename, lineno, file=None, line=None):
    pass
warnings.showwarning = silence_warnings

# Silence logging from transformers and bitsandbytes
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
import uvicorn

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

app = FastAPI(title="SmartPlate Hybrid RAG AI API")


# 1) RAG Database Initialization (Local & Web)

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

# Fonction outil pour chercher sur Internet en temps réel
def web_search(query: str) -> str:
    """
    Recherche sur Internet de manière robuste.
    Tente d'abord DuckDuckGo, et utilise Bing avec de bons en-têtes en cas de secours.
    """
    print(f"🌐 Triggering Web Search for: '{query}'")
    
    # 1 : DUCKDUCKGO
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # On cherche les 3 meilleurs résultats liés à la nutrition/santé
            results = list(ddgs.text(f"{query} nutrition facts health", max_results=3))
            if results:
                context = "[REAL-TIME INTERNET SEARCH RESULTS]\n"
                for i, r in enumerate(results, 1):
                    context += f"Web Source [{i}] ({r['href']}): {r['title']} - {r['body']}\n\n"
                return context
    except Exception as e:
        print(f"⚠️ DuckDuckGo search failed ({e}), trying fallback to Bing...")

    # 2 : BING AVEC COMPOSANT USER-AGENT (Fallback)
    try:
        import requests
        # not to be blocked by Bing, we use a common User-Agent string
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = f"https://www.bing.com/search?q={requests.utils.quote(query + ' nutrition facts health')}"
        
        # timeout
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            # extraction text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            snippets = [s.get_text() for s in soup.find_all("p")[:4]]
            
            if snippets:
                context = "[REAL-TIME INTERNET SEARCH RESULTS (Fallback)]\n"
                context += "\n".join(snippets)
                return context
    except Exception as e:
        print(f"❌ All web search methods failed: {e}")
        
    return ""



# 2) LLM Model & Quantization Setup

BASE_MODEL_ID    = "Qwen/Qwen3-4B-Base"
ADAPTER_MODEL_ID = "KlibiYoussef/smart-plate-nutrition-adapter" # My fine-tuned adapter on top of Qwen3-4B-Base

print(f"📦 Charging weights since Hugging Face : {ADAPTER_MODEL_ID}")

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

# additional stop tokens to prevent the model from generating unwanted artifacts
STOP_WORDS = [
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|end_of_text|>",       
    "<|extra_0|>",           
    "<|quad_start|>",  
    "<|quad_end|>",    
]

stop_token_ids: list[int] = []
if tokenizer.eos_token_id is not None:
    stop_token_ids.append(tokenizer.eos_token_id)

for sw in STOP_WORDS:
    tid = tokenizer.convert_tokens_to_ids(sw)
    if tid is not None and tid != tokenizer.unk_token_id and tid not in stop_token_ids:
        stop_token_ids.append(tid)



# 3) Post-Processing Utilities
NON_LATIN_PATTERN = re.compile(r'[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0E00-\u0EFF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]')

def clean_response(text: str) -> str:
    hard_stops = [
        "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end_of_text|>",
        "<|extra_0|>", "<|quad_start|>", "<|quad_end|>", "user\n", "assistant\n", 
        "\nuser", "\nassistant", "user:", "assistant:", "คุณuser",   
    ]
    for sw in hard_stops:
        if sw in text:
            text = text.split(sw)[0]

    match_start = re.search(r'[a-zA-ZÀ-ÿ]', text)
    if match_start:
        text = text[match_start.start():]
    else:
        return "Je n'ai pas pu générer une réponse valide. Veuillez reformuler."

    match_nonlatin = NON_LATIN_PATTERN.search(text)
    if match_nonlatin:
        text = text[:match_nonlatin.start()].rstrip(" .,;:!?")

    #security: remove excessive newlines and spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()

# 4) FastAPI Endpoint
class ChatRequest(BaseModel):
    user_message: str
    context_rag: str = ""
    lang: str = "auto"  


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        retrieved_context = ""
        
        #  local RAG
        if rag_index is not None and len(rag_documents) > 0:
            query_vector = rag_embedding_model.encode([request.user_message]).astype('float32')
            distances, indices = rag_index.search(query_vector, k=2) # Top 2 locaux
            matched_contexts = [rag_documents[idx] for idx in indices[0] if idx < len(rag_documents)]
            retrieved_context = "\n".join(matched_contexts)

        # DEBUG
        print(f"\n🔍 [DEBUG LOCAL FAISS] Context found :\n{retrieved_context}")
        
        # A) 2. RAG WEB : internet research
        web_context = web_search(request.user_message) 
        print(f"🌐 [DEBUG WEB RESULTS] Context found :\n{web_context}\n")
        final_context_blocks = []
        if retrieved_context:
            final_context_blocks.append(f"[INTERNAL NUTRITION DATABASE]\n{retrieved_context}")
        if web_context:
            final_context_blocks.append(f"[REAL-TIME INTERNET SEARCH RESULTS]\n{web_context}")
        
        # On sauvegarde le contexte global fusionné
        request.context_rag = "\n\n".join(final_context_blocks)

        # B) Détection de la langue
        langue_cible = request.lang.lower()
        if langue_cible == "auto":
            try:
                langue_detectee = detect(request.user_message)
                langue_cible = "en" if langue_detectee == "en" else "fr"
            except:
                langue_cible = "fr"

        # C) Instructions system
        SYS_FR = (
            "Tu es l'assistant de 'Smart Plate Nutrition'. Ton rôle est de fournir des "
            "informations nutritionnelles ultra-précises en croisant la base de données interne et les recherches Internet fournies. "
            "Tu ne dois pas répondre à des questions hors sujet et tu dois toujours te concentrer sur la nutrition. "
            "Exprime-toi avec politesse (vouvoiement). "
            "IMPORTANT : Donne une réponse directe, fluide et naturelle. Ne conclus JAMAIS avec des phrases "
            "d'encouragement répétitives ou des slogans robotiques (comme 'Restez dévoué à votre régime !')."
        )

        SYS_EN = (
            "You are the 'Smart Plate Nutrition' AI assistant. Your role is to provide "
            "ultra-precise nutritional information using both internal database facts and real-time web results. "
            "Focus strictly on nutrition and ignore off-topic queries. Be helpful and professional. "
            "IMPORTANT: Provide a direct and natural answer. Do NOT append repetitive catchphrases, "
            "robotic slogans, or generic closing statements (such as 'Stay dedicated to your diet!')."
        )

        current_sys_prompt = SYS_FR if langue_cible == "fr" else SYS_EN

        # D) Construction of the Prompt ChatML 
        if request.context_rag:
            prompt = f"<|im_start|>system\n{current_sys_prompt}\n\n[ADDITIONAL NUTRITIONAL CONTEXT]:\n{request.context_rag}<|im_end|>\n"
        else:
            prompt = f"<|im_start|>system\n{current_sys_prompt}<|im_end|>\n"

        prompt += (
            f"<|im_start|>user\n{request.user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Tokenization
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, 
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                use_cache=True,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        generated_tokens = outputs[0][prompt_length:]
        
        # skip_special_tokens=True élimine structurellement l'affichage des artefacts de tokens
        raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        clean = clean_response(raw_response)

        return {
            "response": clean, 
            "detected_language": langue_cible,
            "injected_context": request.context_rag
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)