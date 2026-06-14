import os
import re
import torch
from langdetect import detect


if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn

app = FastAPI(title="SmartPlate Nutrition AI API")


BASE_MODEL_ID    = "Qwen/Qwen3-4B-Base"
ADAPTER_MODEL_ID = "KlibiYoussef/smart-plate-nutrition-adapter"

print(f"📦 Charging weights since Hugging Face : {ADAPTER_MODEL_ID}")


# 1) Configuration 4-bit RTX 5060
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# 2) Charging tokenizer and base model

# Importing tokenizer and Adapter from Hugging Face 
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)


# 3) Fusion adapter LoRA
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)
model.eval()


# 4. Construction stop token IDs 
STOP_WORDS = [
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|end_of_text|>",       
    "<|extra_0|>",           # tokens special Qwen3
]

stop_token_ids: list[int] = []


if tokenizer.eos_token_id is not None:
    stop_token_ids.append(tokenizer.eos_token_id)

# conversion stops word to tokens IDs
for sw in STOP_WORDS:
    tid = tokenizer.convert_tokens_to_ids(sw)
    if tid is not None and tid != tokenizer.unk_token_id and tid not in stop_token_ids:
        stop_token_ids.append(tid)

print(f"🛑 Stop token IDs configurés : {stop_token_ids}")
print("\n🚀 Serveur SmartPlate IA opérationnel avec décodage sécurisé !")



# 4) Regex detection of scripts non-latins
NON_LATIN_PATTERN = re.compile(
    r'['
    r'\u0400-\u04FF'   # Cyrillique
    r'\u0590-\u05FF'   # Hébreu
    r'\u0600-\u06FF'   # Arabe
    r'\u0E00-\u0EFF'   # Thaï
    r'\u4E00-\u9FFF'   # Chinois
    r'\u3040-\u30FF'   # Hiragana / Katakana
    r'\uAC00-\uD7AF'   # Coréen
    r']'
)

def clean_response(text: str) -> str:
    """
    Nettoyage multi-passes de la réponse générée :
      1. Coupe aux balises de fin de tour connues
      2. Supprime les préfixes non-latins (ex: tokens thaï au début)
      3. Tronque dès l'apparition d'un caractère non-latin en milieu/fin de texte
      4. Nettoyage final des espaces / sauts de ligne superflus
    """

    # ── Passe 1 : coupe aux stop words textuels ──────────────────────────────
    hard_stops = [
        "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end_of_text|>",
        "<|extra_0|>", "user\n", "assistant\n", "\nuser", "\nassistant",
        "user:", "assistant:", "คุณuser",   # token thaï récurrent
    ]
    for sw in hard_stops:
        if sw in text:
            text = text.split(sw)[0]

    # ── Passe 2 : supprime tout préfixe avant la 1ʳᵉ lettre latine/accentuée ─
    match_start = re.search(r'[a-zA-ZÀ-ÿ]', text)
    if match_start:
        text = text[match_start.start():]
    else:
        # Aucune lettre latine trouvée → réponse vide, on renvoie un fallback
        return "Je n'ai pas pu générer une réponse valide. Veuillez reformuler."

    # ── Passe 3 : tronque dès le 1ᵉʳ caractère non-latin ────────────────────
    match_nonlatin = NON_LATIN_PATTERN.search(text)
    if match_nonlatin:
        text = text[:match_nonlatin.start()].rstrip(" .,;:!?")

    # ── Passe 4 : supprime tokens "garbage" isolés type NdrFc, paździ, etc. ──
    text = re.sub(r'\b[A-Z][a-z]?[A-Z][a-z]?[A-Z]\w*\b', '', text)  # ex: NdrFc

    # ── Passe 5 : nettoyage des espaces en double / lignes vides ─────────────
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


# ─────────────────────────────────────────────
# 6) Modèle de requête
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_message: str
    context_rag: str = ""
    lang: str = "auto"  # can be "fr", "en", or "auto" for automatic detection


# 7) Endpoint chatbot

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        SYS_FR = (
            "Tu es l'assistant de 'Smart Plate Nutrition'. Ton rôle est de fournir des "
            "informations nutritionnelles ultra-précises, scientifiques et personnalisées. "
            "tu ne dois pas répondre à des questions hors sujet (ex: météo, politique) et tu dois toujours te concentrer sur la nutrition. "
            "Tu dois impérativement t'exprimer avec une grande politesse, utiliser le vouvoiement, "
            "commencer par une salutation chaleureuse et terminer en encourageant l'utilisateur de manière bienveillante."
        )

        SYS_EN = (
            "You are the 'Smart Plate Nutrition' AI assistant. Your role is to provide "
            "ultra-precise, scientific, and personalized nutritional information. You must "
            "always be polite, helpful, start with a warm greeting, and end with an encouraging "
            "health-focused closing statement."
            "You should not answer off-topic questions (e.g., weather, politics) and must always focus on nutrition."
        )

        langue_cible = request.lang.lower()
        
        if langue_cible == "auto":
            try:
                langue_detectee = detect(request.user_message)
                langue_cible = "en" if langue_detectee == "en" else "fr"
            except:
                langue_cible = "fr"

        current_sys_prompt = SYS_EN if langue_cible == "en" else SYS_FR

        # 8) Construction of the prompt ChatML
        prompt = f"<|im_start|>system\n{current_sys_prompt}<|im_end|>\n"

        if request.context_rag:
            prompt += f"<|im_start|>context\n{request.context_rag}<|im_end|>\n"

        prompt += (
            f"<|im_start|>user\n{request.user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

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
    min_new_tokens=1,
    eos_token_id=stop_token_ids,
    pad_token_id=tokenizer.eos_token_id,
    forced_eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
)

        generated_tokens = outputs[0][prompt_length:]
        raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        clean = clean_response(raw_response)

        return {"response": clean, "detected_language": langue_cible}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)