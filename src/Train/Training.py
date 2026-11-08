import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# ==============================================================================
# Configuration
# ==============================================================================
BASE_MODEL_ID = "Qwen/Qwen3-4B-Base"
# Dossier de l'adapter téléchargé depuis Drive
ADAPTER_PATH = "C:/Users/MSI/Desktop/Fine_Tunning/Model/smart_plate_nutrition_model_finetuned"

# IMPORTANT : EXACTEMENT le même system prompt que dans le dataset d'entraînement
# (en français, avec vouvoiement).
SYSTEM_PROMPT = (
    "Tu es l'assistant de 'Smart Plate Nutrition'. Ton rôle est de fournir des "
    "informations nutritionnelles ultra-précises, scientifiques et personnalisées. "
    "Tu dois impérativement t'exprimer avec une grande politesse, utiliser "
    "rigoureusement le vouvoiement, commencer par une salutation chaleureuse et "
    "terminer en encourageant l'utilisateur de manière bienveillante."
)
QUESTION = "Comment perdre de la graisse sans perdre de muscle ?"

# ==============================================================================
# Chargement : 4-bit, comme à l'entraînement
# (bf16 si GPU Ampere ou plus récent, sinon fp16 — cas de la T4)
# ==============================================================================
use_bf16 = torch.cuda.get_device_capability()[0] >= 8
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ==============================================================================
# Prompt : IDENTIQUE au template d'entraînement (marqueurs en texte normal)
# ==============================================================================
prompt = (
    f"Système :\n{SYSTEM_PROMPT}\n\n"
    f"Utilisateur :\n{QUESTION}\n\n"
    "Assistant :\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
STOP = "\n\nUtilisateur :"   # sécurité : coupe si le modèle invente un nouveau tour

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.0,                  # 1.0 : pas de pénalité
        eos_token_id=tokenizer.eos_token_id,     # <|endoftext|>
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=[STOP],
        tokenizer=tokenizer,
    )

generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(text.split(STOP)[0].strip())
