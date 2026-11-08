import os
import pathlib

# --- Correctif Windows : UnicodeDecodeError ---
_orig_read_text = pathlib.Path.read_text
def _patched_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = "utf-8"
    return _orig_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = _patched_read_text

import torch

# --- Compatibilité anciennes versions de PyTorch ---
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ==============================================================================
# Chemins
# ==============================================================================
dataset_path = "C:/Users/MSI/Desktop/Fine_Tunning/Data/smart_plate_nutrition_dataset_clean.jsonl"
DOSSIER_SAUVEGARDE_MODELE = "C:/Users/MSI/Desktop/Fine_Tunning/Model/smart_plate_nutrition_model_finetuned"
os.makedirs(DOSSIER_SAUVEGARDE_MODELE, exist_ok=True)

# ==============================================================================
# Modèle
# ------------------------------------------------------------------------------
# On part d'un vrai modèle de base. Il ne connaît pas (ou mal) les tokens ChatML
# <|im_start|> / <|im_end|> : on entraîne donc explicitement leurs embeddings
# (voir trainable_token_indices dans la config LoRA plus bas).
# ==============================================================================
model_id = "Qwen/Qwen3-4B-Base"

# ==============================================================================
# Tokenizer
# ==============================================================================
print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

# IMPORTANT : pad doit être DIFFÉRENT de eos.
# Sinon le collator met les labels des tokens de fin à -100 (masqués) et le
# modèle n'apprend jamais à s'arrêter. Sur Instruct c'est déjà le cas ; sur Base
# eos et pad valent tous les deux <|endoftext|>.
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|endoftext|>"

# Template ChatML SIMPLE. Le template Qwen3 d'origine insère "<think>\n\n</think>\n\n"
# au début de chaque réponse : on ne le veut pas dans les données d'entraînement.
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# ==============================================================================
# Dataset : format prompt / completion
# ------------------------------------------------------------------------------
# Ainsi la loss ne porte QUE sur la réponse de l'assistant (plus sur le system
# prompt ni sur les messages user), et le modèle n'apprend plus à inventer des
# tours "user".
# ==============================================================================
print(f"Chargement du dataset depuis : {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")

def to_prompt_completion(example):
    msgs = example["messages"]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}

dataset = dataset.map(to_prompt_completion, remove_columns=dataset.column_names)

# Vérification visuelle : regarde bien le texte final (system, user, assistant,
# <|im_end|> en fin de réponse, pas de balises <think> inattendues).
example_text = tokenizer.apply_chat_template(
    dataset[0]["prompt"] + dataset[0]["completion"], tokenize=False
)
print("\n--- Exemple formaté ---")
print(example_text)
print("--- Fin de l'exemple ---\n")
assert "<think>" not in example_text, "Le template ajoute encore un bloc <think>."
assert example_text.rstrip("\n").endswith("<|im_end|>"), "L'exemple ne finit pas par <|im_end|>."

# ==============================================================================
# Modèle 4-bit (QLoRA)
# ==============================================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Chargement du modèle en 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ==============================================================================
# LoRA
# ==============================================================================
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"IDs ChatML : <|im_start|>={im_start_id}  <|im_end|>={im_end_id}")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Entraîne UNIQUEMENT les lignes d'embedding de ces 2 tokens (très léger en VRAM).
    # LoRA seul ne modifie pas les embeddings, donc sans ça ces tokens restent "bruts".
    # Nécessite un PEFT récent : pip install -U peft
    # Si ta version est trop ancienne ou si ça échoue, remplace par :
    #   modules_to_save=["embed_tokens", "lm_head"]   (beaucoup plus gourmand en VRAM)
    trainable_token_indices={"embed_tokens": [im_start_id, im_end_id]},
)
print("Configuration LoRA terminée.")

# ==============================================================================
# Config d'entraînement
# ==============================================================================
training_args = SFTConfig(
    output_dir=DOSSIER_SAUVEGARDE_MODELE,
    max_length=1024,                 # remplace l'ancien training_args.max_seq_length (ignoré)
    completion_only_loss=True,       # loss uniquement sur la réponse de l'assistant
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    optim="paged_adamw_8bit",
    bf16=True,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

print("\nDébut de l'entraînement...")
trainer.train()

print(f"\nEntraînement terminé ! Sauvegarde dans : {DOSSIER_SAUVEGARDE_MODELE}")
trainer.model.save_pretrained(DOSSIER_SAUVEGARDE_MODELE)
tokenizer.save_pretrained(DOSSIER_SAUVEGARDE_MODELE)