import os
import sys
import pathlib


#  CORRECTION WINDOWS crash UnicodeDecodeError 

_orig_read_text = pathlib.Path.read_text
def _patched_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'  # forcing UTF-8 encoding
    return _orig_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = _patched_read_text
# ==============================================================================
import torch

# Correcting the problems of the float8_e8m0fnu attribute for compatibility with older versions of PyTorch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# Where you want to save your fine-tuned model and the path to your dataset

dataset_path = "C:/Users/MSI/Desktop/Fine_Tunning/smart_plate_nutrition_dataset_complete_PRO.jsonl"
DOSSIER_SAUVEGARDE_MODELE = "C:/Users/MSI/Desktop/Fine_Tunning/smart_plate_nutrition_model_finetuned"

if not os.path.exists(DOSSIER_SAUVEGARDE_MODELE):
    os.makedirs(DOSSIER_SAUVEGARDE_MODELE)
# ==============================================================================

model_id = "Qwen/Qwen3-4B-Base"

# Charging the dataset
print(f"Chargement du dataset depuis : {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Chargement tokenizer and padding configuration
print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configuration of the quantization for the model 4-bits
# We used bfloat16 for the compute_dtype to avoid the issues with the GradScaler and mixed precision training that we faced with fp16. 
# This allows us to leverage the benefits of 4-bit quantization while maintaining stability during training.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True
)

print("Charging the base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# enabling gradient checkpointing in case of memory issues during training, and preparing the model for k-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Configuration of LoRA for fine-tuning
# We targeted the projection layers of the transformer architecture, which are important for the attention mechanism and the feed-forward networks.
#You can adjust the r and lora_alpha parameters based on your specific needs and computational resources. The chosen values of r=16 and lora_alpha=32 provides me a good balance between model capacity and training efficiency for many tasks.
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
print("Configuration LoRA done successfully.")

# SftConfig for training
training_args = SFTConfig(
    output_dir=DOSSIER_SAUVEGARDE_MODELE,  
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
    report_to="none"
)

# 🛠️ ASTUCE : On conserve l'injection de la longueur maximale
training_args.max_seq_length = 1024

# 7. Initialisation of SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer, 
    args=training_args,
)

# 8. beginning of the training
print("\n the beginning of the training... ")
trainer.train()

# 9. Sauvegarde finale dans TON dossier personnalisé
print(f"\n✅ Entraînement terminé ! Sauvegarde des poids finaux dans : {DOSSIER_SAUVEGARDE_MODELE}")
trainer.model.save_pretrained(DOSSIER_SAUVEGARDE_MODELE)
tokenizer.save_pretrained(DOSSIER_SAUVEGARDE_MODELE)