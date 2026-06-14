import torch

# Correcting the problem of compatibility of float8_e8m0fnu
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# Le reste de ton code original reste inchangé :
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen3-4B-Base"

# Configuration 4-bit pour ta RTX 5060 (Consommation : ~2.5 Go de VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Charging tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Charging the model (4-bit mode)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# TEST 
prompt = "hello how are you"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n Generating response...")
outputs = model.generate(
    **inputs, 
    max_new_tokens=100, 
    temperature=0.7, 
    do_sample=True
)

print("\n--- RESULTS ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))