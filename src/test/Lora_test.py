import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-4B-Base"
ADAPTER_ID = "KlibiYoussef/smart-plate-nutrition-adapter"

# 4-bit loading, same as during training (bf16 on Ampere+ GPUs, fp16 otherwise, e.g. T4)
use_bf16 = torch.cuda.get_device_capability()[0] >= 8
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()

# English example. For a French answer, use the French system prompt
# (see "Prompt format") and a French question.
SYSTEM_PROMPT = (
    "You are the 'Smart Plate Nutrition' AI assistant. Your role is to provide ultra-precise, "
    "scientific, and personalized nutritional information. You must always be polite, helpful, "
    "start with a warm greeting, and end with an encouraging health-focused closing statement."
)
question = "How can I lose fat without losing muscle?" # you can change this question to test different inputs

prompt = f"Système :\n{SYSTEM_PROMPT}\n\nUtilisateur :\n{question}\n\nAssistant :\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
STOP = "\n\nUtilisateur :"  # safety: cut if the model starts a new user turn

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,  # <|endoftext|>
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=[STOP],
        tokenizer=tokenizer,
    )

generated = outputs[0][inputs.input_ids.shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True).split(STOP)[0].strip())