# SmartPlate - Nutrition Fine-Tuned LLM Chatbot

## Overview

SmartPlate is an AI-powered chatbot specialized in nutrition and dietary advice.

<p align="center">
  <img src="Assets/Model_finetuned_Response.png" alt="SmartPlate Demo Screenshot" width="600">
</p>

Our chatbot is built on top of the **Qwen3-4B Base Model**, which originally performs only next-token prediction without prioritizing structured outputs or domain constraints.
However, with our own touch, the model was fine-tuned to become a **nutrition-focused assistant with structured and controlled responses**.

---

## Problem with Base Model

The original Qwen3-4B model:

- Cannot properly answer a question
- Can only predict the next word without checking if the phrase makes sense 
- Has no domain specialization
- Can answer any topic without restriction
- Behaves as a generic next-token predictor

<p align="center">
  <img src="Assets/Bae_Model_Response.png" alt="Base Model Response" width="600">
</p>
---

## Fine-Tuning Objectives

The model was fine-tuned to:

-  Focus only on nutrition-related questions
-  Produce structured and well-formatted responses
-  Maintain politeness (greetings, respectful tone)
-  Reject or ignore questions out of the context of nutrition 
-  Improve conversational coherence and proper language
-  Understand queries in English and French and respond according to the language of the question 

---

## Training Data

The dataset includes:

- Nutrition Q&A pairs and nutritional values of various food products
- Healthy diet recommendations
- Meal suggestions
- Structured chatbot conversations
- Politeness and greeting examples
- Negative samples for out-of-domain rejection

### Training Configuration (QLoRA):
- **Base Model:** Qwen/Qwen3-4B-Base
- **Quantization:** 4-bit (NF4)
- **LoRA Rank (r):** 16 | **LoRA Alpha:** 32
- **Target Modules:** `q_proj`, `v_proj`, `k_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`

We targeted both the attention and feed-forward modules to modify the behavior of our base model. This configuration offers a great balance between response quality and stability.
You can see our training notebook in Google Collab : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YoussefKlibi/SmartPlate-llm-chatbot/blob/feature/model-pipeline/src/Train/Llm_training.ipynb)
You have the choice to train in local with our training file `train/Training.py` or train in Google Collab with our Notebook

---


##  Key Features
- - **Fine-Tuned Adapter:** Available on Hugging Face as [KlibiYoussef/smart-plate-nutrition-adapter](https://huggingface.co/KlibiYoussef/smart-plate-nutrition-adapter)
- Fine-tuned LLM based on Qwen3-4B Base Model
- Nutrition-specialized assistant
- Structured response generation
- Controlled domain behavior
- Hugging Face compatible training pipeline

---

##  Tech Stack & Methodology

The training pipeline relies entirely on the Hugging Face ecosystem:
- **Core Frameworks:** PyTorch, Transformers
- **Fine-Tuning Technique:** PEFT (LoRA / QLoRA) for memory-efficient training
- **Alignment:** TRL (Transformer Reinforcement Learning library) for Supervised Fine-Tuning (SFT)

---

## Known Limitations & Future Work

### Current Limitations
- **Model Capacity Constraints:** While the Qwen3-4B base model is exceptionally fast and efficient for local deployment, its smaller parameter count means it may struggle with highly complex, multi-step medical or clinical metabolic reasoning compared to larger architectures. The choice of the Qwen3-4B base model was solely motivated by its ability to be handled without problems by our local RTX 5060 hardware. 
- **Numerical & Factual Hallucinations:** the model can occasionally hallucinate specific quantitative data, such as inventing inaccurate protein counts, macronutrient breakdowns, or arbitrary nutritional percentages. Since it operates on probabilistic token prediction, it lacks an inherent mathematical engine to accurately calculate or verify precise nutritional values.
- **Adversarial Prompting (Jailbreaking):** Although negative sampling was integrated into the Supervised Fine-Tuning (SFT) dataset to reject out-of-domain queries (politics, weather ...), sophisticated prompt-injection techniques might still occasionally bypass the guardrails.

### Future Roadmap
- **Dynamic RAG Integration:** Connect the FastAPI backend to an authoritative nutritional database API to ground the model's responses in real-time, verified nutritional data and reduce mathematical hallucinations in calorie calculations. Additionally, explore web-based RAG to enrich the model's responses with up-to-date information from reliable online sources.

##  Installation

```bash
git clone https://github.com/YoussefKlibi/SmartPlate-llm-chatbot.git
cd smartplate-llm-chatbot
git switch feature/model-pipeline
pip install -r requirements.txt
```

You can test our Lora Adapter in the code file `test/Lora_test.py` (after installing the libraries)



## Author
This project was independently developed from scratch by **Klibi Youssef** to explore advanced LLM domain adaptation, and efficient QLoRA fine-tuning workflows.
