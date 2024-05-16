# This program will download LLaMA-7B weights from HuggingFace and store the resulting model into the path specified by $HF_HOME
# By Default $HF_HOME = ~/.cache/huggingface

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

# Try out an example to make sure the base model is downloaded successfully and working
input_text = "The future of AI in healthcare is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
# Decode and print the generated text
print(tokenizer.decode(output[0], skip_special_tokens=True))