import torch
import tiktoken
from model import GPT # Import from your new, correct model.py

# --- Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley in the Andes Mountains. The unicorns were unlike any ever imagined;"

# --- Load the Model using the Foolproof Method ---
print("Loading model using the foolproof from_pretrained method...")
model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)
print("Model loaded successfully!")

# --- Tokenizer and Inference ---
enc = tiktoken.get_encoding("gpt2")
start_ids = enc.encode(prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print("Generating text...")
with torch.no_grad():
    y = model.generate(x, max_new_tokens=150, temperature=0.7, top_k=50)
    generated_text = enc.decode(y[0].tolist())
    print("\n--- YOUR MODEL'S OUTPUT ---")
    print(generated_text)
    print("--------------------")