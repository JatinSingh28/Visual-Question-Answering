from transformers import AutoTokenizer, AutoModel
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
else:
    device = torch.device("cpu")  # Use the CPU
    
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def generateTextEmbedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    # Extract the embeddings for the [CLS] token (sentence-level representation)
    embedding = output.last_hidden_state[:, 0, :]
    return embedding
