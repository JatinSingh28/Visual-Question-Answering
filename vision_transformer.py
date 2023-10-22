import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
else:
    device = torch.device("cpu")  # Use the CPU
    
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor(model_name)
model = ViTModel.from_pretrained(model_name).to(device)

def generateEmbeddings(imagePath):
    image = Image.open(imagePath)
    # Resize the image to the model's expected size (e.g., 384x384 for ViT models)
    image = image.resize((384, 384))

    # Convert the image to a PyTorch tensor
    image_tensor = feature_extractor(images=image, return_tensors="pt")["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Extract the embeddings for the [CLS] token
    return embeddings
