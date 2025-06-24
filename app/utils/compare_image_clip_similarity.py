import zipfile
from io import BytesIO
from PIL import Image
import os
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


def compare_clip_similarity(img1: Image.Image, img2: Image.Image):
    """
    So sánh mức độ giống nhau giữa 2 ảnh dựa trên embedding từ mô hình CLIP.
    Trả về giá trị cosine similarity (càng gần 1 là càng giống).
    """
    inputs = processor(images=[img1, img2], return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # normalize

    similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
    return similarity

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
