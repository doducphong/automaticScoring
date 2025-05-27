import zipfile
from io import BytesIO
from PIL import Image
import os
from PIL import Image, UnidentifiedImageError
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np


def extract_images_as_pil(docx_path):
    """
    Trích xuất tất cả hình ảnh hợp lệ từ file .docx và trả về danh sách đối tượng PIL.Image.
    Bỏ qua ảnh lỗi hoặc không nhận diện được.
    """
    images = []

    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        image_files = [f for f in docx_zip.namelist() if f.startswith("word/media/")]

        for image_file in image_files:
            try:
                image_data = docx_zip.read(image_file)
                pil_image = Image.open(BytesIO(image_data)).convert("RGB")
                images.append({
                    'filename': os.path.basename(image_file),
                    'image': pil_image
                })
            except UnidentifiedImageError:
                print(f" bo qua '{image_file}' vi khong phai anh hop le.")
            except Exception as e:
                print(f" loi khong xac dinh voi '{image_file}': {e}")

    return images

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

images1 = extract_images_as_pil('D:/doAnTuChamDiemFileWordExcel/dethi1.docx')
images2 = extract_images_as_pil('D:/doAnTuChamDiemFileWordExcel/bailam1.docx')

for i in range(len(images1)):
    for j in range(len(images2)):
        sim_score = compare_clip_similarity(images1[i]['image'], images2[j]['image'])
        print(f"Similarity giua {images1[i]['filename']} va {images2[j]['filename']}: {sim_score:.4f}")

# # Ví dụ: Hiển thị từng ảnh
# for img_info in images1:
#     print("🖼️", img_info["filename"])
#     img_info["image"].show()