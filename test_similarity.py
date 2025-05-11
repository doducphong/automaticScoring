from evaluate_docx_submission import calculate_text_similarity, init_phobert_model, get_text_embedding
import time
import torch
import torch.nn as nn

# Initialize the model (this happens in the original script too but we want to make sure)
print("Initializing model...")
global phobert_model, phobert_tokenizer, device
phobert_model, phobert_tokenizer, device = init_phobert_model()

# Let's wait a bit for model to load
time.sleep(2)

# Test the pair from vietnamese_sentence_pairs_example.csv
text1 = "Chúng tôi đang đi du lịch ở biển."
text2 = "Tôi đang đọc một cuốn sách hay."
expected_similarity = 0.9

def calculate_text_similarity_phoBERT(text1: str, text2: str) -> float:
    """
    Predict similarity score between two texts using the fine-tuned PhoBERT model.
    
    Args:
        text1: First input text
        text2: Second input text
    
    Returns:
        A similarity score (float)
    """
    if phobert_model is None or phobert_tokenizer is None:
        return 0.0  # or fallback using difflib if cần

    try:
        # Tokenize cặp câu
        encoded = phobert_tokenizer(
            text1,
            text2,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        # Dự đoán
        phobert_model.eval()
        with torch.no_grad():
            if "token_type_ids" in encoded:
                del encoded["token_type_ids"]
            output = phobert_model(**encoded)
            score = output.item() if output.numel() == 1 else output.squeeze().tolist()
        
        return float(score)
    
    except Exception as e:
        print(f"Error predicting similarity: {e}")
        return 0.0

# Test with direct module import to replicate potential user issue
print(f"thu model phoBERT: {calculate_text_similarity_phoBERT(text1,text2):.4f}")