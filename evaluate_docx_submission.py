#!/usr/bin/env python3
import os
import sys
import zipfile
import json
from lxml import etree
import difflib
from typing import Dict, List, Any, Tuple
import docx
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import numpy as np
import re
from docx_properties_extractor import (
    extract_drop_caps, 
    get_font_info, 
    extract_wordart, 
    extract_columns, 
    extract_symbols, 
    extract_image_info, 
    extract_relationships,
    get_docx_margins,
    extract_tables,
    namespaces
)

# Define a custom model class that matches the structure of the fine-tuned model
class PhoBERTSimilarityModel(nn.Module):
    def __init__(self):
        super(PhoBERTSimilarityModel, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        # Add classifier layers that match the structure in the saved model
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # First linear layer
            nn.ReLU(),           # ReLU activation
            nn.Dropout(0.1),     # Dropout layer
            nn.Linear(256, 1)    # Output layer
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply classifier
        similarity = self.classifier(cls_output)
        return similarity
    
    def get_embedding(self, input_ids, attention_mask):
        # For similarity calculation, just return the [CLS] embeddings
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

# Initialize PhoBERT model and tokenizer
def init_phobert_model():
    """
    Initialize and return the PhoBERT model and tokenizer.
    Uses a fine-tuned model from the local directory.
    """
    try:
        print("Loading fine-tuned PhoBERT model...")
        # Load the tokenizer from the local directory
        tokenizer_path = os.path.join("fine_tuned_phobert", "tokenizer")
        model_path = os.path.join("fine_tuned_phobert", "phobert_similarity_model_v2.pth")
        
        # Check if the paths exist
        if not os.path.exists(tokenizer_path) or not os.path.exists(model_path):
            print(f"Warning: Fine-tuned model files not found at {tokenizer_path} or {model_path}")
            print("Falling back to pre-trained PhoBERT model")
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            model = AutoModel.from_pretrained("vinai/phobert-base")
        else:
            # Load tokenizer from local directory
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Create a custom model instance
            print("Creating custom PhoBERT similarity model...")
            model = PhoBERTSimilarityModel()
            
            # Load the fine-tuned weights
            print(f"Loading fine-tuned weights from {model_path}")
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("Fine-tuned model loaded successfully")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        model.eval()  # Set the model to evaluation mode
        
        return model, tokenizer, device
    except Exception as e:
        import traceback
        print(f"Error loading PhoBERT model: {e}")
        traceback.print_exc()
        return None, None, None

# Global variables for model, tokenizer, and device
phobert_model, phobert_tokenizer, device = init_phobert_model()

def extract_text_content(docx_file: str) -> List[Dict[str, Any]]:
    """
    Extract text content from a .docx file paragraph by paragraph, including font information.
    
    Args:
        docx_file: Path to the .docx file
        
    Returns:
        List of dictionaries containing paragraph text and font information
    """
    try:
        doc = docx.Document(docx_file)
        paragraphs = []
        
        for p in doc.paragraphs:
            if not p.text.strip():
                continue
            
            # Get font info from the first run
            font_name = None
            font_size = None
            line_spacing = None
            
            if p.runs:
                run = p.runs[0]
                if hasattr(run, 'font') and hasattr(run.font, 'name'):
                    font_name = run.font.name
                
                if hasattr(run, 'font') and hasattr(run.font, 'size') and run.font.size is not None:
                    try:
                        # Font size in docx is in half-points, convert to points
                        font_size = run.font.size / 12700
                    except (TypeError, ValueError):
                        font_size = None

            # Get line spacing
            if hasattr(p, 'paragraph_format') and hasattr(p.paragraph_format, 'line_spacing'):
                line_spacing = p.paragraph_format.line_spacing

            # Extract paragraph text and font info
            para_info = {
                "text": p.text,
                "font_name": font_name,
                "font_size": font_size,
                "line_spacing": line_spacing
            }
            
            paragraphs.append(para_info)
        
        return paragraphs
    except Exception as e:
        import traceback
        print(f"Error extracting text content from {docx_file}: {e}")
        traceback.print_exc()
        return []

def get_text_embedding(text: str):
    """
    Get embedding vector for a text using PhoBERT.
    
    Args:
        text: Input text
        
    Returns:
        Embedding vector
    """
    if phobert_model is None or phobert_tokenizer is None:
        # Fallback to simple text comparison if model isn't available
        return None
    
    try:
        # Tokenize the text
        inputs = phobert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            if hasattr(phobert_model, 'get_embedding'):
                # Use custom model's get_embedding method
                embeddings = phobert_model.get_embedding(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                return embeddings.cpu().numpy()[0]
            else:
                # Use base model
                outputs = phobert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                return embeddings[0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using PhoBERT embeddings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Get embeddings
    emb1 = get_text_embedding(text1)
    emb2 = get_text_embedding(text2)
    
    # If embeddings are available, use cosine similarity
    if emb1 is not None and emb2 is not None:
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(emb1, emb2)
        return float(similarity)
    
    # Fallback to sequence matcher if embeddings fail
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def compare_margins(sample_margins: Dict[str, float], submission_margins: Dict[str, float], tolerance=0.1) -> Dict[str, Any]:
    """
    Compare page margins between sample and submission.
    
    Args:
        sample_margins: Dictionary of sample margins in cm
        submission_margins: Dictionary of submission margins in cm
        tolerance: Acceptable margin difference in cm
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {}
    }
    
    for key in sample_margins:
        sample_value = sample_margins[key]
        submission_value = submission_margins.get(key, 0)
        
        if abs(sample_value - submission_value) > tolerance:
            result["matches"] = False
            result["differences"][key] = {
                "expected": sample_value,
                "actual": submission_value,
                "difference": abs(sample_value - submission_value)
            }
    
    return result

def compare_columns(sample_columns: List[Dict], submission_columns: List[Dict]) -> Dict[str, Any]:
    """
    Compare column settings between sample and submission.
    
    Args:
        sample_columns: List of column settings from sample
        submission_columns: List of column settings from submission
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {}
    }
    
    # Check if both have columns or not
    if bool(sample_columns) != bool(submission_columns):
        result["matches"] = False
        result["differences"]["has_columns"] = {
            "expected": bool(sample_columns),
            "actual": bool(submission_columns)
        }
        return result
    
    # If both don't have columns, they match
    if not sample_columns and not submission_columns:
        return result
    
    # Compare number of column settings (sections with columns)
    if len(sample_columns) != len(submission_columns):
        result["matches"] = False
        result["differences"]["num_sections_with_columns"] = {
            "expected": len(sample_columns),
            "actual": len(submission_columns)
        }
    
    # Compare each section's column settings (up to the minimum number of sections)
    min_sections = min(len(sample_columns), len(submission_columns))
    for i in range(min_sections):
        sample_section = sample_columns[i]
        submission_section = submission_columns[i]
        
        # Compare number of columns
        if sample_section["num_columns"] != submission_section["num_columns"]:
            result["matches"] = False
            result["differences"][f"section_{i+1}_num_columns"] = {
                "expected": sample_section["num_columns"],
                "actual": submission_section["num_columns"]
            }
        
        # Compare if it has a separator
        if sample_section.get("has_separator", False) != submission_section.get("has_separator", False):
            result["matches"] = False
            result["differences"][f"section_{i+1}_has_separator"] = {
                "expected": sample_section.get("has_separator", False),
                "actual": submission_section.get("has_separator", False)
            }
    
    return result

def compare_wordart(sample_wordart: List[Dict], submission_wordart: List[Dict]) -> Dict[str, Any]:
    """
    Compare WordArt elements between sample and submission.
    
    Args:
        sample_wordart: List of WordArt elements from sample
        submission_wordart: List of WordArt elements from submission
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {}
    }
    
    # Check if both have WordArt or not
    sample_has_wordart = len(sample_wordart) > 0
    submission_has_wordart = len(submission_wordart) > 0
    
    if sample_has_wordart != submission_has_wordart:
        result["matches"] = False
        result["differences"]["has_wordart"] = {
            "expected": sample_has_wordart,
            "actual": submission_has_wordart
        }
        return result
    
    # If both don't have WordArt, they match
    if not sample_has_wordart and not submission_has_wordart:
        return result
    
    # Compare number of WordArt elements
    if len(sample_wordart) != len(submission_wordart):
        result["matches"] = False
        result["differences"]["count"] = {
            "expected": len(sample_wordart),
            "actual": len(submission_wordart)
        }
    
    # Compare each WordArt element (up to the minimum number)
    min_elements = min(len(sample_wordart), len(submission_wordart))
    for i in range(min_elements):
        sample_element = sample_wordart[i]
        submission_element = submission_wordart[i]
        
        # Compare text content
        sample_text = sample_element["text"]
        submission_text = submission_element["text"]
        similarity = calculate_text_similarity(sample_text, submission_text)
        
        if similarity < 0.99:  # 99% similarity threshold
            result["matches"] = False
            result["differences"][f"wordart_{i+1}_text"] = {
                "expected": sample_text,
                "actual": submission_text,
                "similarity": round(similarity, 2)
            }
        
        # Compare type
        if sample_element["type"] != submission_element["type"]:
            result["matches"] = False
            result["differences"][f"wordart_{i+1}_type"] = {
                "expected": sample_element["type"],
                "actual": submission_element["type"]
            }
    
    return result

def compare_drop_caps(sample_caps: List[Dict], submission_caps: List[Dict]) -> Dict[str, Any]:
    """
    Compare drop caps between sample and submission.
    
    Args:
        sample_caps: List of drop caps from sample
        submission_caps: List of drop caps from submission
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {}
    }
    
    # Check if both have drop caps or not
    sample_has_caps = len(sample_caps) > 0
    submission_has_caps = len(submission_caps) > 0
    
    if sample_has_caps != submission_has_caps:
        result["matches"] = False
        result["differences"]["has_drop_caps"] = {
            "expected": sample_has_caps,
            "actual": submission_has_caps
        }
        return result
    
    # If both don't have drop caps, they match
    if not sample_has_caps and not submission_has_caps:
        return result
    
    # Compare number of drop caps
    if len(sample_caps) != len(submission_caps):
        result["matches"] = False
        result["differences"]["count"] = {
            "expected": len(sample_caps),
            "actual": len(submission_caps)
        }
    
    # Compare each drop cap (up to the minimum number)
    min_elements = min(len(sample_caps), len(submission_caps))
    for i in range(min_elements):
        sample_cap = sample_caps[i]
        submission_cap = submission_caps[i]
        
        # Compare character
        if sample_cap["char"] != submission_cap["char"]:
            result["matches"] = False
            result["differences"][f"drop_cap_{i+1}_char"] = {
                "expected": sample_cap["char"],
                "actual": submission_cap["char"]
            }
        
        # Compare type
        if sample_cap["type"] != submission_cap["type"]:
            result["matches"] = False
            result["differences"][f"drop_cap_{i+1}_type"] = {
                "expected": sample_cap["type"],
                "actual": submission_cap["type"]
            }
        
        # Compare lines
        if sample_cap["lines"] != submission_cap["lines"]:
            result["matches"] = False
            result["differences"][f"drop_cap_{i+1}_lines"] = {
                "expected": sample_cap["lines"],
                "actual": submission_cap["lines"]
            }
    
    return result

def compare_symbols(sample_symbols: List[Dict], submission_symbols: List[Dict]) -> Dict[str, Any]:
    """
    Compare symbols between sample and submission.
    
    Args:
        sample_symbols: List of symbols from sample
        submission_symbols: List of symbols from submission
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {}
    }
    
    # Check if both have symbols or not
    sample_has_symbols = len(sample_symbols) > 0
    submission_has_symbols = len(submission_symbols) > 0
    
    if sample_has_symbols != submission_has_symbols:
        result["matches"] = False
        result["differences"]["has_symbols"] = {
            "expected": sample_has_symbols,
            "actual": submission_has_symbols
        }
        return result
    
    # If both don't have symbols, they match
    if not sample_has_symbols and not submission_has_symbols:
        return result
    
    # Group symbols by type for easier comparison
    sample_types = {}
    for sym in sample_symbols:
        sym_type = sym["type"]
        if sym_type not in sample_types:
            sample_types[sym_type] = []
        sample_types[sym_type].append(sym)
    
    submission_types = {}
    for sym in submission_symbols:
        sym_type = sym["type"]
        if sym_type not in submission_types:
            submission_types[sym_type] = []
        submission_types[sym_type].append(sym)
    
    # Compare symbol types
    all_types = set(list(sample_types.keys()) + list(submission_types.keys()))
    for sym_type in all_types:
        sample_count = len(sample_types.get(sym_type, []))
        submission_count = len(submission_types.get(sym_type, []))
        
        if sample_count != submission_count:
            result["matches"] = False
            result["differences"][f"symbol_type_{sym_type}_count"] = {
                "expected": sample_count,
                "actual": submission_count
            }
    
    # For explicit symbols and unicode symbols, check if specific characters are present
    # For each type in sample_types, check if each character appears in submission
    for sym_type in ["explicit_symbol", "unicode_symbol", "symbol_font_character"]:
        if sym_type not in sample_types:
            continue
            
        sample_chars = set()
        for sym in sample_types[sym_type]:
            sample_chars.add(sym["character"])
        
        submission_chars = set()
        if sym_type in submission_types:
            for sym in submission_types[sym_type]:
                submission_chars.add(sym["character"])
        
        missing_chars = sample_chars - submission_chars
        if missing_chars:
            result["matches"] = False
            result["differences"][f"missing_{sym_type}_chars"] = list(missing_chars)
    
    return result

def compare_images(sample_images: List[Dict], submission_images: List[Dict], tolerance=0.1) -> Dict[str, Any]:
    """
    Compare images between sample and submission.
    
    Args:
        sample_images: List of images from sample
        submission_images: List of images from submission
        tolerance: Acceptable difference ratio for image dimensions
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {}
    }
    
    # Check if both have images or not
    sample_has_images = len(sample_images) > 0
    submission_has_images = len(submission_images) > 0
    
    if sample_has_images != submission_has_images:
        result["matches"] = False
        result["differences"]["has_images"] = {
            "expected": sample_has_images,
            "actual": submission_has_images
        }
        return result
    
    # If both don't have images, they match
    if not sample_has_images and not submission_has_images:
        return result
    
    # Compare number of images
    if len(sample_images) != len(submission_images):
        result["matches"] = False
        result["differences"]["count"] = {
            "expected": len(sample_images),
            "actual": len(submission_images)
        }
    
    # Compare each image (up to the minimum number)
    min_elements = min(len(sample_images), len(submission_images))
    for i in range(min_elements):
        sample_img = sample_images[i]
        submission_img = submission_images[i]
        
        # Compare position type
        if sample_img["position_type"] != submission_img["position_type"]:
            result["matches"] = False
            result["differences"][f"image_{i+1}_position_type"] = {
                "expected": sample_img["position_type"],
                "actual": submission_img["position_type"]
            }
        
        # Compare dimensions with tolerance
        # Convert to float to ensure we can compare them
        sample_width = float(sample_img["width"]) if not isinstance(sample_img["width"], str) else 0
        sample_height = float(sample_img["height"]) if not isinstance(sample_img["height"], str) else 0
        
        submission_width = float(submission_img["width"]) if not isinstance(submission_img["width"], str) else 0
        submission_height = float(submission_img["height"]) if not isinstance(submission_img["height"], str) else 0
        
        # Check width
        if sample_width > 0 and submission_width > 0:
            width_ratio = min(sample_width, submission_width) / max(sample_width, submission_width)
            if width_ratio < 1 - tolerance:
                result["matches"] = False
                result["differences"][f"image_{i+1}_width"] = {
                    "expected": sample_width,
                    "actual": submission_width,
                    "ratio": round(width_ratio, 2)
                }
        
        # Check height
        if sample_height > 0 and submission_height > 0:
            height_ratio = min(sample_height, submission_height) / max(sample_height, submission_height)
            if height_ratio < 1 - tolerance:
                result["matches"] = False
                result["differences"][f"image_{i+1}_height"] = {
                    "expected": sample_height,
                    "actual": submission_height,
                    "ratio": round(height_ratio, 2)
                }
        
        # Compare format if available
        if sample_img["format"] != "unknown" and submission_img["format"] != "unknown":
            if sample_img["format"] != submission_img["format"]:
                result["matches"] = False
                result["differences"][f"image_{i+1}_format"] = {
                    "expected": sample_img["format"],
                    "actual": submission_img["format"]
                }
    
    return result

def clean_token(token):
    return re.sub(r'[^\wÀ-ỹ]', '', token)  # Giữ lại chữ cái tiếng Việt có dấu

def compare_content(sample_paragraphs: List[Dict[str, Any]], submission_paragraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare text content between sample and submission, including font information.
    
    Args:
        sample_paragraphs: List of paragraph dictionaries from sample
        submission_paragraphs: List of paragraph dictionaries from submission
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {
            "completely_different_content": False,
            "errors": []
        }
    }
    
    # Get only the text content for initial matching
    sample_texts = [p.get("text", "") for p in sample_paragraphs]
    submission_texts = [p.get("text", "") for p in submission_paragraphs]
    
    # Check if total paragraph count matches
    if len(sample_texts) != len(submission_texts):
        result["matches"] = False
        result["differences"]["paragraph_count"] = {
            "expected": len(sample_texts),
            "actual": len(submission_texts)
        }
    
    # Track which submission paragraphs have been matched
    matched_pairs = []  # List of (sample_idx, submission_idx, similarity) tuples
    matched_submission_indices = set()
    
    # For each sample paragraph, find the best matching submission paragraph
    for i, sample_text in enumerate(sample_texts):
        best_match_index = -1
        best_similarity = 0
        
        # Find the best matching paragraph in submission
        for j, submission_text in enumerate(submission_texts):
            if j in matched_submission_indices:
                continue  # Skip already matched paragraphs
                
            similarity = calculate_text_similarity(sample_text, submission_text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = j
        
        # Check if a match was found
        if best_match_index >= 0:
            matched_submission_indices.add(best_match_index)
            matched_pairs.append((i, best_match_index, best_similarity))
        else:
            # No match found at all
            result["matches"] = False
            result["differences"]["errors"].append({
                "type": "missing_paragraph",
                "sample_index": i,
                "text": sample_text
            })
    
    # Process matched pairs based on similarity score
    has_high_similarity = False
    for i, j, similarity in matched_pairs:
        sample_text = sample_texts[i]
        submission_text = submission_texts[j]
        
        if abs(similarity - 1.0) < 1e-6:
            has_high_similarity = True
            
            # If similarity = 1.0, check font information
            sample_para = sample_paragraphs[i]
            submission_para = submission_paragraphs[j]
            
            # Compare font name
            if sample_para["font_name"] != submission_para["font_name"]:
                result["matches"] = False
                result["differences"]["errors"].append({
                    "type": "font_name_mismatch",
                    "sample_index": i,
                    "submission_index": j,
                    "sample_text": sample_text,
                    "submission_text": submission_text,
                    "expected": sample_para["font_name"],
                    "actual": submission_para["font_name"],
                    "similarity": round(similarity, 2)
                })
            
            # Compare font size (with tolerance)
            if sample_para["font_size"] is not None and submission_para["font_size"] is not None:
                # Allow for small differences in font size (0.5 pt)
                if abs(sample_para["font_size"] - submission_para["font_size"]) > 0.5:
                    result["matches"] = False
                    result["differences"]["errors"].append({
                        "type": "font_size_mismatch",
                        "sample_index": i,
                        "submission_index": j,
                        "sample_text": sample_text,
                        "submission_text": submission_text,
                        "expected": sample_para["font_size"],
                        "actual": submission_para["font_size"],
                        "similarity": round(similarity, 2)
                    })

            # Compare line_spacing
            if sample_para["line_spacing"] != submission_para["line_spacing"]:
                result["matches"] = False
                result["differences"]["errors"].append({
                    "type": "line_spacing_mismatch",
                    "sample_index": i,
                    "submission_index": j,
                    "sample_text": sample_text,
                    "submission_text": submission_text,
                    "expected": sample_para["line_spacing"],
                    "actual": submission_para["line_spacing"],
                    "similarity": round(similarity, 2)
                })
        
        elif similarity >= 0.6 and similarity <0.98:
            # Partially similar content (0.4 <= similarity <= 0.7)
            result["matches"] = False
            
            # Use difflib to get differences
            differ = difflib.Differ()
            diff = list(differ.compare(sample_text.split(), submission_text.split()))

            added = []
            removed = []
            common = []

            for item in diff:
                word = item[2:]
                cleaned_word = clean_token(word)
                if not cleaned_word:  # Bỏ qua nếu sau khi làm sạch không còn gì
                    continue

                if item.startswith("+ "):
                    added.append(cleaned_word)
                elif item.startswith("- "):
                    removed.append(cleaned_word)
                elif item.startswith("  "):
                    common.append(cleaned_word)

            diff_result = {
                "common": " ".join(common),
                "removed": " ".join(removed),
                "added": " ".join(added)
            }
            result["differences"]["errors"].append({
                "type": "partially_similar_content",
                "sample_index": i,
                "submission_index": j,
                "sample_text": sample_text,
                "submission_text": submission_text,
                "similarity": round(similarity, 2),
                "diff": diff_result
            })
            # Compare font name
            if sample_para["font_name"] != submission_para["font_name"]:
                result["matches"] = False
                result["differences"]["errors"].append({
                    "type": "font_name_mismatch",
                    "sample_index": i,
                    "submission_index": j,
                    "sample_text": sample_text,
                    "submission_text": submission_text,
                    "expected": sample_para["font_name"],
                    "actual": submission_para["font_name"],
                    "similarity": round(similarity, 2)
                })
            
            # Compare font size (with tolerance)
            if sample_para["font_size"] is not None and submission_para["font_size"] is not None:
                # Allow for small differences in font size (0.5 pt)
                if abs(sample_para["font_size"] - submission_para["font_size"]) > 0.5:
                    result["matches"] = False
                    result["differences"]["errors"].append({
                        "type": "font_size_mismatch",
                        "sample_index": i,
                        "submission_index": j,
                        "sample_text": sample_text,
                        "submission_text": submission_text,
                        "expected": sample_para["font_size"],
                        "actual": submission_para["font_size"],
                        "similarity": round(similarity, 2)
                    })

            # Compare line_spacing
            if sample_para["line_spacing"] != submission_para["line_spacing"]:
                result["matches"] = False
                result["differences"]["errors"].append({
                    "type": "line_spacing_mismatch",
                    "sample_index": i,
                    "submission_index": j,
                    "sample_text": sample_text,
                    "submission_text": submission_text,
                    "expected": sample_para["line_spacing"],
                    "actual": submission_para["line_spacing"],
                    "similarity": round(similarity, 2)
                })
        
        elif similarity <= 0.6:
            # Similarity < 0.6 - consider as different content
            result["matches"] = False
            result["differences"]["errors"].append({
                "type": "different_content",
                "sample_index": i,
                "submission_index": j,
                "sample_text": sample_text,
                "submission_text": submission_text,
                "similarity": round(similarity, 2)
            })
    
    # Check for extra paragraphs in submission
    for j, submission_text in enumerate(submission_texts):
        if j not in matched_submission_indices:
            result["matches"] = False
            result["differences"]["errors"].append({
                "type": "extra_paragraph",
                "submission_index": j,
                "text": submission_text
            })
    
    # If no high similarity matches found, mark as completely different
    if not has_high_similarity and len(sample_texts) > 0 and len(submission_texts) > 0:
        result["matches"] = False
        result["differences"]["completely_different_content"] = True
        result["differences"]["errors"].insert(0, {
            "type": "completely_different_content",
            "message": "No paragraphs with similarity > 0.7 found between documents"
        })
    
    return result

def compare_tables(sample_tables: List[List[List[str]]], submission_tables: List[List[List[str]]]) -> Dict[str, Any]:
    """
    Compare tables between sample and submission.
    
    Args:
        sample_tables: List of tables extracted from sample DOCX
        submission_tables: List of tables extracted from submission DOCX
        
    Returns:
        Dictionary with comparison results
    """
    result = {
        "matches": True,
        "differences": {}
    }

    # So sánh số lượng bảng
    if len(sample_tables) != len(submission_tables)-1:
        result["matches"] = False
        result["differences"]["table_count"] = {
            "expected": len(sample_tables),
            "actual": len(submission_tables)
        }

    min_tables = min(len(sample_tables), len(submission_tables))
    for i in range(min_tables):
        sample_table = sample_tables[i]
        submission_table = submission_tables[i+1]

        # So sánh số hàng
        if len(sample_table) != len(submission_table):
            result["matches"] = False
            result["differences"][f"table_{i+1}_row_count"] = {
                "expected": len(sample_table),
                "actual": len(submission_table)
            }

        # So sánh từng dòng
        min_rows = min(len(sample_table), len(submission_table))
        for r in range(min_rows):
            sample_row = sample_table[r]
            submission_row = submission_table[r]

            # So sánh số ô trong mỗi dòng
            if len(sample_row) != len(submission_row):
                result["matches"] = False
                result["differences"][f"table_{i+1}_row_{r+1}_cell_count"] = {
                    "expected": len(sample_row),
                    "actual": len(submission_row)
                }

            # So sánh nội dung ô
            min_cells = min(len(sample_row), len(submission_row))
            for c in range(min_cells):
                sample_cell = sample_row[c].strip()
                submission_cell = submission_row[c].strip()
                if sample_cell != submission_cell:
                    result["matches"] = False
                    result["differences"][f"table_{i+1}_row_{r+1}_cell_{c+1}_text"] = {
                        "expected": sample_cell,
                        "actual": submission_cell
                    }

    return result

def extract_document_properties(docx_file: str) -> Dict[str, Any]:
    """
    Extract all needed properties from a .docx file.
    
    Args:
        docx_file: Path to the .docx file
        
    Returns:
        Dictionary with extracted properties
    """
    properties = {}
    
    try:
        # Extract margins
        print(f"Extracting margins from {docx_file}")
        properties["margins"] = get_docx_margins(docx_file)
        
        # Extract text content
        print(f"Extracting text content from {docx_file}")
        properties["paragraphs"] = extract_text_content(docx_file)
        
        # Open the .docx file as a zip archive
        print(f"Opening {docx_file} as ZIP archive")
        with zipfile.ZipFile(docx_file) as zip_ref:
            # Extract relationships for image info
            print("Extracting relationships")
            relationships = extract_relationships(docx_file)
            
            # Extract the main document.xml file
            print("Extracting document.xml")
            xml_content = zip_ref.read('word/document.xml')
            root = etree.fromstring(xml_content)
            
            # Extract each property
            print("Extracting WordArt")
            properties["wordart"] = extract_wordart(root)
            print("Extracting columns")
            properties["columns"] = extract_columns(root)
            print("Extracting drop caps")
            properties["drop_caps"] = extract_drop_caps(root)
            print("Extracting tables")
            properties["tables"] = extract_tables(root)
            print("Extracting symbols")
            properties["symbols"] = extract_symbols(root)
            print("Extracting images")
            properties["images"] = extract_image_info(root, relationships)
            
    except Exception as e:
        import traceback
        print(f"Error extracting properties from {docx_file}: {e}")
        traceback.print_exc()
    
    return properties

def evaluate_submission(sample_file: str, submission_file: str) -> Dict[str, Any]:
    """
    Evaluate a student's .docx submission against a sample file.
    
    Args:
        sample_file: Path to the sample .docx file
        submission_file: Path to the student's submission .docx file
        
    Returns:
        Dictionary with evaluation results
    """
    # Extract properties from both documents
    print(f"Extracting properties from sample file: {sample_file}")
    sample_properties = extract_document_properties(sample_file)
    
    print(f"Extracting properties from submission file: {submission_file}")
    submission_properties = extract_document_properties(submission_file)
    

    # Compare properties
    evaluation = {
        "overall_match": True,
        "formatting": {
            "margins": compare_margins(sample_properties["margins"], submission_properties["margins"]),
            "columns": compare_columns(sample_properties["columns"], submission_properties["columns"]),
            "wordart": compare_wordart(sample_properties["wordart"], submission_properties["wordart"]),
            "drop_caps": compare_drop_caps(sample_properties["drop_caps"], submission_properties["drop_caps"]),
            "symbols": compare_symbols(sample_properties["symbols"], submission_properties["symbols"]),
            "images": compare_images(sample_properties["images"], submission_properties["images"]),
            "tables": {
                "matches": True,
                "differences": {}
            }
        },
        "content": compare_content(sample_properties["paragraphs"], submission_properties["paragraphs"]),
        "information_studen": []
    }
    if submission_properties["tables"]:
        evaluation["information_studen"] = submission_properties["tables"][0];
    
    if sample_properties["tables"]:
        evaluation["formatting"]["tables"] = compare_tables(sample_properties["tables"], submission_properties["tables"])
    
    # Determine overall match
    for category, results in evaluation["formatting"].items():
        if not results["matches"]:
            evaluation["overall_match"] = False
            break
    
    if not evaluation["content"]["matches"]:
        evaluation["overall_match"] = False
    
    return evaluation

def format_evaluation_report(evaluation: Dict[str, Any]) -> str:
    """
    Format the evaluation report in a readable form.
    
    Args:
        evaluation: Evaluation results dictionary
        
    Returns:
        Formatted report as a string
    """
    report = []
    report.append("\n" + "="*60)
    report.append("DOCX SUBMISSION EVALUATION REPORT")
    report.append("="*60)

    report.append("STUDEN'S INFORMATION:")
    if evaluation['information_studen']:
        for row_idx, row in enumerate(evaluation['information_studen'], 1):
            row_str = " ".join(cell if cell else "[Empty]" for cell in row)
            report.append(f"  {row_str}")
    else:
        report.append("No information found.")
    report.append("="*60)
    
    # Overall result
    report.append(f"\nOVERALL RESULT: {'PASS' if evaluation['overall_match'] else 'FAIL'}")
    
    # Formatting section
    report.append("\n" + "-"*60)
    report.append("FORMATTING EVALUATION")
    report.append("-"*60)
    
    # Process each formatting category
    for category, results in evaluation["formatting"].items():
        category_name = category.replace("_", " ").title()
        report.append(f"\n{category_name}: {'MATCH' if results['matches'] else 'MISMATCH'}")
        
        if not results["matches"] and results.get("differences"):
            for diff_key, diff_value in results["differences"].items():
                report.append(f"  - {diff_key.replace('_', ' ').title()}:")
                if isinstance(diff_value, dict):
                    for k, v in diff_value.items():
                        report.append(f"      {k}: {v}")
                else:
                    report.append(f"      {diff_value}")
    
    # Content section
    report.append("\n" + "-"*60)
    report.append("CONTENT EVALUATION")
    report.append("-"*60)
    report.append(f"\nContent: {'MATCH' if evaluation['content']['matches'] else 'MISMATCH'}")
    
    if not evaluation["content"]["matches"] and evaluation["content"].get("differences"):
        content_diffs = evaluation["content"]["differences"]
        
        # Check if completely different content
        if content_diffs.get("completely_different_content", False):
            report.append("\n⚠️ COMPLETELY DIFFERENT CONTENT DETECTED ⚠️")
        
        # Paragraph count difference
        if "paragraph_count" in content_diffs:
            report.append(f"\nParagraph count:")
            report.append(f"  - Expected: {content_diffs['paragraph_count']['expected']}")
            report.append(f"  - Actual: {content_diffs['paragraph_count']['actual']}")
        
        # Process all errors
        if "errors" in content_diffs and content_diffs["errors"]:
            # Group errors by type
            errors_by_type = {}
            for error in content_diffs["errors"]:
                error_type = error.get("type", "unknown")
                if error_type not in errors_by_type:
                    errors_by_type[error_type] = []
                errors_by_type[error_type].append(error)
            
            # Report each error type
            for error_type, errors in errors_by_type.items():
                type_title = error_type.replace("_", " ").title()
                report.append(f"\n{type_title} ({len(errors)}):")
                
                # Format based on error type
                if error_type == "completely_different_content":
                    for error in errors:
                        report.append(f"  - {error.get('message', 'Documents have completely different content')}")
                
                elif error_type == "missing_paragraph":
                    for error in errors:
                        report.append(f"  - Sample paragraph {error.get('sample_index', '?')+1}:")
                        report.append(f"    Text: {error.get('text', '')[:100]}" + 
                                    ("..." if len(error.get('text', '')) > 100 else ""))
                
                elif error_type == "extra_paragraph":
                    for error in errors:
                        report.append(f"  - Submission paragraph {error.get('submission_index', '?')+1}:")
                        report.append(f"    Text: {error.get('text', '')[:100]}" + 
                                    ("..." if len(error.get('text', '')) > 100 else ""))
                
                elif error_type in ["font_name_mismatch", "font_size_mismatch"]:
                    for error in errors:
                        report.append(f"  - Paragraph with similarity {error.get('similarity', 0):.2f}:")
                        report.append(f"    Sample: {error.get('sample_text', '')[:50]}" + 
                                    ("..." if len(error.get('sample_text', '')) > 50 else ""))
                        if error.get('similarity', 0) < 0.99:
                            report.append(f"    submission: {error.get('submission_text', '')[:50]}" + 
                                        ("..." if len(error.get('submission_text', '')) > 50 else ""))
                        report.append(f"    Expected {error_type.split('_')[1]}: {error.get('expected', '?')}")
                        report.append(f"    Actual {error_type.split('_')[1]}: {error.get('actual', '?')}")
                        
                elif error_type == "line_spacing_mismatch":
                    for error in errors:
                        report.append(f"  - Paragraph with similarity {error.get('similarity', 0):.2f}:")
                        report.append(f"    Sample: {error.get('sample_text', '')[:50]}" + 
                                    ("..." if len(error.get('sample_text', '')) > 50 else ""))
                        if error.get('similarity', 0) < 0.99:
                            report.append(f"    submission: {error.get('submission_text', '')[:50]}" + 
                                        ("..." if len(error.get('submission_text', '')) > 50 else ""))
                        report.append(f"    Expected spacing: {error.get('expected', '?')}")
                        report.append(f"    Actual spacing: {error.get('actual', '?')}")

                elif error_type == "partially_similar_content":
                    for error in errors:
                        report.append(f"  - Similarity score: {error.get('similarity', 0):.2f}")
                        report.append(f"    Sample ({error.get('sample_index', '?')+1}): {error.get('sample_text', '')[:50]}" + 
                                    ("..." if len(error.get('sample_text', '')) > 50 else ""))
                        report.append(f"    Submission ({error.get('submission_index', '?')+1}): {error.get('submission_text', '')[:50]}" + 
                                    ("..." if len(error.get('submission_text', '')) > 50 else ""))
                        if "diff" in error:
                            report.append(f"    Differences: {error['diff']}")
                
                elif error_type == "different_content":
                    for error in errors:
                        report.append(f"  - Similarity score: {error.get('similarity', 0):.2f}")
                        report.append(f"    Sample ({error.get('sample_index', '?')+1}): {error.get('sample_text', '')[:50]}" + 
                                    ("..." if len(error.get('sample_text', '')) > 50 else ""))
                        report.append(f"    Submission ({error.get('submission_index', '?')+1}): {error.get('submission_text', '')[:50]}" + 
                                    ("..." if len(error.get('submission_text', '')) > 50 else ""))
                
                else:
                    # Generic error format for unknown types
                    for error in errors:
                        report.append(f"  - {str(error)}")
    
    report.append("\n" + "="*60)
    return "\n".join(report)

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_docx_submission.py <sample_file.docx> <submission_file.docx>")
        sys.exit(1)
    
    sample_file = sys.argv[1]
    submission_file = sys.argv[2]

    # print(f"content 1: {extract_text_content(sample_file)[1]}")
    # print(f"content 2: {extract_text_content(submission_file)[1]}")

    # emb1 = 'Trà Cổ nổi tiếng với đường bờ biển dài nhất Việt Nam - hơn 17km.'
    # emb2 = 'Trà Cổ không nổi tiếng với đường bờ biển dài nhì Việt Nam - hơn 17km.'
    # print(f"do giong nhau: {calculate_text_similarity(emb1, emb2)}")
    print(f"Sample file path: {sample_file}")
    print(f"Submission file path: {submission_file}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if files exist
    if not os.path.exists(sample_file):
        print(f"Error: Sample file '{sample_file}' does not exist")
        sys.exit(1)
    
    if not os.path.exists(submission_file):
        print(f"Error: Submission file '{submission_file}' does not exist")
        sys.exit(1)
    
    print(f"Both files exist and are accessible")
    
    # Evaluate submission
    try:
        evaluation = evaluate_submission(sample_file, submission_file)
        
        # Format and print report
        report = format_evaluation_report(evaluation)
        print(report)
        
        # Optionally save report to a file - use UTF-8 encoding to handle non-ASCII characters
        with open("evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        # Also save JSON version for programmatic use
        with open("evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed evaluation reports saved to 'evaluation_report.txt' and 'evaluation_report.json'")
    except Exception as e:
        import traceback
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 