#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
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

from app.services.scorer.docx_properties_extractor import (
    extract_drop_caps,
    get_font_info,
    extract_wordart,
    extract_columns,
    extract_symbols,
    extract_image_info,
    extract_relationships,
    get_docx_margins,
    extract_tables,
    extract_images_as_pil,
    namespaces
)

from app.utils.compare_image_clip_similarity import compare_clip_similarity

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
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PHOBERT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "fine_tuned_phobert"))
        # Load the tokenizer from the local directory
        tokenizer_path = os.path.join(PHOBERT_DIR, "tokenizer")
        model_path = os.path.join(PHOBERT_DIR, "phobert_similarity_model_v2.pth")
        
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
    Extract text content from a .docx file paragraph by paragraph, excluding drop cap,
    and including font and line spacing info.
    """
    try:
        import docx
        doc = docx.Document(docx_file)
        paragraphs = []

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue

            # Bỏ qua đoạn nghi là drop cap (1-2 ký tự viết hoa)
            if len(text) <= 2 and text.isupper():
                continue

            font_name = None
            font_size = None
            line_spacing = None

            if p.runs:
                run = p.runs[0]
                if hasattr(run, 'font') and hasattr(run.font, 'name'):
                    font_name = run.font.name

                for run in p.runs:
                    if run.font.size is not None:
                        try:
                            font_size = run.font.size / 12700
                        except (TypeError, ValueError):
                            font_size = None
                        break

            if hasattr(p, 'paragraph_format') and hasattr(p.paragraph_format, 'line_spacing'):
                line_spacing = p.paragraph_format.line_spacing

            para_info = {
                "text": text,
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
        Dictionary with match result, differences, and similarity score
    """
    result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    penalty_per_mismatch = 0.25

    for key in ["top", "bottom", "left", "right"]:
        sample_value = sample_margins.get(key, 0)
        submission_value = submission_margins.get(key, 0)

        if abs(sample_value - submission_value) > tolerance:
            result["matches"] = False
            result["similarity_score"] -= penalty_per_mismatch
            result["differences"][key] = {
                "expected": sample_value,
                "actual": submission_value,
                "difference": round(abs(sample_value - submission_value), 3)
            }

    result["similarity_score"] = max(0.0, round(result["similarity_score"], 4))
    return result


def compare_columns(sample_columns: List[Dict], submission_columns: List[Dict]) -> Dict[str, Any]:
    """
    Compare column settings between sample and submission.

    Returns:
        Dictionary with match result, differences, and similarity score.
    """
    result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    if bool(sample_columns) != bool(submission_columns):
        result["matches"] = False
        result["similarity_score"] = 0.0
        result["differences"]["has_columns"] = {
            "expected": bool(sample_columns),
            "actual": bool(submission_columns)
        }
        return result

    if not sample_columns and not submission_columns:
        return result

    if len(sample_columns) != len(submission_columns):
        result["matches"] = False
        result["differences"]["num_sections_with_columns"] = {
            "expected": len(sample_columns),
            "actual": len(submission_columns)
        }

    min_sections = min(len(sample_columns), len(submission_columns))

    for i in range(min_sections):
        sample_section = sample_columns[i]
        submission_section = submission_columns[i]

        expected_cols = sample_section.get("num_columns", 1)
        actual_cols = submission_section.get("num_columns", 1)

        # Nếu số lượng cột khác → similarity = 0.0 và kết thúc
        if expected_cols != actual_cols:
            result["matches"] = False
            result["similarity_score"] = 0.0
            result["differences"][f"section_{i+1}_num_columns"] = {
                "expected": expected_cols,
                "actual": actual_cols
            }
            return result

        # So sánh separator
        expected_sep = sample_section.get("has_separator", False)
        actual_sep = submission_section.get("has_separator", False)
        if expected_sep != actual_sep:
            result["matches"] = False
            result["similarity_score"] -= 0.15
            result["differences"][f"section_{i+1}_has_separator"] = {
                "expected": expected_sep,
                "actual": actual_sep
            }

        # So sánh khoảng cách giữa các cột
        expected_space = str(sample_section.get("space_between", "425"))
        actual_space = str(submission_section.get("space_between", "425"))
        if expected_space != actual_space:
            result["matches"] = False
            result["similarity_score"] -= 0.15
            result["differences"][f"section_{i+1}_space_between"] = {
                "expected": expected_space,
                "actual": actual_space
            }

    # Đảm bảo không âm
    result["similarity_score"] = max(0.0, round(result["similarity_score"], 4))
    return result


def compare_wordart(sample_wordart: List[Dict], submission_wordart: List[Dict]) -> Dict[str, Any]:
    result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    sample_count = len(sample_wordart)
    submission_count = len(submission_wordart)

    matched_sample_indices = set()
    matched_pairs = []

    if sample_count == submission_count:
        matched_pairs = [(i, i) for i in range(sample_count)]
    else:
        for sub_idx, submission in enumerate(submission_wordart):
            best_match_idx = -1
            best_score = 0.0
            sub_text = " ".join(seg.get("text", "") for seg in submission.get("text_segments", []))
            best_sample_text = ""

            for samp_idx, sample in enumerate(sample_wordart):
                if samp_idx in matched_sample_indices:
                    continue
                samp_text = " ".join(seg.get("text", "") for seg in sample.get("text_segments", []))
                score = calculate_text_similarity(samp_text, sub_text)
                if score > best_score:
                    best_score = score
                    best_match_idx = samp_idx
                    best_sample_text = samp_text

            if best_match_idx != -1 and best_score >= 0.9:
                matched_sample_indices.add(best_match_idx)
                matched_pairs.append((best_match_idx, sub_idx))
            else:
                result["matches"] = False
                result["similarity_score"] -= 0.1 / max(sample_count, 1)
                result["differences"][f"unmatched_submission_wordart_{sub_idx+1}"] = {
                    "submission_text": sub_text,
                    "sample_text": best_sample_text,
                    "best_similarity": round(best_score, 2)
                }

    for sample_idx, submission_idx in matched_pairs:
        sample = sample_wordart[sample_idx]
        submission = submission_wordart[submission_idx]

        sample_segments = sample.get("text_segments", [])
        submission_segments = submission.get("text_segments", [])
        segment_penalty = 0.1 / max(len(sample_segments), 1) / max(sample_count, 1)

        for j in range(min(len(sample_segments), len(submission_segments))):
            seg1 = sample_segments[j]
            seg2 = submission_segments[j]

            text1 = seg1.get("text", "")
            text2 = seg2.get("text", "")
            sim = calculate_text_similarity(text1, text2)
            if sim < 0.99:
                result["matches"] = False
                result["similarity_score"] -= segment_penalty
                result["differences"][f"wordart_{sample_idx+1}_segment_{j+1}_text"] = {
                    "expected": text1,
                    "actual": text2,
                    "similarity": round(sim, 2)
                }

            for key in [
                "font_name", "font_size", "text_fill", "text_outline",
                "glow", "shadow", "reflection", "bold", "italic", "underline", "color"
            ]:
                if seg1.get(key) != seg2.get(key):
                    result["matches"] = False
                    result["similarity_score"] -= segment_penalty
                    result["differences"][f"wordart_{sample_idx+1}_segment_{j+1}_{key}"] = {
                        "expected": seg1.get(key),
                        "actual": seg2.get(key)
                    }

        sample_shape = sample.get("shape_info", {})
        submission_shape = submission.get("shape_info", {})

        if sample_shape.get("geometry") != submission_shape.get("geometry"):
            result["matches"] = False
            result["similarity_score"] -= 0.1 / max(sample_count, 1)
            result["differences"][f"wordart_{sample_idx+1}_shape_info_geometry"] = {
                "expected": sample_shape.get("geometry"),
                "actual": submission_shape.get("geometry")
            }

        if sample_shape.get("effects", {}).get("transform") != submission_shape.get("effects", {}).get("transform"):
            result["matches"] = False
            result["similarity_score"] -= 0.1 / max(sample_count, 1)
            result["differences"][f"wordart_{sample_idx+1}_shape_info_transform"] = {
                "expected": sample_shape.get("effects", {}).get("transform"),
                "actual": submission_shape.get("effects", {}).get("transform")
            }

    if sample_count != submission_count:
        result["matches"] = False
        result["similarity_score"] -= abs(sample_count - submission_count) / max(sample_count, 1)
        result["differences"]["count_mismatch"] = {
            "expected": sample_count,
            "actual": submission_count
        }

    result["similarity_score"] = round(max(result["similarity_score"], 0.0), 4)
    return result



def compare_drop_caps(sample_caps: List[Dict], submission_caps: List[Dict]) -> Dict[str, Any]:
    """
    So sánh drop caps giữa 2 file .docx một cách tối ưu bằng ghép cặp tốt nhất.
    """
    result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    sample_count = len(sample_caps)
    submission_count = len(submission_caps)

    if (sample_count > 0) != (submission_count > 0):
        result["matches"] = False
        result["differences"]["has_drop_caps"] = {
            "expected": sample_count > 0,
            "actual": submission_count > 0
        }
        result["similarity_score"] = 0.0
        return result

    if sample_count == 0 and submission_count == 0:
        return result

    similarity = 1.0

    # Trừ điểm nếu số lượng khác nhau
    count_penalty = abs(sample_count - submission_count) / max(1, sample_count)
    if sample_count != submission_count:
        result["matches"] = False
        result["differences"]["count"] = {
            "expected": sample_count,
            "actual": submission_count
        }
    similarity -= count_penalty

    # Đánh dấu drop caps đã ghép
    matched_samples = set()
    matched_submissions = set()
    content_penalty = 0.0

    for j, submission_cap in enumerate(submission_caps):
        best_score = float('inf')
        best_i = -1
        best_diffs = {}

        for i, sample_cap in enumerate(sample_caps):
            if i in matched_samples:
                continue

            diff_score = 0.0
            local_diffs = {}

            if sample_cap["char"] != submission_cap["char"]:
                diff_score += 0.5 / max(1, sample_count)
                local_diffs["char"] = {
                    "expected": sample_cap["char"],
                    "actual": submission_cap["char"]
                }

            if sample_cap["type"] != submission_cap["type"]:
                diff_score += 0.15 / max(1, sample_count)
                local_diffs["type"] = {
                    "expected": sample_cap["type"],
                    "actual": submission_cap["type"]
                }

            if sample_cap["lines"] != submission_cap["lines"]:
                diff_score += 0.15 / max(1, sample_count)
                local_diffs["lines"] = {
                    "expected": sample_cap["lines"],
                    "actual": submission_cap["lines"]
                }

            if diff_score < best_score:
                best_score = diff_score
                best_i = i
                best_diffs = local_diffs

        if best_i >= 0:
            matched_samples.add(best_i)
            matched_submissions.add(j)
            if best_score > 0:
                result["matches"] = False
                for k, v in best_diffs.items():
                    result["differences"][f"drop_cap_sample_{best_i+1}_{k}"] = v
                content_penalty += best_score

    # Những drop cap trong sample không được ghép
    for i in range(sample_count):
        if i not in matched_samples:
            result["matches"] = False
            result["differences"][f"drop_cap_{i+1}_missing"] = "Không tìm thấy tương ứng"

    similarity -= content_penalty
    result["similarity_score"] = max(0.0, round(similarity, 2))
    return result


def compare_symbols(sample_symbols: List[Dict], submission_symbols: List[Dict]) -> Dict[str, Any]:
    """
    Compare symbols between sample and submission in detail and calculate similarity score.
    
    Returns:
        Dictionary with match result, differences, and similarity score (0 to 1)
    """
    result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    # Check presence
    if bool(sample_symbols) != bool(submission_symbols):
        result["matches"] = False
        result["similarity_score"] = 0.0
        result["differences"]["has_symbols"] = {
            "expected": bool(sample_symbols),
            "actual": bool(submission_symbols)
        }
        return result

    if not sample_symbols and not submission_symbols:
        return result

    sample_len = len(sample_symbols)
    submission_len = len(submission_symbols)

    # Base penalty for different lengths
    if sample_len != submission_len:
        result["matches"] = False
        length_diff_penalty = abs(sample_len - submission_len) / sample_len
        result["similarity_score"] -= length_diff_penalty
        result["differences"]["symbol_count"] = {
            "expected": sample_len,
            "actual": submission_len
        }

    # Compare symbols one by one
    min_len = min(sample_len, submission_len)
    symbol_differences = []
    per_field_penalty = 0.15 / min_len if min_len > 0 else 0

    for i in range(min_len):
        s_sym = sample_symbols[i]
        sub_sym = submission_symbols[i]
        differences = {}
        penalty = 0.0

        # Compare fields (excluding character)

        for field in ["type", "run_index", "char_offset"]:
            if s_sym.get(field) != sub_sym.get(field):
                differences[field] = {
                    "expected": s_sym.get(field),
                    "actual": sub_sym.get(field)
                }
                penalty += per_field_penalty

        if differences:
            symbol_differences.append({
                "index": i,
                "differences": differences
            })
            result["matches"] = False
            result["similarity_score"] -= penalty

    # Extra symbols beyond min_len
    for i in range(min_len, sample_len):
        symbol_differences.append({
            "index": i,
            "differences": {
                "expected": sample_symbols[i],
                "actual": None
            }
        })
    for i in range(min_len, submission_len):
        symbol_differences.append({
            "index": i,
            "differences": {
                "expected": None,
                "actual": submission_symbols[i]
            }
        })

    if symbol_differences:
        result["differences"]["symbol_details"] = symbol_differences

    # Ensure score stays in [0, 1]
    result["similarity_score"] = max(0.0, round(result["similarity_score"], 4))

    return result


def compare_images(sample_images: List[Dict], submission_images: List[Dict],
                   sample_pil_images: List[Dict], submission_pil_images: List[Dict],
                   tolerance=0.1) -> Dict[str, Any]:
    """
    So sánh thông tin ảnh giữa file mẫu và bài làm, bao gồm cả nội dung ảnh CLIP.
    """
    result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    sample_count = len(sample_images)
    submission_count = len(submission_images)

    if (sample_count == 0) != (submission_count == 0):
        result["matches"] = False
        result["differences"]["has_images"] = {
            "expected": sample_count > 0,
            "actual": submission_count > 0
        }
        result["similarity_score"] = 0.0
        return result

    if sample_count == 0 and submission_count == 0:
        return result

    if sample_count != submission_count:
        result["matches"] = False
        result["differences"]["count"] = {
            "expected": sample_count,
            "actual": submission_count
        }

    similarity = 1.0 - (abs(sample_count - submission_count) / max(1, sample_count))

    def safe_float(val):
        try:
            return float(val)
        except:
            return 0.0

    def safe_int(val):
        try:
            return int(val)
        except:
            return 0

    min_elements = min(sample_count, submission_count)
    for i in range(min_elements):
        sample_img = sample_images[i]
        submission_img = submission_images[i]

        # Compare position_type
        if sample_img["position_type"] != submission_img["position_type"]:
            result["matches"] = False
            result["differences"][f"image_{i+1}_position_type"] = {
                "expected": sample_img["position_type"],
                "actual": submission_img["position_type"]
            }
            similarity -= 0.25

        # Compare dimensions
        sw, sh = safe_float(sample_img["width"]), safe_float(sample_img["height"])
        aw, ah = safe_float(submission_img["width"]), safe_float(submission_img["height"])

        if sw > 0 and aw > 0:
            width_ratio = min(sw, aw) / max(sw, aw)
            if width_ratio < 1 - tolerance:
                result["matches"] = False
                result["differences"][f"image_{i+1}_width"] = {
                    "expected": sw,
                    "actual": aw,
                    "ratio": round(width_ratio, 2)
                }
                similarity -= 0.1

        if sh > 0 and ah > 0:
            height_ratio = min(sh, ah) / max(sh, ah)
            if height_ratio < 1 - tolerance:
                result["matches"] = False
                result["differences"][f"image_{i+1}_height"] = {
                    "expected": sh,
                    "actual": ah,
                    "ratio": round(height_ratio, 2)
                }
                similarity -= 0.1

        # Compare format
        if sample_img["format"] != "unknown" and submission_img["format"] != "unknown":
            if sample_img["format"].lower() != submission_img["format"].lower():
                result["matches"] = False
                result["differences"][f"image_{i+1}_format"] = {
                    "expected": sample_img["format"],
                    "actual": submission_img["format"]
                }
                similarity -= 0.1

        # Compare offset if anchored
        if sample_img["position_type"] == "anchored" and submission_img["position_type"] == "anchored":
            sample_pos = sample_img.get("position_details", {})
            submission_pos = submission_img.get("position_details", {})

            for axis in ["horizontal_position", "vertical_position"]:
                s_axis = sample_pos.get(axis, {})
                a_axis = submission_pos.get(axis, {})

                s_offset = safe_int(s_axis.get("offset", "0"))
                a_offset = safe_int(a_axis.get("offset", "0"))

                if s_offset > 0 and a_offset > 0:
                    diff = abs(s_offset - a_offset)
                    max_offset = max(s_offset, a_offset)
                    offset_ratio = diff / max_offset if max_offset else 0
                    if offset_ratio > tolerance:
                        result["matches"] = False
                        result["differences"][f"image_{i+1}_{axis}_offset"] = {
                            "expected": s_offset,
                            "actual": a_offset,
                            "difference": diff,
                            "ratio": round(offset_ratio, 2)
                        }
                        similarity -= 0.1

        # ✅ So sánh nội dung ảnh bằng CLIP nếu cả 2 ảnh hợp lệ
        if i < len(sample_pil_images) and i < len(submission_pil_images):
            img1 = sample_pil_images[i]['image']
            img2 = submission_pil_images[i]['image']
            sim_score = compare_clip_similarity(img1, img2)

            if sim_score < 0.5:
                result["matches"] = False
                result["differences"][f"image_{i+1}_clip_similarity"] = {
                    "score": round(sim_score, 4),
                    "note": "Ảnh khác nội dung đáng kể"
                }
                similarity -= 0.2

    result["similarity_score"] = max(0.0, round(similarity, 2))
    return result


def clean_token(token):
    return re.sub(r'[^\wÀ-ỹ]', '', token)  # Giữ lại chữ cái tiếng Việt có dấu

def compare_font_properties(
    result: Dict[str, Any],
    sample_para: Dict[str, Any],
    submission_para: Dict[str, Any],
    sample_idx: int,
    submission_idx: int,
    penalty_tracker: Dict[str, float],
    paragraph_count: int
) -> None:
    for key in ["font_name", "font_size", "line_spacing"]:
        sample_val = sample_para.get(key)
        submission_val = submission_para.get(key)

        if sample_val != submission_val:
            result["matches"] = False
            error_detail = {
                "type": "font_difference",
                "property": key,
                "sample_index": sample_idx,
                "submission_index": submission_idx,
                "expected": sample_val,
                "actual": submission_val,
                "sample_text": sample_para.get("text", ""),
                "submission_text": submission_para.get("text", "")
            }
            result["differences"]["errors"].append(error_detail)

            # Giảm điểm theo lỗi
            penalty_tracker["font"] += 0.15 / paragraph_count

def compare_content(sample_paragraphs: List[Dict[str, Any]], submission_paragraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "matches": True,
        "differences": {
            "completely_different_content": False,
            "errors": []
        },
        "similarity_score": 1.0
    }

    sample_texts = [p.get("text", "") for p in sample_paragraphs]
    submission_texts = [p.get("text", "") for p in submission_paragraphs]

    sample_count = len(sample_texts)
    submission_count = len(submission_texts)

    # Trừ điểm nếu số lượng đoạn văn khác nhau
    if sample_count != submission_count:
        result["matches"] = False
        result["differences"]["paragraph_count"] = {
            "expected": sample_count,
            "actual": submission_count
        }
        penalty = abs(sample_count - submission_count) / max(submission_count, 1)
        result["similarity_score"] -= penalty

    matched_pairs = []
    matched_submission_indices = set()
    penalty_tracker = {"font": 0.0, "content": 0.0}

    for i, sample_text in enumerate(sample_texts):
        best_match_index = -1
        best_similarity = 0
        for j, submission_text in enumerate(submission_texts):
            if j in matched_submission_indices:
                continue
            similarity = calculate_text_similarity(sample_text, submission_text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = j
        if best_match_index >= 0:
            matched_submission_indices.add(best_match_index)
            matched_pairs.append((i, best_match_index, best_similarity))
        else:
            result["matches"] = False
            result["differences"]["errors"].append({
                "type": "missing_paragraph",
                "sample_index": i,
                "text": sample_text
            })

    has_high_similarity = False
    for i, j, similarity in matched_pairs:
        sample_para = sample_paragraphs[i]
        submission_para = submission_paragraphs[j]
        sample_text = sample_texts[i]
        submission_text = submission_texts[j]

        compare_font_properties(
            result, sample_para, submission_para, i, j,
            penalty_tracker, max(sample_count, 1)
        )

        if abs(similarity - 1.0) < 1e-6:
            has_high_similarity = True
        else:
            if similarity <= 0.6:
                result["matches"] = False
                result["differences"]["errors"].append({
                    "type": "different_content",
                    "sample_index": i,
                    "submission_index": j,
                    "sample_text": sample_text,
                    "submission_text": submission_text,
                    "similarity": round(similarity, 2)
                })
                penalty_tracker["content"] += 0.55 / max(sample_count, 1)
            else:
                has_high_similarity = True
                result["matches"] = False
                differ = difflib.Differ()
                diff = list(differ.compare(sample_text.split(), submission_text.split()))

                added, removed, common = [], [], []
                for item in diff:
                    word = item[2:]
                    cleaned_word = clean_token(word)
                    if not cleaned_word:
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

                sample_len = len(sample_text.strip().split())
                common_len = len(common)
                if sample_len > 0:
                    penalty_tracker["content"] += (common_len / sample_len) / max(sample_count, 1) * 0.55

    for j, submission_text in enumerate(submission_texts):
        if j not in matched_submission_indices:
            result["matches"] = False
            result["differences"]["errors"].append({
                "type": "extra_paragraph",
                "submission_index": j,
                "text": submission_text
            })

    if not has_high_similarity and sample_count > 0 and submission_count > 0:
        result["matches"] = False
        result["differences"]["completely_different_content"] = True
        result["differences"]["errors"].insert(0, {
            "type": "completely_different_content",
            "message": "No paragraphs with similarity > 0.7 found between documents"
        })

    result["similarity_score"] -= (penalty_tracker["content"] + penalty_tracker["font"])
    result["similarity_score"] = round(max(result["similarity_score"], 0.0), 4)
    return result


def compare_tables(
    sample_tables: List[List[List[Dict[str, Any]]]],
    submission_tables: List[List[List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    sample_table_count = len(sample_tables)
    submission_table_count = len(submission_tables)

    # Xác định bảng thông tin cá nhân: 6 hàng, mỗi hàng 2 cột
    def is_candidate_info_table(table):
        return len(table) == 6 and all(len(row) == 2 for row in table)

    # Loại bảng thông tin cá nhân nếu có trong bài làm
    submission_main_tables = [
        tbl for tbl in submission_tables if not is_candidate_info_table(tbl)
    ]

    # Nếu số bảng chính không khớp mẫu → báo lỗi và trả điểm 0
    if sample_table_count != len(submission_main_tables):
        result["matches"] = False
        result["differences"]["table_count"] = {
            "expected": sample_table_count,
            "actual": len(submission_main_tables)
        }
        result["similarity_score"] = 0.0
        return result

    # So sánh từng bảng
    for i in range(sample_table_count):
        sample_table = sample_tables[i]
        submission_table = submission_main_tables[i]

        total_cells = sum(len(row) for row in sample_table)
        penalty = 0.0

        # So sánh số dòng
        if len(sample_table) != len(submission_table):
            result["matches"] = False
            result["differences"][f"table_{i+1}_row_count"] = {
                "expected": len(sample_table),
                "actual": len(submission_table)
            }

        min_rows = min(len(sample_table), len(submission_table))
        for r in range(min_rows):
            sample_row = sample_table[r]
            submission_row = submission_table[r]

            # So sánh số ô
            if len(sample_row) != len(submission_row):
                result["matches"] = False
                result["differences"][f"table_{i+1}_row_{r+1}_cell_count"] = {
                    "expected": len(sample_row),
                    "actual": len(submission_row)
                }
                penalty += abs(len(sample_row) - len(submission_row)) / total_cells

            min_cells = min(len(sample_row), len(submission_row))
            for c in range(min_cells):
                sample_cell = sample_row[c]
                submission_cell = submission_row[c]

                diff_key = f"table_{i+1}_row_{r+1}_cell_{c+1}"
                cell_diffs = {}

                # So sánh text không phân biệt hoa thường
                sample_text = sample_cell.get("text", "").strip().lower()
                submission_text = submission_cell.get("text", "").strip().lower()
                if sample_text != submission_text:
                    cell_diffs["text"] = {
                        "expected": sample_cell.get("text"),
                        "actual": submission_cell.get("text")
                    }
                    penalty += 0.2 / total_cells

                # So sánh các thuộc tính định dạng chính
                if sample_cell.get("text_color") != submission_cell.get("text_color"):
                    cell_diffs["text_color"] = {
                        "expected": sample_cell.get("text_color"),
                        "actual": submission_cell.get("text_color")
                    }
                    penalty += 0.2 / total_cells

                if sample_cell.get("background_color") != submission_cell.get("background_color"):
                    cell_diffs["background_color"] = {
                        "expected": sample_cell.get("background_color"),
                        "actual": submission_cell.get("background_color")
                    }
                    penalty += 0.2 / total_cells

                # So sánh các thuộc tính định dạng phụ
                for key in ["is_uppercase", "is_bold", "font_name", "font_size"]:
                    if sample_cell.get(key) != submission_cell.get(key):
                        cell_diffs[key] = {
                            "expected": sample_cell.get(key),
                            "actual": submission_cell.get(key)
                        }
                        penalty += 0.1 / total_cells

                if cell_diffs:
                    result["matches"] = False
                    result["differences"][diff_key] = cell_diffs

            # Trừ điểm nếu thiếu hoặc thừa ô trong dòng
            if len(sample_row) < len(submission_row):
                penalty += (len(submission_row) - len(sample_row)) / total_cells
            elif len(sample_row) > len(submission_row):
                penalty += (len(sample_row) - len(submission_row)) / total_cells

        # Trừ điểm nếu thiếu hoặc thừa dòng
        if len(sample_table) < len(submission_table):
            penalty += sum(len(row) for row in submission_table[min_rows:]) / total_cells
        elif len(sample_table) > len(submission_table):
            penalty += sum(len(row) for row in sample_table[min_rows:]) / total_cells

        # Trừ điểm vào tổng điểm
        result["similarity_score"] -= penalty

    result["similarity_score"] = round(max(result["similarity_score"], 0.0), 4)
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
    """
    # Extract properties from both documents
    print(f"Extracting properties from sample file: {sample_file}")
    sample_properties = extract_document_properties(sample_file)
    
    print(f"Extracting properties from submission file: {submission_file}")
    submission_properties = extract_document_properties(submission_file)

    # Chuẩn bị giá trị mặc định cho ảnh
    image_result = {
        "matches": True,
        "differences": {},
        "similarity_score": 1.0
    }

    # Nếu cả hai file đều có ảnh thì mới trích xuất và so sánh nội dung
    if sample_properties["images"] and submission_properties["images"]:
        print("Extracting PIL images from both files for CLIP comparison...")
        sample_pil_images = extract_images_as_pil(sample_file)
        submission_pil_images = extract_images_as_pil(submission_file)
        
        image_result = compare_images(
            sample_properties["images"],
            submission_properties["images"],
            sample_pil_images,
            submission_pil_images
        )
    else:
        print("Skipping image content comparison: one or both files have no images.")

    # Evaluate all formatting
    evaluation = {
        "overall_match": True,
        "formatting": {
            "margins": compare_margins(sample_properties["margins"], submission_properties["margins"]),
            "columns": compare_columns(sample_properties["columns"], submission_properties["columns"]),
            "wordart": compare_wordart(sample_properties["wordart"], submission_properties["wordart"]),
            "drop_caps": compare_drop_caps(sample_properties["drop_caps"], submission_properties["drop_caps"]),
            "symbols": compare_symbols(sample_properties["symbols"], submission_properties["symbols"]),
            "images": image_result,
            "tables": {
                "matches": True,
                "differences": {}
            }
        },
        "content": compare_content(sample_properties["paragraphs"], submission_properties["paragraphs"]),
        "information_studen": []
    }

    # Tìm bảng thông tin thí sinh trong bài làm
    def is_candidate_info_table(table):
        return len(table) == 6 and all(len(row) == 2 for row in table)

    candidate_info_table = None
    for table in submission_properties["tables"]:
        if is_candidate_info_table(table):
            candidate_info_table = table
            break

    # Gán thông tin nếu có, lấy đúng 4 dòng đầu
    if candidate_info_table:
        try:
            evaluation["information_studen"] = [
                candidate_info_table[0][1]["text"],  # Số báo danh
                candidate_info_table[1][1]["text"],  # Họ và tên
                candidate_info_table[2][1]["text"],  # Ngày sinh
                candidate_info_table[3][1]["text"],
                candidate_info_table[5][1]["text"],  # Đề thi
            ]
        except Exception as e:
            print(f"⚠️ Lỗi khi trích xuất bảng thông tin thí sinh: {e}")
            evaluation["information_studen"] = None
    else:
        evaluation["information_studen"] = None
    
    if sample_properties["tables"]:
        evaluation["formatting"]["tables"] = compare_tables(
            sample_properties["tables"], submission_properties["tables"]
        )

    # Đánh giá tổng thể
    for category, results in evaluation["formatting"].items():
        if not results["matches"]:
            evaluation["overall_match"] = False
            break

    if not evaluation["content"]["matches"]:
        evaluation["overall_match"] = False

    return evaluation

def compute_final_score(evaluation: dict) -> dict:
    """
    Tính toán điểm số cuối cùng dựa vào evaluation report và từng phần similarity score.
    """

    # Phát hiện đề: dựa vào các phần có mặt trong sample
    sample_formatting = evaluation["formatting"]

    has_images = bool(sample_formatting.get("images", {}).get("similarity_score", 0) > 0)
    has_table = bool(sample_formatting.get("tables", {}).get("similarity_score", 0) > 0)

    is_de1 = has_images and not has_table
    is_de2 = has_table and not has_images

    # Mặc định là đề 1 nếu không rõ
    format_type = "de1" if is_de1 else "de2" if is_de2 else "unknown"

    score = 0.0
    detailed_scores = {}

    # Info student (luôn có nếu bảng đầu tiên tồn tại)
    if evaluation["information_studen"]:
        detailed_scores["info_student"] = 1.0
        score += 1.0
    else:
        detailed_scores["info_student"] = 0.0

    if format_type == "de1":
        parts = {
            "content": 2,
            "wordart": 2,
            "margins": 1,
            "symbols": 1,
            "columns": 1,
            "drop_caps": 1,
            "images": 1,
        }
    elif format_type == "de2":
        parts = {
            "content": 2,
            "wordart": 2,
            "margins": 1,
            "drop_caps": 1,
            "tables": 3,
        }
    else:
        # Trường hợp không xác định rõ đề, đánh theo đề 1 mặc định
        parts = {
            "content": 2,
            "wordart": 1,
            "margins": 1,
            "symbols": 1,
            "columns": 1,
            "drop_caps": 1,
            "images": 1,
            "tables": 1,
        }

    for key, weight in parts.items():
        sim_score = 0.0
        if key in evaluation["formatting"]:
            sim_score = evaluation["formatting"][key].get("similarity_score", 1.0)
        elif key == "content":
            sim_score = evaluation["content"].get("similarity_score", 1.0)

        part_score = round(sim_score * weight, 2)
        detailed_scores[key] = part_score
        score += part_score

    return {
        "final_score": round(score, 2),
        "format_type": format_type,
        "detailed_scores": detailed_scores
    }


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
            row_str = " ".join(cell["text"] for cell in row if isinstance(cell, dict) and cell.get("text"))
            if row_str.strip():  # Chỉ thêm dòng nếu có nội dung
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
            if results.get("similarity_score") is not None:
                report.append(f"  Similarity Score: {results['similarity_score']}")
    
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
    report.append(f"Similarity Score: {evaluation['content']['similarity_score']}")
    report.append("\n" + "="*60)
    final_score_info = compute_final_score(evaluation)
    report.append(f" Score: {final_score_info['final_score']}")
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