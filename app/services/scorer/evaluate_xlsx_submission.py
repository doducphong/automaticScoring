import re
import json
from typing import Dict, List, Any
from zipfile import ZipFile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.services.scorer.xlsx_properties_extractor import (
    extract_shared_strings, 
    extract_styles, 
    extract_sheet
)

def expand_sum_range(formula: str) -> str:
    """
    Tìm và mở rộng tất cả các hàm SUM([r1,c]:[r2,c]) thành cộng từng ô.
    Chỉ hỗ trợ dải cùng cột.
    """
    pattern = r'SUM\(\s*(\[\d+,\d+\])\s*:\s*(\[\d+,\d+\])\s*\)'
    matches = re.findall(pattern, formula)

    for start, end in matches:
        r1, c1 = map(int, re.findall(r'\d+', start))
        r2, c2 = map(int, re.findall(r'\d+', end))
        if c1 != c2:
            continue  # bỏ qua nếu không cùng cột
        if r1 > r2:
            r1, r2 = r2, r1  # đảo lại nếu cần

        expanded = '+'.join(f'[{r},{c1}]' for r in range(r1, r2 + 1))
        formula = formula.replace(f'SUM({start}:{end})', expanded)

    return formula

def normalize_commutative_groups(expr: str, op: str) -> str:
    """
    Tìm các cụm toán hạng giao hoán (dùng op) và sắp xếp lại từng cụm theo thứ tự.
    Ví dụ: [3,4]+[2,4]+[4,4] => [2,4]+[3,4]+[4,4]
    """
    pattern = rf'(\[\d+,\d+\](?:\{op}\[\d+,\d+\])+)'  # ví dụ: [a,b]+[c,d]+[e,f]
    matches = re.findall(pattern, expr)
    for match in matches:
        parts = match.split(op)
        sorted_parts = sorted(parts)
        normalized = op.join(sorted_parts)
        expr = expr.replace(match, normalized)
    return expr

def normalize_formula(formula: str) -> str:
    """
    Chuẩn hóa công thức:
    - Mở rộng các SUM(...) thành cộng từng phần tử
    - Sắp xếp các nhóm + và * để tránh lỗi do thứ tự
    """
    if not formula:
        return formula

    formula = expand_sum_range(formula)
    formula = normalize_commutative_groups(formula, '*')
    formula = normalize_commutative_groups(formula, '+')

    return formula

def group_by_table(tables: List[dict]) -> Dict[int, List[dict]]:
    grouped = {}
    for table in tables:
        tid = table["table_id"]
        for cell in table["cells"]:
            grouped.setdefault(tid, []).append(cell)
    return grouped


def evaluate_excel_result(model_result: Dict[str, Any], student_result: Dict[str, Any]) -> Dict[str, Any]:
    model_tables = group_by_table(model_result["Sheet1"]["tables"])
    student_tables = group_by_table(student_result["Sheet1"]["tables"])


    total_cells = sum(len(cells) for cells in model_tables.values())
    total_formula = 0
    wrong_formula = 0
    wrong_cells = 0
    missing_cells = 0
    cell_errors = []

    for tid, model_cells in model_tables.items():
        student_cells = student_tables.get(tid, [])
        for m_cell in model_cells:
            s_cell = next(
                (c for c in student_cells
                 if c["relative_row"] == m_cell["relative_row"] and c["relative_col"] == m_cell["relative_col"]),
                None
            )
            if not s_cell:
                missing_cells += 1
                if m_cell.get("formula"):  # Nếu ô mẫu có công thức → trừ công thức luôn
                    wrong_formula += 1
                cell_errors.append({
                    "table_id": tid,
                    "coordinate": m_cell.get("coordinate"),
                    "error": "missing",
                    "expected_formula": m_cell.get("formula") if m_cell.get("formula") else None
                })
                continue

            keys = ["value", "background_color", "is_merged", "merged_origin", "border", "relative_row", "relative_col"]
            error_fields = []
            for k in keys:
                if m_cell.get(k) != s_cell.get(k):
                    error_fields.append(k)

            if error_fields:
                wrong_cells += 1
                cell_errors.append({
                    "table_id": tid,
                    "coordinate": m_cell.get("coordinate"),
                    "relative_row": m_cell.get("relative_row"),
                    "relative_col": m_cell.get("relative_col"),
                    "errors": error_fields,
                    "expected": {k: m_cell.get(k) for k in error_fields},
                    "actual": {k: s_cell.get(k) for k in error_fields}
                })

            # So sánh công thức
            if m_cell.get("formula"):
                total_formula += 1
                expected_formula = normalize_formula(m_cell.get("formula"))
                actual_formula = normalize_formula(s_cell.get("formula"))
                print("Expected (normalized):", expected_formula)
                print("Actual   (normalized):", actual_formula)
                if expected_formula != actual_formula:
                    wrong_formula += 1
                    cell_errors.append({
                        "table_id": tid,
                        "coordinate": m_cell.get("coordinate"),
                        "error": "formula_mismatch",
                        "expected": expected_formula,
                        "actual": actual_formula
                    })

    formula_penalty = (wrong_formula / total_formula * 2.2) if total_formula else 0
    cell_penalty = (wrong_cells / total_cells * 0.9) if total_cells else 0
    missing_penalty = (missing_cells / total_cells * 0.9 * 7) if total_cells else 0

    sort_penalty = 0
    if model_result["Sheet1"]["tables"] and student_result["Sheet1"]["tables"]:
        model_sort = model_result["Sheet1"]["tables"][0].get("sort_analysis", {})
        student_sort = student_result["Sheet1"]["tables"][0].get("sort_analysis", {})
        for col, m_val in model_sort.items():
            s_val = student_sort.get(col)
            if s_val and s_val != m_val:
                sort_penalty += 0.5
                break
    
    # Student info
    info_student = student_result["Sheet1"].get("student_info")
    student_info_penalty = 1.0 if info_student is None else 0.0

    total_score = 10 - (formula_penalty + cell_penalty + sort_penalty + student_info_penalty + missing_penalty)
    return {
        "score": round(max(total_score, 0), 2),
        "info_student": info_student,
        "penalties": {
            "cell_penalty": round(cell_penalty, 2),
            "formula_penalty": round(formula_penalty, 2),
            "sort_penalty": round(sort_penalty, 2),
            "student_info_penalty": round(student_info_penalty, 2),
            "missing_penalty": round(missing_penalty, 2)
        },
        "matched_cells": total_cells - wrong_cells,
        "total_cells": total_cells,
        "wrong_formula": wrong_formula,
        "total_formula": total_formula,
        "cell_errors": cell_errors
    }

def load_excel_result(xlsx_path: str):
    with ZipFile(xlsx_path) as zipf:
        shared = extract_shared_strings(zipf)
        fills, borders = extract_styles(zipf)
        return extract_sheet(zipf, shared, fills, borders)

def main():
    file_model = r"E:\doAnTuChamDiemFileWordExcel\dethi2.xlsx"
    file_submission = r"E:\doAnTuChamDiemFileWordExcel\bailam1.xlsx"

    model_result = load_excel_result(file_model)
    student_result = load_excel_result(file_submission)

    result = evaluate_excel_result(model_result, student_result)

    print("\n===== KẾT QUẢ CHẤM ĐIỂM =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()