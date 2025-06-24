from zipfile import ZipFile
from lxml import etree
import re
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

xlsx_path = r"E:\doAnTuChamDiemFileWordExcel\bailam1.xlsx"
ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def col_letter_to_index(col):
    index = 0
    for c in col:
        index = index * 26 + (ord(c.upper()) - ord('A') + 1)
    return index - 1

def index_to_col_letter(index):
    result = ''
    while index >= 0:
        result = chr(index % 26 + ord('A')) + result
        index = index // 26 - 1
    return result

def extract_shared_strings(zipf):
    shared = []
    with zipf.open('xl/sharedStrings.xml') as f:
        tree = etree.parse(f)
        for si in tree.findall('.//main:si', namespaces=ns):
            text = ''.join(si.xpath('.//main:t/text()', namespaces=ns))
            shared.append(text)
    return shared

def convert_formula_to_relative(formula: str, min_row: int, min_col: int) -> str:
    if not formula:
        return formula

    # Regex để tìm các ô: ví dụ A1, F13, B2
    cell_pattern = r'([A-Z]+)(\d+)'
    matches = re.findall(cell_pattern, formula)

    rel_coords = {}
    for col_letters, row_str in matches:
        abs_col = col_letter_to_index(col_letters)
        abs_row = int(row_str) - 1  # Excel rows start at 1
        rel_row = abs_row - min_row
        rel_col = abs_col - min_col
        rel_coords[f"{col_letters}{row_str}"] = f"[{rel_row},{rel_col}]"

    # Thay thế trong công thức
    for abs_ref, rel_ref in rel_coords.items():
        formula = formula.replace(abs_ref, rel_ref)

    return formula

def extract_styles(zipf):
    with zipf.open('xl/styles.xml') as f:
        tree = etree.parse(f)

    fills, borders = [], []
    for fill in tree.findall('.//main:fill', namespaces=ns):
        fg = fill.find('.//main:fgColor', namespaces=ns)
        fills.append(fg.attrib.get('rgb') if fg is not None else None)

    for border in tree.findall('.//main:border', namespaces=ns):
        b = {}
        for side in ['left', 'right', 'top', 'bottom']:
            el = border.find(f'main:{side}', namespaces=ns)
            b[side] = el.attrib.get('style') if el is not None else None
        borders.append(b)

    return fills, borders

def extract_sheet(zipf, shared, fills, borders):
    with zipf.open('xl/worksheets/sheet1.xml') as f:
        tree = etree.parse(f)

    sheet_data = tree.find('.//main:sheetData', namespaces=ns)

    # 🔍 Lấy dữ liệu ô A1 và A2
    info_student = {"A1": None, "A2": None}
    for row in sheet_data.findall('main:row', namespaces=ns):
        for c in row.findall('main:c', namespaces=ns):
            coord = c.attrib['r']
            if coord in ("A1", "A2"):
                v = c.find('main:v', namespaces=ns)
                value = shared[int(v.text)] if c.attrib.get('t') == 's' and v is not None else (v.text if v is not None else None)
                info_student[coord] = value

    # 🔎 Kiểm tra A2 có đúng định dạng số báo danh - đề thi hay không
    a1_text = info_student["A1"]
    a2_text = info_student["A2"]
    match = re.search(r"Số\s+báo\s+danh\s+(\d+)\s*-\s*Đề\s+thi\s+số\s+(\d+)", a2_text or "", re.IGNORECASE)

    if a1_text and match:
        student_info = {
            "name": a1_text.strip(),
            "student_code": match.group(1),
            "exam_code": match.group(2)
        }
        data_start_row = 2
    else:
        student_info = None
        data_start_row = 0

    # 📦 Extract cell data
    merged_map = {}
    for mc in tree.findall('.//main:mergeCell', namespaces=ns):
        ref = mc.attrib['ref']
        start, end = ref.split(':')
        start_col, start_row = re.findall(r'([A-Z]+)([0-9]+)', start)[0]
        end_col, end_row = re.findall(r'([A-Z]+)([0-9]+)', end)[0]
        for r in range(int(start_row), int(end_row)+1):
            for c in range(col_letter_to_index(start_col), col_letter_to_index(end_col)+1):
                coord = f"{index_to_col_letter(c)}{r}"
                merged_map[coord] = start

    grid = {}
    for row in sheet_data.findall('main:row', namespaces=ns):
        r_idx = int(row.attrib['r']) - 1
        if r_idx < data_start_row:
            continue
        for c in row.findall('main:c', namespaces=ns):
            coord = c.attrib['r']
            col_letter = re.findall(r'[A-Z]+', coord)[0]
            col_idx = col_letter_to_index(col_letter)

            value = None
            if c.attrib.get('t') == 's':
                v = c.find('main:v', namespaces=ns)
                value = shared[int(v.text)] if v is not None else None
            else:
                v = c.find('main:v', namespaces=ns)
                value = v.text if v is not None else None

            formula = c.findtext('main:f', default=None, namespaces=ns)
            s = int(c.attrib.get('s', 0))
            bg = fills[s] if s < len(fills) else None
            border = borders[s] if s < len(borders) else {'top': None, 'bottom': None, 'left': None, 'right': None}
            merged_origin = merged_map.get(coord)
            is_merged = coord in merged_map

            grid[(r_idx, col_idx)] = {
                'coordinate': coord,
                'value': value,
                'formula': formula,
                'background_color': bg,
                'is_merged': is_merged,
                'merged_origin': merged_origin,
                'border': border,
                'row': r_idx,
                'col': col_idx,
            }

    # 🔗 Phát hiện bảng (table) bằng DFS
    visited = set()
    tables = []
    def dfs(r, c, table):
        stack = [(r, c)]
        while stack:
            r0, c0 = stack.pop()
            if (r0, c0) in visited or (r0, c0) not in grid:
                continue
            visited.add((r0, c0))
            table.append(grid[(r0, c0)])
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if abs(dr) + abs(dc) == 0: continue
                    nr, nc = r0 + dr, c0 + dc
                    if (nr, nc) in grid and (nr, nc) not in visited:
                        stack.append((nr, nc))

    for r, c in grid:
        if (r, c) not in visited:
            table = []
            dfs(r, c, table)
            tables.append(table)

    # 🔎 Phân tích bảng
    result_tables = []
    for tid, table in enumerate(tables, start=1):
        min_row = min(cell['row'] for cell in table)
        min_col = min(cell['col'] for cell in table)
        for cell in table:
            cell['table_id'] = tid
            cell['relative_row'] = cell['row'] - min_row
            cell['relative_col'] = cell['col'] - min_col

            if cell['formula']:
                cell['formula'] = convert_formula_to_relative(cell['formula'], min_row, min_col)

        coord_map = {
            (c['row'], c['col']): (c['relative_row'], c['relative_col'])
            for c in table
        }

        for cell in table:
            if cell['is_merged'] and cell['merged_origin']:
                match = re.match(r"([A-Z]+)(\d+)", cell['merged_origin'])
                if match:
                    origin_col = col_letter_to_index(match.group(1))
                    origin_row = int(match.group(2)) - 1
                    rel_pos = coord_map.get((origin_row, origin_col))
                    if rel_pos:
                        cell['merged_origin'] = {
                            "relative_row": rel_pos[0],
                            "relative_col": rel_pos[1]
                        }

        col_data = {}
        for cell in table:
            col_data.setdefault(cell['relative_col'], []).append(cell)

        sort_analysis = {}
        for rel_col, items in col_data.items():
            items.sort(key=lambda x: x['relative_row'])
            numeric = []
            for i, cell in enumerate(items):
                if i < 2 or i >= len(items) - 1:
                    continue
                try:
                    numeric.append(float(cell['value']))
                except:
                    numeric.append(None)
            clean = [x for x in numeric if x is not None]
            col_letter = index_to_col_letter(rel_col)
            if clean == sorted(clean):
                sort_analysis[col_letter] = 'ascending'
            elif clean == sorted(clean, reverse=True):
                sort_analysis[col_letter] = 'descending'
            else:
                sort_analysis[col_letter] = 'none'

        result_tables.append({
            'table_id': tid,
            'sort_analysis': sort_analysis,
            'cells': table
        })

    return {
        'Sheet1': {
            'student_info': student_info,
            'tables': result_tables
        }
    }


# Run
with ZipFile(xlsx_path) as zipf:
    shared = extract_shared_strings(zipf)
    fills, borders = extract_styles(zipf)
    result = extract_sheet(zipf, shared, fills, borders)

with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("✅ Đã xuất file JSON hoàn chỉnh theo bảng (table) và công thức.")
