#!/usr/bin/env python3
import argparse
import zipfile
import os
from lxml import etree
import docx

# Helper function to get font information from text runs
def get_font_info(run):
    """
    Extract font name and size from a text run using python-docx.
    
    Args:
        run: A w:r element from the document XML
        
    Returns:
        Dictionary with font_name and font_size
    """
    font_info = {
        'font_name': None,
        'font_size': None
    }
    
    try:
        # Check for rFonts element to get font name
        r_fonts = run.xpath('./w:rPr/w:rFonts', namespaces=namespaces)
        if r_fonts:
            # Try to get font from different attributes in order of preference
            for attr in ['{%s}ascii' % namespaces['w'], 
                         '{%s}hAnsi' % namespaces['w'], 
                         '{%s}cs' % namespaces['w'],
                         '{%s}eastAsia' % namespaces['w']]:
                font_name = r_fonts[0].get(attr)
                if font_name:
                    font_info['font_name'] = font_name
                    break
        
        # Check for sz element to get font size (in half-points)
        sz_elem = run.xpath('./w:rPr/w:sz', namespaces=namespaces)
        if sz_elem:
            # Font size is stored in half-points, convert to points
            half_points = sz_elem[0].get('{%s}val' % namespaces['w'])
            if half_points:
                font_info['font_size'] = float(half_points) / 2
    except Exception as e:
        print(f"Error getting font info: {e}")
    
    return font_info

# Define XML namespaces used in .docx files
namespaces = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
    'wpg': 'http://schemas.microsoft.com/office/word/2010/wordprocessingGroup',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
    'o': 'urn:schemas-microsoft-com:office:office',
    'v': 'urn:schemas-microsoft-com:vml',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
}

def extract_wordart(root):
    """Extract WordArt elements from the document."""
    wordart_elements = []
    
    # Approach 1: Look for wps:txbx elements (modern WordArt)
    txbx_elements = root.xpath('//wps:txbx', namespaces=namespaces)
    
    for elem in txbx_elements:
        # Extract text content with font information
        text_segments = []
        text_runs = elem.xpath('.//w:r', namespaces=namespaces)
        
        for run in text_runs:
            text_elements = run.xpath('./w:t', namespaces=namespaces)
            text_content = ''.join([t.text for t in text_elements if t.text])
            
            if text_content:
                # Get font information
                font_info = get_font_info(run)
                
                text_segments.append({
                    'text': text_content,
                    'font_name': font_info['font_name'],
                    'font_size': font_info['font_size']
                })
        
        # Get parent shape info
        parent_shape = elem.getparent().getparent()  # Often two levels up from txbx
        shape_info = {}
        
        if parent_shape is not None:
            # Try to get shape ID
            shape_id = parent_shape.get('{%s}id' % namespaces['w14'], None)
            if shape_id:
                shape_info['id'] = shape_id
            
            # Check for text effects
            text_effects = parent_shape.xpath('.//a:effectLst', namespaces=namespaces)
            if text_effects:
                shape_info['has_effects'] = True
                
            # Check for preset geometry (common in WordArt)
            preset_geom = parent_shape.xpath('.//a:prstGeom', namespaces=namespaces)
            if preset_geom:
                geom_type = preset_geom[0].get('prst', 'unknown')
                shape_info['geometry'] = geom_type
        
        # Combine all text for backward compatibility
        full_text = ''.join([segment['text'] for segment in text_segments])
        
        wordart_elements.append({
            'type': 'modern_wordart',
            'text': full_text,
            'text_segments': text_segments,
            'shape_info': shape_info
        })
    
    # Approach 2: Look for mc:AlternateContent with wps requirement (often WordArt)
    alt_content = root.xpath('//mc:AlternateContent/mc:Choice[@Requires="wps"]', namespaces=namespaces)
    
    for elem in alt_content:
        # Check if it contains text boxes which could be WordArt
        txbx_elements = elem.xpath('.//wps:txbx', namespaces=namespaces)
        
        if txbx_elements:
            for txbx in txbx_elements:
                # Skip if we've already processed this text box above
                if txbx in txbx_elements:
                    continue
                
                # Extract text content with font information
                text_segments = []
                text_runs = txbx.xpath('.//w:r', namespaces=namespaces)
                
                for run in text_runs:
                    text_elements = run.xpath('./w:t', namespaces=namespaces)
                    text_content = ''.join([t.text for t in text_elements if t.text])
                    
                    if text_content:
                        # Get font information
                        font_info = get_font_info(run)
                        
                        text_segments.append({
                            'text': text_content,
                            'font_name': font_info['font_name'],
                            'font_size': font_info['font_size']
                        })
                
                # Combine all text for backward compatibility
                full_text = ''.join([segment['text'] for segment in text_segments])
                
                # Check if this has WordArt characteristics
                wsp_elements = elem.xpath('ancestor::wps:wsp', namespaces=namespaces)
                shape_info = {}
                
                if wsp_elements:
                    # Look for text effects
                    text_effects = wsp_elements[0].xpath('.//a:effectLst', namespaces=namespaces)
                    if text_effects:
                        shape_info['has_effects'] = True
                
                wordart_elements.append({
                    'type': 'modern_wordart_alternate',
                    'text': full_text,
                    'text_segments': text_segments,
                    'shape_info': shape_info
                })
    
    # Approach 3: Look for VML shapes with textboxes (legacy WordArt in older documents)
    vml_shapes = root.xpath('//v:shape', namespaces=namespaces)
    
    for shape in vml_shapes:
        shape_id = shape.get('id', '')
        shape_style = shape.get('style', '')
        
        # Check if this looks like WordArt
        is_wordart = False
        if 'WordArt' in shape_id or 'WordArt' in shape_style:
            is_wordart = True
        
        # Look for textboxes
        textboxes = shape.xpath('.//v:textbox', namespaces=namespaces)
        
        if textboxes:
            for textbox in textboxes:
                # Extract text with font information
                text_segments = []
                text_runs = textbox.xpath('.//w:r', namespaces=namespaces)
                
                for run in text_runs:
                    text_elements = run.xpath('./w:t', namespaces=namespaces)
                    text_content = ''.join([t.text for t in text_elements if t.text])
                    
                    if text_content:
                        # Get font information
                        font_info = get_font_info(run)
                        
                        text_segments.append({
                            'text': text_content,
                            'font_name': font_info['font_name'],
                            'font_size': font_info['font_size']
                        })
                
                # Combine all text for backward compatibility
                full_text = ''.join([segment['text'] for segment in text_segments])
                
                # If it has text effects or is explicitly WordArt, add it
                if is_wordart or 'mso-fit-shape-to-text' in textbox.get('style', ''):
                    wordart_elements.append({
                        'type': 'legacy_wordart',
                        'text': full_text,
                        'text_segments': text_segments,
                        'id': shape_id,
                        'style': shape_style
                    })
    
    return wordart_elements

def extract_columns(root):
    """Extract column settings and content from the document."""
    sections_info = []
    
    # Find all section properties with column settings
    section_elements = root.xpath('//w:sectPr', namespaces=namespaces)
    
    # If no sections found, return empty list
    if not section_elements:
        return sections_info
    
    # Analyze each section for column information
    for i, sect_pr in enumerate(section_elements):
        # Find column settings in this section
        cols_element = sect_pr.xpath('./w:cols', namespaces=namespaces)
        if not cols_element:
            continue
            
        cols = cols_element[0]
        num_cols = int(cols.get('{%s}num' % namespaces['w'], '1'))
        space = cols.get('{%s}space' % namespaces['w'], '425')  # Default space is 425 twips
        has_separator = cols.get('{%s}sep' % namespaces['w'], None) == "1"
        
        # Extract individual column widths if specified
        col_elements = cols.xpath('./w:col', namespaces=namespaces)
        individual_cols = []
        
        if col_elements:
            for col in col_elements:
                width = col.get('{%s}w' % namespaces['w'], 'auto')
                individual_cols.append(f"width: {width}")
        else:
            # If no explicit column widths, assume equal distribution
            individual_cols = ['width: auto'] * num_cols
        
        # Now find the paragraphs that belong to this section
        try:
            # For first section, get all paragraphs before the next section break
            # For subsequent sections, get paragraphs between this section break and the next
            
            # Determine the range of paragraphs for this section
            all_paragraphs = root.xpath('//w:p', namespaces=namespaces)
            
            if i == 0:
                # For the first section, find all paragraphs until the next section break
                # or until the end if this is the only section
                section_start = 0
            else:
                # Find the paragraph containing this section break
                section_prs = root.xpath('//w:p[.//w:sectPr]', namespaces=namespaces)
                if i-1 < len(section_prs):
                    section_start_p = section_prs[i-1]
                    section_start = all_paragraphs.index(section_start_p) + 1
                else:
                    # This shouldn't happen in well-formed documents, but just in case
                    section_start = 0
            
            # Find the next section break or the end of document
            if i + 1 < len(section_elements):
                next_section_prs = root.xpath('//w:p[.//w:sectPr]', namespaces=namespaces)
                if i < len(next_section_prs):
                    next_section_p = next_section_prs[i]
                    section_end = all_paragraphs.index(next_section_p) + 1
                else:
                    section_end = len(all_paragraphs)
            else:
                section_end = len(all_paragraphs)
            
            # Get all paragraphs for this section
            section_paragraphs = all_paragraphs[section_start:section_end]
            
            # Store columns content with font information
            columns_text = [[] for _ in range(num_cols)]  # List of lists of text segments
            
            if num_cols == 1:
                # If only one column, all text goes there
                for p in section_paragraphs:
                    runs = p.xpath('.//w:r', namespaces=namespaces)
                    
                    for run in runs:
                        text_elements = run.xpath('./w:t', namespaces=namespaces)
                        text_content = ''.join([t.text for t in text_elements if t.text])
                        
                        if text_content:
                            # Get font information
                            font_info = get_font_info(run)
                            
                            columns_text[0].append({
                                'text': text_content,
                                'font_name': font_info['font_name'],
                                'font_size': font_info['font_size']
                            })
            else:
                # For multiple columns, distribute paragraphs using a simple approach
                # In real documents, text flows based on rendering, which we can only approximate
                
                # Method 1: Divide paragraphs evenly among columns (simple but not always accurate)
                if section_paragraphs:
                    paragraphs_per_column = max(1, len(section_paragraphs) // num_cols)
                    
                    for col_idx in range(num_cols):
                        start_idx = col_idx * paragraphs_per_column
                        end_idx = start_idx + paragraphs_per_column
                        
                        # For the last column, include any remaining paragraphs
                        if col_idx == num_cols - 1:
                            end_idx = len(section_paragraphs)
                        
                        # Ensure indices are valid
                        start_idx = min(start_idx, len(section_paragraphs))
                        end_idx = min(end_idx, len(section_paragraphs))
                        
                        # Get text from paragraphs assigned to this column
                        for p in section_paragraphs[start_idx:end_idx]:
                            runs = p.xpath('.//w:r', namespaces=namespaces)
                            
                            for run in runs:
                                text_elements = run.xpath('./w:t', namespaces=namespaces)
                                text_content = ''.join([t.text for t in text_elements if t.text])
                                
                                if text_content:
                                    # Get font information
                                    font_info = get_font_info(run)
                                    
                                    columns_text[col_idx].append({
                                        'text': text_content, 
                                        'font_name': font_info['font_name'],
                                        'font_size': font_info['font_size']
                                    })
            
            # Create section info with both column settings and content
            section_info = {
                'num_columns': num_cols,
                'space_between': space,
                'has_separator': has_separator,
                'individual_columns': individual_cols,
                'columns_text': columns_text,
                # Add plain text version for backward compatibility
                'columns_plain_text': [''.join([segment['text'] for segment in col]) for col in columns_text]
            }
            
            sections_info.append(section_info)
        except Exception as e:
            # If there's an error determining sections, create a basic entry
            print(f"Warning: Error processing section {i+1}: {e}")
            section_info = {
                'num_columns': num_cols,
                'space_between': space,
                'has_separator': has_separator,
                'individual_columns': individual_cols
            }
            sections_info.append(section_info)
    
    return sections_info

def extract_drop_caps(root):
    """Extract drop cap formatting and the corresponding character from the document."""
    drop_caps = []
    
    # Find paragraphs with drop cap settings
    paragraphs_with_dropcaps = root.xpath('//w:p[.//w:pPr/w:framePr[@w:dropCap]]', namespaces=namespaces)
    
    for paragraph in paragraphs_with_dropcaps:
        # Get the drop cap settings
        frame_pr = paragraph.xpath('.//w:pPr/w:framePr[@w:dropCap]', namespaces=namespaces)[0]
        drop_type = frame_pr.get('{%s}dropCap' % namespaces['w'], 'none')
        lines = frame_pr.get('{%s}lines' % namespaces['w'], '1')
        
        # Find the first text run and extract the first character
        text_runs = paragraph.xpath('.//w:r', namespaces=namespaces)
        first_run = text_runs[0] if len(text_runs) > 0 else None
        first_char = ''
        font_info = {'font_name': None, 'font_size': None}
        
        if first_run is not None:
            # Get font information
            font_info = get_font_info(first_run)
            
            # Get text content
            text_elements = first_run.xpath('./w:t', namespaces=namespaces)
            if len(text_elements) > 0 and text_elements[0].text:
                first_char = text_elements[0].text[0]  # Get the first character
        
        drop_caps.append({
            'type': drop_type,
            'lines': lines,
            'char': first_char,
            'font_name': font_info['font_name'],
            'font_size': font_info['font_size']
        })
    
    return drop_caps

def extract_tables(root):
    """Extract tables and their contents from the document."""
    tables = []
    table_elements = root.xpath('//w:tbl', namespaces=namespaces)
    
    for tbl in table_elements:
        table_data = []
        rows = tbl.xpath('.//w:tr', namespaces=namespaces)
        
        for row in rows:
            row_data = []
            cells = row.xpath('.//w:tc', namespaces=namespaces)
            for cell in cells:
                # Lấy tất cả đoạn văn trong cell
                paragraphs = cell.xpath('.//w:p', namespaces=namespaces)
                cell_text = ''
                for para in paragraphs:
                    texts = para.xpath('.//w:t', namespaces=namespaces)
                    para_text = ''.join([t.text for t in texts if t.text])
                    cell_text += para_text + '\n'
                row_data.append(cell_text.strip())
            table_data.append(row_data)
        
        tables.append(table_data)
    
    return tables

def extract_symbols(root):
    """Extract special characters or symbols from the document."""
    symbols = []
    
    # Approach 1: Find explicit symbol elements (w:sym)
    sym_elements = root.xpath('//w:sym', namespaces=namespaces)
    
    for sym in sym_elements:
        char = sym.get('{%s}char' % namespaces['w'], '')
        font = sym.get('{%s}font' % namespaces['w'], 'Default')
        
        # Try to convert the character code to actual character
        try:
            if char.startswith('F'):  # Often symbols use 'F' prefix in hex
                char_int = int(char, 16)
                actual_char = chr(char_int)
            else:
                actual_char = char
        except:
            actual_char = char
            
        # For explicit symbols, get font size from parent run if available
        font_size = None
        parent_run = sym.getparent()
        if parent_run is not None:
            run_font_info = get_font_info(parent_run)
            font_size = run_font_info['font_size']
            
        symbols.append({
            'type': 'explicit_symbol',
            'character': actual_char,
            'char_code': char,
            'font_name': font,
            'font_size': font_size
        })
    
    # Approach 2: Look for special characters in text runs with specific symbol fonts
    # Symbol fonts commonly used for special characters
    symbol_font_list = ['Symbol', 'Wingdings', 'Wingdings 2', 'Wingdings 3', 
                        'Webdings', 'MT Extra', 'MS UI Gothic',
                        'Segoe UI Symbol', 'Arial Unicode MS']
    
    font_xpath = ' or '.join([f"contains(@w:ascii, '{font}') or contains(@w:hAnsi, '{font}')" 
                              for font in symbol_font_list])
    
    symbol_runs = root.xpath(f"//w:r[.//w:rFonts[{font_xpath}]]", namespaces=namespaces)
    
    for run in symbol_runs:
        # Get font information
        font_info = get_font_info(run)
        
        text_elements = run.xpath('.//w:t', namespaces=namespaces)
        if text_elements:
            text_content = ''.join([t.text for t in text_elements if t.text])
            if text_content:
                # Get the font information
                font_elem = run.xpath('.//w:rFonts', namespaces=namespaces)[0]
                font_ascii = font_elem.get('{%s}ascii' % namespaces['w'], 'Default')
                font_hansi = font_elem.get('{%s}hAnsi' % namespaces['w'], 'Default')
                font_name = font_ascii if font_ascii != 'Default' else font_hansi
                
                for char in text_content:
                    code_point = ord(char)
                    # Skip alphabetic characters and accented letters
                    if char.isalpha():
                        continue
                        
                    symbols.append({
                        'type': 'symbol_font_character',
                        'character': char,
                        'unicode': f"U+{code_point:04X}",
                        'font_name': font_name,
                        'font_size': font_info['font_size']
                    })
    
    # Approach 3: Look for true symbol characters using more precise Unicode ranges
    # Refined list of Unicode ranges for only symbols (not letters)
    symbol_ranges = [
        (0x00A0, 0x00BF),  # Latin-1 Punctuation and Symbols
        (0x00D7, 0x00D7),  # Multiplication sign
        (0x00F7, 0x00F7),  # Division sign
        (0x2000, 0x206F),  # General Punctuation
        (0x20A0, 0x20CF),  # Currency Symbols
        (0x2100, 0x214F),  # Letterlike Symbols (™, ©, ®, etc.)
        (0x2190, 0x21FF),  # Arrows
        (0x2200, 0x22FF),  # Mathematical Operators
        (0x2300, 0x23FF),  # Miscellaneous Technical
        (0x2460, 0x24FF),  # Enclosed Alphanumerics
        (0x2500, 0x257F),  # Box Drawing
        (0x2580, 0x259F),  # Block Elements
        (0x25A0, 0x25FF),  # Geometric Shapes
        (0x2600, 0x26FF),  # Miscellaneous Symbols
        (0x2700, 0x27BF),  # Dingbats
        (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
        (0x27F0, 0x27FF),  # Supplemental Arrows-A
        (0x2900, 0x297F),  # Supplemental Arrows-B
        (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
        (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
        (0x2B00, 0x2BFF),  # Miscellaneous Symbols and Arrows
    ]
    
    # Unicode General Categories for symbols (not letters)
    def is_symbol_category(char):
        import unicodedata
        # Check general category of character
        category = unicodedata.category(char)
        # Only accept true symbols: S* categories, P* (punctuation), and some other specific categories
        symbol_categories = ['Sm', 'Sc', 'Sk', 'So', 'Po', 'Ps', 'Pe', 'Pi', 'Pf']
        return category in symbol_categories
    
    # Find all text runs 
    all_text_runs = root.xpath('//w:r', namespaces=namespaces)
    
    for run in all_text_runs:
        # Get font information
        font_info = get_font_info(run)
        
        text_elements = run.xpath('.//w:t', namespaces=namespaces)
        if not text_elements:
            continue
            
        # Get run properties and font info
        run_props = run.xpath('./w:rPr', namespaces=namespaces)
        font_name = "Default"
        if run_props:
            fonts = run_props[0].xpath('./w:rFonts', namespaces=namespaces)
            if fonts:
                ascii_font = fonts[0].get('{%s}ascii' % namespaces['w'], 'Default')
                hansi_font = fonts[0].get('{%s}hAnsi' % namespaces['w'], 'Default')
                font_name = ascii_font if ascii_font != 'Default' else hansi_font
        
        # Check each character in text
        for text_elem in text_elements:
            if not text_elem.text:
                continue
                
            for char in text_elem.text:
                code_point = ord(char)
                
                # Skip basic ASCII letters and numbers
                if ord(char) < 127 and (char.isalpha() or char.isdigit() or char.isspace()):
                    continue
                
                # Double check if this is a letter from any language - skip if it is
                if char.isalpha():
                    continue
                    
                # Check if it's in a symbol range
                is_in_symbol_range = False
                for start, end in symbol_ranges:
                    if start <= code_point <= end:
                        is_in_symbol_range = True
                        break
                
                # Only proceed if in symbol range AND has a symbol category
                if is_in_symbol_range and is_symbol_category(char):
                    # Check if we've already recorded this symbol
                    is_duplicate = False
                    for sym in symbols:
                        if (sym.get('type') == 'unicode_symbol' and 
                            sym.get('character') == char and 
                            sym.get('font_name') == font_name):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        symbols.append({
                            'type': 'unicode_symbol',
                            'character': char,
                            'unicode': f"U+{code_point:04X}",
                            'font_name': font_name,
                            'font_size': font_info['font_size']
                        })
    
    # Approach 4: Look for mathematical symbols
    math_elements = root.xpath('//m:oMath', namespaces=namespaces)
    
    for math in math_elements:
        # Extract text version of the equation
        text_runs = math.xpath('.//w:r', namespaces=namespaces)
        equation_text = ""
        symbol_chars = []
        
        for run in text_runs:
            # Get font information
            font_info = get_font_info(run)
            
            text_elements = run.xpath('.//w:t', namespaces=namespaces)
            run_text = ''.join([t.text for t in text_elements if t.text])
            equation_text += run_text
            
            # Look for symbol characters in this run
            for char in run_text:
                code_point = ord(char)
                # Skip alphabetic characters
                if char.isalpha():
                    continue
                    
                # Only include math operators and symbols
                if is_symbol_category(char) or (0x2200 <= code_point <= 0x22FF):
                    symbol_chars.append({
                        'character': char,
                        'unicode': f"U+{code_point:04X}",
                        'font_name': font_info['font_name'],
                        'font_size': font_info['font_size']
                    })
        
        if symbol_chars:
            symbols.append({
                'type': 'math_symbol',
                'equation': equation_text,
                'symbols_found': symbol_chars
            })
    
    return symbols

def extract_image_info(root, relationships):
    """Extract image formatting properties from the document."""
    images = []
    
    # Find all drawing elements containing pictures
    drawing_elements = root.xpath('//w:drawing', namespaces=namespaces)
    
    for drawing in drawing_elements:
        # Check if it's an inline or anchored image
        inline = drawing.xpath('./wp:inline', namespaces=namespaces)
        anchor = drawing.xpath('./wp:anchor', namespaces=namespaces)
        
        if inline:
            position_type = "inline"
            container = inline[0]
        elif anchor:
            position_type = "anchored"
            container = anchor[0]
        else:
            continue
        
        # Get image size
        extent = container.xpath('./wp:extent', namespaces=namespaces)
        width = extent[0].get('cx') if extent else 'unknown'
        height = extent[0].get('cy') if extent else 'unknown'
        
        # Get image ID and format
        blip = container.xpath('.//a:blip', namespaces=namespaces)
        if blip:
            r_id = blip[0].get('{%s}embed' % namespaces['r'])
            image_path = relationships.get(r_id, 'unknown')
            image_format = os.path.splitext(image_path)[1] if image_path != 'unknown' else 'unknown'
        else:
            r_id = 'unknown'
            image_path = 'unknown'
            image_format = 'unknown'
            
        # Get position information if it's anchored
        position_info = {}
        if position_type == "anchored":
            pos_h = container.xpath('./wp:positionH', namespaces=namespaces)
            pos_v = container.xpath('./wp:positionV', namespaces=namespaces)
            
            if pos_h:
                position_info['horizontal_position'] = {
                    'relative_from': pos_h[0].get('relativeFrom', 'unknown'),
                    'align': pos_h[0].xpath('./wp:align', namespaces=namespaces)[0].text if pos_h[0].xpath('./wp:align', namespaces=namespaces) else 'unknown',
                    'offset': pos_h[0].xpath('./wp:posOffset', namespaces=namespaces)[0].text if pos_h[0].xpath('./wp:posOffset', namespaces=namespaces) else 'unknown'
                }
            
            if pos_v:
                position_info['vertical_position'] = {
                    'relative_from': pos_v[0].get('relativeFrom', 'unknown'),
                    'align': pos_v[0].xpath('./wp:align', namespaces=namespaces)[0].text if pos_v[0].xpath('./wp:align', namespaces=namespaces) else 'unknown',
                    'offset': pos_v[0].xpath('./wp:posOffset', namespaces=namespaces)[0].text if pos_v[0].xpath('./wp:posOffset', namespaces=namespaces) else 'unknown'
                }
        
        images.append({
            'id': r_id,
            'path': image_path,
            'format': image_format,
            'width': width,
            'height': height,
            'position_type': position_type,
            'position_details': position_info if position_type == "anchored" else {}
        })
    
    return images

def extract_relationships(docx_file):
    """Extract relationship information from the .docx file."""
    rel_dict = {}
    
    try:
        with zipfile.ZipFile(docx_file) as zip_ref:
            # Try to extract relationship file
            try:
                rel_content = zip_ref.read('word/_rels/document.xml.rels')
                rel_root = etree.fromstring(rel_content)
                
                for rel in rel_root.xpath('//Relationship', namespaces={}):
                    rel_id = rel.get('Id')
                    target = rel.get('Target')
                    rel_dict[rel_id] = target
                    
            except KeyError:
                print("Warning: Relationship file not found")
    except Exception as e:
        print(f"Error extracting relationships: {e}")
    
    return rel_dict

def extract_properties(docx_file):
    """Extract all requested properties from the .docx file."""
    results = {
        'wordart': [],
        'columns': [],
        'drop_caps': [],
        'symbols': [],
        'images': [],
        'tables':[]
    }
    
    try:
        # Extract relationships first
        relationships = extract_relationships(docx_file)
        
        # Open the .docx file as a zip archive
        with zipfile.ZipFile(docx_file) as zip_ref:
            # Extract the main document.xml file
            xml_content = zip_ref.read('word/document.xml')
            root = etree.fromstring(xml_content)
            
            # Extract each property
            results['wordart'] = extract_wordart(root)
            results['columns'] = extract_columns(root)
            results['drop_caps'] = extract_drop_caps(root)
            results['symbols'] = extract_symbols(root)
            results['images'] = extract_image_info(root, relationships)
            results['tables'] = extract_tables(root)
            
        # Extract margin information and add it to results
        margins = get_docx_margins(docx_file)
        results['margins'] = margins
            
    except Exception as e:
        print(f"Error extracting properties: {e}")
    
    return results

def display_results(results):
    """Display the extracted properties in a structured format."""
    print("\n" + "="*50)
    print("DOCX PROPERTIES EXTRACTION REPORT")
    print("="*50)
    
    # Display margin information
    print("\n--- DOCUMENT MARGINS (CM) ---")
    if 'margins' in results:
        margins = results['margins']
        print(f"  Top margin: {margins['top_margin']} cm")
        print(f"  Bottom margin: {margins['bottom_margin']} cm")
        print(f"  Left margin: {margins['left_margin']} cm")
        print(f"  Right margin: {margins['right_margin']} cm")
    else:
        print("No margin information found")
    
    # Display WordArt information
    print("\n--- WORDART ELEMENTS ---")
    if results['wordart']:
        print(f"Found {len(results['wordart'])} WordArt elements")
        for i, wa in enumerate(results['wordart'], 1):
            print(f"WordArt {i}:")
            print(f"  Type: {wa['type']}")
            print(f"  Text: {wa['text']}")
            
            # Display font information for each text segment
            if 'text_segments' in wa:
                print(f"  Text segments with font information:")
                for j, segment in enumerate(wa['text_segments'][:3], 1):  # Show first 3 segments
                    print(f"    Segment {j}: '{segment['text'][:20]}{'...' if len(segment['text']) > 20 else ''}'")
                    print(f"      Font: {segment['font_name'] or 'Default'}")
                    print(f"      Size: {segment['font_size'] or 'Default'} pt")
                
                if len(wa['text_segments']) > 3:
                    print(f"    ... and {len(wa['text_segments']) - 3} more segments")
            
            if 'shape_info' in wa and wa['shape_info']:
                print(f"  Shape info:")
                for key, value in wa['shape_info'].items():
                    print(f"    {key}: {value}")
            elif 'id' in wa:
                print(f"  ID: {wa['id']}")
                if 'style' in wa:
                    print(f"  Style: {wa['style'][:50]}..." if len(wa['style']) > 50 else f"  Style: {wa['style']}")
    else:
        print("No WordArt elements found")
    
    # Display Column information
    print("\n--- COLUMN SETTINGS ---")
    if results['columns']:
        for i, cols in enumerate(results['columns'], 1):
            print(f"Column Setting {i}:")
            print(f"  Number of columns: {cols['num_columns']}")
            print(f"  Space between columns: {cols['space_between']} twips")
            print(f"  Has separator: {cols.get('has_separator', False)}")
            
            if 'individual_columns' in cols and cols['individual_columns']:
                print("  Individual column settings:")
                for j, col in enumerate(cols['individual_columns'], 1):
                    print(f"    Column {j}: {col}")
            
            # Display column text content if available
            if 'columns_text' in cols and cols['columns_text']:
                print("  Column content:")
                for j, content in enumerate(cols['columns_text'], 1):
                    if not content:
                        print(f"    Column {j} text: (empty)")
                        continue
                    
                    # Show first few text segments with font info
                    print(f"    Column {j} text segments:")
                    for k, segment in enumerate(content[:3], 1):  # Show first 3 segments
                        print(f"      Segment {k}: '{segment['text'][:30]}{'...' if len(segment['text']) > 30 else ''}'")
                        print(f"        Font: {segment['font_name'] or 'Default'}")
                        print(f"        Size: {segment['font_size'] or 'Default'} pt")
                    
                    if len(content) > 3:
                        print(f"      ... and {len(content) - 3} more segments")
    else:
        print("No column settings found")
    
    # Display Drop Cap information
    print("\n--- DROP CAPS ---")
    if results['drop_caps']:
        for i, dc in enumerate(results['drop_caps'], 1):
            print(f"Drop Cap {i}:")
            print(f"  Type: {dc['type']}")
            print(f"  Lines: {dc['lines']}")
            print(f"  Character: {dc['char']}")
            print(f"  Font: {dc['font_name'] or 'Default'}")
            print(f"  Font Size: {dc['font_size'] or 'Default'} pt")
    else:
        print("No drop caps found")
    
    # Display Symbol information
    print("\n--- SYMBOLS ---")
    if results['symbols']:
        print(f"Found {len(results['symbols'])} symbols")
        
        # Group symbols by type for cleaner display
        by_type = {}
        for sym in results['symbols']:
            sym_type = sym['type']
            if sym_type not in by_type:
                by_type[sym_type] = []
            by_type[sym_type].append(sym)
        
        for sym_type, symbols_list in by_type.items():
            print(f"\n  {sym_type.replace('_', ' ').title()} ({len(symbols_list)}):")
            
            for i, sym in enumerate(symbols_list[:5], 1):  # Show first 5 of each type to avoid clutter
                if sym_type == 'explicit_symbol':
                    print(f"    Symbol {i}: Character '{sym['character']}', Font: {sym['font_name']}, Size: {sym['font_size'] or 'Default'} pt")
                elif sym_type == 'unicode_symbol':
                    print(f"    Symbol {i}: '{sym['character']}' ({sym['unicode']}), Font: {sym['font_name']}, Size: {sym['font_size'] or 'Default'} pt")
                elif sym_type == 'symbol_font_character':
                    print(f"    Symbol {i}: Character '{sym['character']}', Font: {sym['font_name']}, Size: {sym['font_size'] or 'Default'} pt")
                elif sym_type == 'math_symbol':
                    print(f"    Math expression {i}: {sym['equation'][:30]}..." if len(sym['equation']) > 30 else f"    Math expression {i}: {sym['equation']}")
                    if 'symbols_found' in sym and sym['symbols_found']:
                        for j, math_sym in enumerate(sym['symbols_found'][:3], 1):
                            print(f"      Symbol {j}: '{math_sym['character']}' ({math_sym['unicode']}), Font: {math_sym['font_name'] or 'Default'}, Size: {math_sym['font_size'] or 'Default'} pt")
                        if len(sym['symbols_found']) > 3:
                            print(f"      ... and {len(sym['symbols_found']) - 3} more symbols")
            
            if len(symbols_list) > 5:
                print(f"    ... and {len(symbols_list) - 5} more")
    else:
        print("No symbols found")
    
    # Display Image information
    print("\n--- IMAGES ---")
    if results['images']:
        for i, img in enumerate(results['images'], 1):
            print(f"Image {i}:")
            print(f"  ID: {img['id']}")
            print(f"  Path: {img['path']}")
            print(f"  Format: {img['format']}")
            print(f"  Size: {img['width']} x {img['height']} EMUs")
            print(f"  Position type: {img['position_type']}")
            
            if img['position_details']:
                print("  Position details:")
                if 'horizontal_position' in img['position_details']:
                    hp = img['position_details']['horizontal_position']
                    print(f"    Horizontal: relative to {hp['relative_from']}, "
                          f"align: {hp['align']}, offset: {hp['offset']}")
                if 'vertical_position' in img['position_details']:
                    vp = img['position_details']['vertical_position']
                    print(f"    Vertical: relative to {vp['relative_from']}, "
                          f"align: {vp['align']}, offset: {vp['offset']}")
    else:
        print("No images found")
    # Display tables information
    print("\n--- TABLES ---")
    if results['tables']:
        for i, table in enumerate(results['tables'], 1):
            print(f"Table {i}:")
            for row_idx, row in enumerate(table, 1):
                row_str = " | ".join(cell if cell else "[Empty]" for cell in row)
                print(f"  Row {row_idx}: {row_str}")
    else:
        print("No tables found.")

def get_docx_margins(file_path: str) -> dict:
    """
    Extract margin information from a .docx file.
    
    Args:
        file_path: Path to the .docx file
        
    Returns:
        Dictionary containing margin values in centimeters:
        {
            'top_margin': float,
            'bottom_margin': float,
            'left_margin': float,
            'right_margin': float
        }
    """
    # Define default margins in centimeters (2.54 cm = 1 inch)
    margins = {
        'top_margin': 2.54,  # Default 2.54 cm (1 inch)
        'bottom_margin': 2.54,
        'left_margin': 2.54,
        'right_margin': 2.54
    }
    
    try:
        # Open the .docx file as a zip archive
        with zipfile.ZipFile(file_path) as zip_ref:
            # Extract the main document.xml file
            xml_content = zip_ref.read('word/document.xml')
            root = etree.fromstring(xml_content)
            
            # Find all section properties that contain margin settings
            sect_prs = root.xpath('//w:sectPr', namespaces=namespaces)
            
            # If any section properties found, get the last one (default document settings)
            if sect_prs:
                # Use the last sectPr (which contains the default document settings)
                sect_pr = sect_prs[-1]
                
                # Find page margin settings
                pg_mar = sect_pr.xpath('./w:pgMar', namespaces=namespaces)
                
                if pg_mar:
                    # Get margin values in twips
                    top = pg_mar[0].get('{%s}top' % namespaces['w'])
                    bottom = pg_mar[0].get('{%s}bottom' % namespaces['w'])
                    left = pg_mar[0].get('{%s}left' % namespaces['w'])
                    right = pg_mar[0].get('{%s}right' % namespaces['w'])
                    
                    # Convert twips to centimeters (1 inch = 1440 twips, 1 inch = 2.54 cm)
                    # Therefore 1 twip = 2.54/1440 cm = 0.00176 cm
                    if top:
                        margins['top_margin'] = float(top) * 2.54 / 1440.0
                    if bottom:
                        margins['bottom_margin'] = float(bottom) * 2.54 / 1440.0
                    if left:
                        margins['left_margin'] = float(left) * 2.54 / 1440.0
                    if right:
                        margins['right_margin'] = float(right) * 2.54 / 1440.0
                        
                    # Round to 2 decimal places for readability
                    for key in margins:
                        margins[key] = round(margins[key], 2)
            
    except Exception as e:
        print(f"Error extracting margins: {e}")
    
    return margins

def main():
    parser = argparse.ArgumentParser(description='Extract properties from a .docx file')
    parser.add_argument('file', help='Path to the .docx file')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist")
        return
    
    if not args.file.endswith('.docx'):
        print("Error: File must be a .docx document")
        return
    
    print(f"Extracting properties from '{args.file}'...")
    results = extract_properties(args.file)
    display_results(results)

if __name__ == "__main__":
    main() 