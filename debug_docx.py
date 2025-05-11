#!/usr/bin/env python3
import zipfile
import os
from lxml import etree

DOCX_FILE = "test.docx"

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

def list_docx_files():
    """List all files in the .docx archive"""
    with zipfile.ZipFile(DOCX_FILE) as zip_ref:
        print("\n=== FILES IN DOCX ARCHIVE ===")
        for file in zip_ref.namelist():
            print(file)

def dump_document_xml():
    """Extract and print document.xml content"""
    with zipfile.ZipFile(DOCX_FILE) as zip_ref:
        xml_content = zip_ref.read('word/document.xml')
        root = etree.fromstring(xml_content)
        print("\n=== DOCUMENT.XML STRUCTURE ===")
        print(etree.tostring(root, pretty_print=True, encoding='utf-8').decode('utf-8')[:2000] + "...")

def find_wordart_elements():
    """Try various XPath queries to locate WordArt elements"""
    with zipfile.ZipFile(DOCX_FILE) as zip_ref:
        xml_content = zip_ref.read('word/document.xml')
        root = etree.fromstring(xml_content)
        
        print("\n=== SEARCHING FOR WORDART ELEMENTS ===")
        
        # Approach 1: Look for specific WordArt elements
        print("\n1. Looking for wps:txbx elements:")
        txbx_elements = root.xpath('//wps:txbx', namespaces=namespaces)
        print(f"Found {len(txbx_elements)} wps:txbx elements")
        for i, elem in enumerate(txbx_elements):
            print(f"\nwps:txbx element {i+1}:")
            # Extract text content from the textbox
            text_elements = elem.xpath('.//w:t', namespaces=namespaces)
            text_content = ''.join([t.text for t in text_elements if t.text])
            print(f"Text content: {text_content}")
            # Look for any indicators this is WordArt
            parent_shape = elem.getparent().getparent()  # Often two levels up from txbx
            if parent_shape is not None:
                shape_id = parent_shape.get('{%s}id' % namespaces['w14'], 'No ID')
                print(f"Parent shape ID: {shape_id}")
                # Print some of the XML for inspection
                print(etree.tostring(elem, pretty_print=True, encoding='utf-8').decode('utf-8')[:300] + "...")
        
        # Approach 2: Look for v:textbox or v:shape with WordArt
        print("\n2. Looking for VML shapes that might contain WordArt:")
        vml_shapes = root.xpath('//v:shape', namespaces=namespaces)
        print(f"Found {len(vml_shapes)} v:shape elements")
        for i, elem in enumerate(vml_shapes):
            print(f"\nVML shape element {i+1}:")
            shape_id = elem.get('id', 'No ID')
            shape_style = elem.get('style', 'No style')
            print(f"ID: {shape_id}")
            print(f"Style: {shape_style}")
            # Check for textboxes inside
            textboxes = elem.xpath('.//v:textbox', namespaces=namespaces)
            if textboxes:
                print(f"Contains {len(textboxes)} textbox elements")
                for tb in textboxes:
                    style = tb.get('style', 'No style')
                    print(f"Textbox style: {style}")
            # Print raw XML for the first part of the element
            print(etree.tostring(elem, pretty_print=True, encoding='utf-8').decode('utf-8')[:300] + "...")
        
        # Approach 3: Look for alternate content that might contain WordArt
        print("\n3. Looking for mc:AlternateContent that might contain WordArt:")
        alt_content = root.xpath('//mc:AlternateContent', namespaces=namespaces)
        print(f"Found {len(alt_content)} mc:AlternateContent elements")
        for i, elem in enumerate(alt_content):
            print(f"\nAlternateContent element {i+1}:")
            # Check if it has a Choice with wps requirement (often used for WordArt)
            choices = elem.xpath('./mc:Choice', namespaces=namespaces)
            for choice in choices:
                requires = choice.get('Requires', 'None')
                print(f"Choice requires: {requires}")
                
                # Check if it contains wps:wsp (WordProcessingShape)
                wsp_elements = choice.xpath('.//wps:wsp', namespaces=namespaces)
                if wsp_elements:
                    print(f"Contains {len(wsp_elements)} WordProcessingShape elements")
                
                # Check for textboxes which could be WordArt
                txbx_elements = choice.xpath('.//wps:txbx', namespaces=namespaces)
                if txbx_elements:
                    print(f"Contains {len(txbx_elements)} text box elements")
                    # Get text content
                    for txbx in txbx_elements:
                        text_elements = txbx.xpath('.//w:t', namespaces=namespaces)
                        text_content = ''.join([t.text for t in text_elements if t.text])
                        print(f"Text content: {text_content}")
                
                # Check for text effects, which are a strong indicator of WordArt
                text_effects = choice.xpath('.//a:effectLst', namespaces=namespaces)
                if text_effects:
                    print(f"Contains {len(text_effects)} text effect elements")
            
            # Print some of the XML for inspection
            print(etree.tostring(elem, pretty_print=True, encoding='utf-8').decode('utf-8')[:300] + "...")

def find_symbol_elements():
    """Try various XPath queries to locate Symbols"""
    with zipfile.ZipFile(DOCX_FILE) as zip_ref:
        xml_content = zip_ref.read('word/document.xml')
        root = etree.fromstring(xml_content)
        
        print("\n=== SEARCHING FOR SYMBOL ELEMENTS ===")
        
        # Approach 1: Look for w:sym elements
        print("\n1. Looking for w:sym elements:")
        sym_elements = root.xpath('//w:sym', namespaces=namespaces)
        print(f"Found {len(sym_elements)} w:sym elements")
        for i, elem in enumerate(sym_elements):
            print(f"\nSymbol element {i+1}:")
            char = elem.get('{%s}char' % namespaces['w'], 'No char')
            font = elem.get('{%s}font' % namespaces['w'], 'No font')
            print(f"Character: {char}, Font: {font}")
            # Print parent run for context
            parent_run = elem.getparent()
            if parent_run is not None:
                print("Parent run XML:")
                print(etree.tostring(parent_run, pretty_print=True, encoding='utf-8').decode('utf-8'))
        
        # Approach 2: Look for special Unicode characters that might be symbols in text runs
        print("\n2. Looking for text runs with potential symbol characters:")
        text_elements = root.xpath('//w:t', namespaces=namespaces)
        symbol_characters = []
        
        # Unicode ranges for symbols (approximate)
        symbol_ranges = [
            (0x2000, 0x206F),  # General Punctuation
            (0x2070, 0x209F),  # Superscripts and Subscripts
            (0x20A0, 0x20CF),  # Currency Symbols
            (0x2100, 0x214F),  # Letterlike Symbols
            (0x2150, 0x218F),  # Number Forms
            (0x2190, 0x21FF),  # Arrows
            (0x2200, 0x22FF),  # Mathematical Operators
            (0x2300, 0x23FF),  # Miscellaneous Technical
            (0x2460, 0x24FF),  # Enclosed Alphanumerics
            (0x2500, 0x257F),  # Box Drawing
            (0x2580, 0x259F),  # Block Elements
            (0x25A0, 0x25FF),  # Geometric Shapes
            (0x2600, 0x26FF),  # Miscellaneous Symbols
            (0x2700, 0x27BF),  # Dingbats
            (0x1D400, 0x1D7FF) # Mathematical Alphanumeric Symbols
        ]
        
        for elem in text_elements:
            if elem.text:
                for char in elem.text:
                    code_point = ord(char)
                    is_symbol = False
                    for start, end in symbol_ranges:
                        if start <= code_point <= end:
                            is_symbol = True
                            break
                    
                    if is_symbol:
                        parent_run = elem.getparent()
                        run_props = parent_run.xpath('./w:rPr', namespaces=namespaces) if parent_run is not None else []
                        font_info = ""
                        if run_props:
                            fonts = run_props[0].xpath('./w:rFonts', namespaces=namespaces)
                            if fonts:
                                ascii_font = fonts[0].get('{%s}ascii' % namespaces['w'], 'Default')
                                hansi_font = fonts[0].get('{%s}hAnsi' % namespaces['w'], 'Default')
                                font_info = f", Font(ascii): {ascii_font}, Font(hAnsi): {hansi_font}"
                        
                        symbol_characters.append((char, f"U+{code_point:04X}{font_info}"))
        
        if symbol_characters:
            print("\nFound potential symbol characters:")
            for char, info in symbol_characters:
                print(f"Symbol: {char} - {info}")
        else:
            print("No potential symbol characters found")
        
        # Approach 3: Look for runs with symbol fonts
        print("\n3. Looking for text runs with 'Symbol' font:")
        symbol_fonts = root.xpath('//w:rFonts[contains(@w:ascii, "Symbol") or contains(@w:hAnsi, "Symbol")]', namespaces=namespaces)
        print(f"Found {len(symbol_fonts)} text runs with Symbol font")
        for i, font_elem in enumerate(symbol_fonts[:5]):  # Show first 5 for brevity
            parent_run = font_elem.getparent().getparent()  # rPr -> run
            if parent_run is not None:
                text_elements = parent_run.xpath('.//w:t', namespaces=namespaces)
                text = ''.join([t.text for t in text_elements if t.text]) if text_elements else "[No text]"
                print(f"Run with Symbol font {i+1}: {text}")

if __name__ == "__main__":
    print(f"Analyzing file: {DOCX_FILE}")
    
    # Run all analysis functions
    list_docx_files()
    dump_document_xml()
    find_wordart_elements()
    find_symbol_elements() 