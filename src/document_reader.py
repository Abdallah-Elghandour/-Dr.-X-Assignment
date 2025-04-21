import os
import pandas as pd
import docx
import csv
import pdfplumber
from typing import Dict, List, Union, Tuple
from docx2pdf import convert

class DocumentReader:
    """
    A class to read and extract text from various document formats including
    DOCX, PDF, CSV, XLSX, XLS, and XLSM files.
    """
    
    def __init__(self):
        """Initialize the DocumentReader."""
        self.supported_extensions = {
            'docx': self._read_docx,
            'pdf': self._read_pdf,
            'csv': self._read_tabular,
            'xlsx': self._read_tabular,
            'xls': self._read_tabular,
            'xlsm': self._read_tabular
        }
    
    def read_document(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Read a document and extract its text content.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_name = os.path.basename(file_path)
        extension = file_path.split('.')[-1].lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")

        result = self.supported_extensions[extension](file_path)
        result['source'] = file_name
        return result
    
    def _read_pdf(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from a PDF file, including tables using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted Pages
        """
        pages = []
        
        # Use pdfplumber for text and table extraction
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract regular text
                page_text = page.extract_text() or ""
                
                # Extract tables
                tables = page.extract_tables()
                
                if tables:
                    tables_text = []
                    for table_num, table in enumerate(tables):
                        # Process each table
                        table_text = []
                        for row in table:
                            # Filter out None values and convert all cells to strings
                            processed_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            # Only include rows that have at least one non-empty cell
                            if any(cell for cell in processed_row):
                                table_text.append(" | ".join(processed_row))
                        
                        if table_text:
                            tables_text.append(f"TABLE {table_num + 1}:\n" + "\n".join(table_text))
                    
                    # Add tables to the page text
                    if tables_text:
                        page_text += f"\n\nTABLES EXTRACTED FROM PAGE {page_num + 1}:\n" + "\n\n".join(tables_text)
                
                pages.append(page_text)
        
        for page in pages:
            print(page)
            print("-"*50)
        return {
            'pages': pages
        }

    def _read_docx(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with extracted Pages
        """
        # Step 1: Convert DOCX to PDF
        pdf_path = file_path.replace(".docx", ".pdf")
        
        # Convert only if not already converted
        if not os.path.exists(pdf_path):
            convert(file_path, pdf_path)

        try:
            pdf = self._read_pdf(pdf_path)
            return {
                'pages': pdf['pages']
            }
        finally:
            # Clean up
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def _read_tabular(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from tabular files (CSV, XLSX, XLS, XLSM).
        
        Args:
            file_path: Path to the tabular file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        extension = file_path.split('.')[-1].lower()
        if extension == 'csv':
            # Read CSV file
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                data = list(reader)
                
                # Create pages for CSV (each page contains ~100 rows)
                rows_per_page = 100
                pages = []
                
                for i in range(0, len(data), rows_per_page):
                    page_data = data[i:i+rows_per_page]
                    page_text = ["TABLE CONTENT (Page {}):".format(len(pages) + 1)]
                    
                    for row in page_data:
                        row_text = [str(cell).strip() for cell in row if str(cell).strip()]
                        if row_text:  # Only add non-empty rows
                            page_text.append(" | ".join(row_text))
                    
                    pages.append("\n".join(page_text))
                
                # If no pages were created, create at least one
                if not pages:
                    pages = ["No data found"]
        else:
            # Read Excel files
            try:
                pages = []
                
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                
                # Treat each sheet as a separate "page"
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    page_text = [f"Sheet: {sheet_name}"]
                    
                    # Convert DataFrame to text representation
                    rows = [df.columns.tolist()] + df.values.tolist()
                    for row in rows:
                        row_text = " | ".join([str(cell) for cell in row])
                        page_text.append(row_text)
                    
                    pages.append("\n".join(page_text))
               
                
                # If no pages were created, create at least one
                if not pages:
                    pages = ["No data found"]
            except Exception as e:
                print(f"Excel reading error: {str(e)}")
                pages = ["Error reading Excel file"]
        
        return {
            'pages': pages
        }


