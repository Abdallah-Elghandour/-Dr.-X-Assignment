import os
import pandas as pd
import docx
import PyPDF2
import csv
from typing import Dict, List, Union, Tuple


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
        result['extension'] = extension
        return result
    
    def _read_docx(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        doc = docx.Document(file_path)
        full_text = []
        
        # Extract text from paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        for table in doc.tables:
            full_text.append("TABLE CONTENT:")
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                full_text.append(" | ".join(row_data))    
        # Join all text with double newlines for readability
        text = '\n\n'.join(full_text)
        return {
            'text': text,
            'pages': [text]  # DOCX doesn't have page info by default
        }
    
    def _read_pdf(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages = []
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                pages.append(page.extract_text())
            
            full_text = '\n\n'.join(pages)
            return {
                'text': full_text,
                'pages': pages
            }
    
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
        else:
            # Read Excel files
            try:
                data = []
                excel_file = pd.ExcelFile(file_path)
                
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    data.append(f"Sheet: {sheet_name}")
                    
                    # Convert DataFrame to text representation
                    rows = [df.columns.tolist()] + df.values.tolist()
                    for row in rows:
                        data.append(" | ".join([str(cell) for cell in row]))
                    
                    data.append("")  # Add space between sheets
            except Exception as e:
                return {
                    'text': f"Error reading Excel file: {str(e)}",
                    'pages': [f"Error reading Excel file: {str(e)}"]
                }
        
        # Convert data to text
        if isinstance(data[0], list):  # For CSV data
            text_data = ["TABLE CONTENT:"]  # Add header
            for row in data:
                row_text = [str(cell).strip() for cell in row if str(cell).strip()]
                if row_text:  # Only add non-empty rows
                    text_data.append(" | ".join(row_text))
            text = "\n".join(text_data)
        else:  # For Excel data already processed
            text = "\n".join(data)
        
        return {
            'text': text,
            'pages': [text]  # Tabular files don't have pages by default
        }


def extract_text_from_file(file_path: str) -> Dict[str, Union[str, List[str]]]:
    """
    Wrapper function to extract text from a file using DocumentReader.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with extracted text and metadata
    """
    reader = DocumentReader()
    return reader.read_document(file_path)