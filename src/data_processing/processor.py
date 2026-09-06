"""
Data Processing Module - Document Ingestion & Processing Pipeline
Handles: Document parsing, text extraction, data cleaning, chunking
"""

import os
import PyPDF2
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from openpyxl import load_workbook
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import re
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

logger.add("logs/data_processing.log", rotation="500 MB")

class DocumentParser:
    """
    Parse different document formats (PDF, DOCX, XLSX, TXT)
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.xlsx', '.txt', '.doc']
        
    def parse_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF for better accuracy
        """
        try:
            text = ""
            pdf_document = fitz.open(file_path)
            
            for page_num, page in enumerate(pdf_document):
                # Extract text
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n" + page_text
                
                # Extract images if needed (for OCR)
                image_list = page.get_images()
                if image_list:
                    logger.info(f"Found {len(image_list)} images on page {page_num + 1}")
            
            pdf_document.close()
            logger.info(f"Successfully extracted text from PDF: {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise
    
    def parse_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX files
        """
        try:
            doc = DocxDocument(file_path)
            text = ""
            
            # Extract paragraphs
            for para in doc.paragraphs:
                text += para.text + "\n"
            
            # Extract table data
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    text += row_text + "\n"
            
            logger.info(f"Successfully extracted text from DOCX: {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path}: {str(e)}")
            raise
    
    def parse_xlsx(self, file_path: str) -> str:
        """
        Extract data from Excel files (BOQ, schedules)
        """
        try:
            workbook = load_workbook(file_path)
            text = ""
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"\n--- Sheet: {sheet_name} ---\n"
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                    text += row_text + "\n"
            
            logger.info(f"Successfully extracted data from XLSX: {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error parsing XLSX {file_path}: {str(e)}")
            raise
    
    def parse_text(self, file_path: str) -> str:
        """
        Read plain text files
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Successfully read text file: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            raise
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        Main method to parse any supported document format
        """
        file_path = str(file_path)
        file_extension = Path(file_path).suffix.lower()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Route to appropriate parser
        if file_extension == '.pdf':
            text = self.parse_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            text = self.parse_docx(file_path)
        elif file_extension == '.xlsx':
            text = self.parse_xlsx(file_path)
        else:  # .txt
            text = self.parse_text(file_path)
        
        return {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'file_type': file_extension,
            'content': text,
            'word_count': len(text.split()),
            'char_count': len(text)
        }


class DataCleaner:
    """
    Clean and normalize extracted text data
    """
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove extra whitespace and normalize line breaks"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def remove_special_characters(text: str, keep_punctuation=True) -> str:
        """Remove unwanted special characters"""
        if keep_punctuation:
            # Keep alphanumeric, punctuation, and whitespace
            text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\-\'"()\[\]]', '', text)
        else:
            # Keep only alphanumeric and whitespace
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text (lowercase, remove accents, etc.)"""
        # Note: For tender analysis, we might want to preserve case
        # This is just a basic normalization
        text = text.strip()
        return text
    
    @staticmethod
    def clean_document(text: str) -> str:
        """Apply all cleaning operations"""
        text = DataCleaner.remove_extra_whitespace(text)
        text = DataCleaner.normalize_text(text)
        return text


class TextChunker:
    """
    Split text into chunks for embedding and processing
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata
        """
        chunks = self.splitter.split_text(text)
        
        chunked_data = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'chunk_id': i,
                'content': chunk,
                'chunk_length': len(chunk),
                'metadata': metadata or {}
            }
            chunked_data.append(chunk_dict)
        
        logger.info(f"Text split into {len(chunked_data)} chunks")
        return chunked_data


class InformationExtractor:
    """
    Extract key information from tender documents
    (BOQ items, clauses, costs, dates, quantities, etc.)
    """
    
    # Regex patterns for common tender information
    PATTERNS = {
        'cost': r'(?:cost|price|amount|rate|fee)[:\s]+[\$£€]?[\d,]+(?:\.\d{2})?',
        'quantity': r'(?:qty|quantity|nos|pieces)[:\s]+\d+(?:\.\d+)?',
        'date': r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
        'reference': r'(?:ref|reference|clause)[:\s]+[A-Z0-9.]+',
        'item_code': r'(?:item|code)[:\s]+[A-Z0-9]+',
    }
    
    @staticmethod
    def extract_boq_items(text: str) -> List[Dict[str, Any]]:
        """
        Extract Bill of Quantities items
        """
        boq_items = []
        
        # Simple pattern matching for BOQ lines
        # In production, this would be more sophisticated
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['item', 'qty', 'rate', 'amount']):
                item = {
                    'line': line.strip(),
                    'extracted_at': 'line_level'
                }
                boq_items.append(item)
        
        return boq_items
    
    @staticmethod
    def extract_costs(text: str) -> List[Dict[str, Any]]:
        """
        Extract cost-related information
        """
        costs = []
        pattern = InformationExtractor.PATTERNS['cost']
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            costs.append({
                'value': match.group(),
                'position': match.start()
            })
        
        return costs
    
    @staticmethod
    def extract_dates(text: str) -> List[Dict[str, Any]]:
        """
        Extract dates (deadlines, submission dates, etc.)
        """
        dates = []
        pattern = InformationExtractor.PATTERNS['date']
        
        matches = re.finditer(pattern, text)
        for match in matches:
            dates.append({
                'value': match.group(),
                'position': match.start()
            })
        
        return dates
    
    @staticmethod
    def extract_quantities(text: str) -> List[Dict[str, Any]]:
        """
        Extract quantity information
        """
        quantities = []
        pattern = InformationExtractor.PATTERNS['quantity']
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            quantities.append({
                'value': match.group(),
                'position': match.start()
            })
        
        return quantities
    
    @staticmethod
    def extract_all_information(text: str) -> Dict[str, Any]:
        """
        Extract all key information from document
        """
        return {
            'boq_items': InformationExtractor.extract_boq_items(text),
            'costs': InformationExtractor.extract_costs(text),
            'dates': InformationExtractor.extract_dates(text),
            'quantities': InformationExtractor.extract_quantities(text)
        }


class ProcessingPipeline:
    """
    Orchestrate the complete data processing pipeline
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.parser = DocumentParser()
        self.cleaner = DataCleaner()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.extractor = InformationExtractor()
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single document
        """
        logger.info(f"Starting processing pipeline for: {file_path}")
        
        # Step 1: Parse document
        parsed_doc = self.parser.parse_document(file_path)
        logger.info(f"Document parsed. Word count: {parsed_doc['word_count']}")
        
        # Step 2: Clean text
        cleaned_text = self.cleaner.clean_document(parsed_doc['content'])
        logger.info("Text cleaned and normalized")
        
        # Step 3: Extract information
        extracted_info = self.extractor.extract_all_information(cleaned_text)
        logger.info(f"Information extracted: {len(extracted_info['boq_items'])} BOQ items found")
        
        # Step 4: Chunk text
        chunks = self.chunker.chunk_text(cleaned_text, {'file_name': parsed_doc['file_name']})
        logger.info(f"Text chunked into {len(chunks)} chunks")
        
        return {
            'document_metadata': {
                'file_name': parsed_doc['file_name'],
                'file_type': parsed_doc['file_type'],
                'word_count': parsed_doc['word_count'],
                'char_count': parsed_doc['char_count']
            },
            'cleaned_text': cleaned_text,
            'extracted_information': extracted_info,
            'chunks': chunks,
            'total_chunks': len(chunks)
        }


if __name__ == "__main__":
    # Example usage
    pipeline = ProcessingPipeline()
    
    # Process a sample document
    # result = pipeline.process_document("data/raw/sample_tender.pdf")
    # print(result)
