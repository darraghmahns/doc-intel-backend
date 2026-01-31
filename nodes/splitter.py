"""
Intelligent Splitter Node - PDF Document Splitting

User Story S1:
As a user, I want multi-document PDFs automatically split into individual 
contracts so each document is processed separately.

User Story S2 (OCR):
As a user, I want scanned/image PDFs converted to searchable text via OCR
so handwritten contracts can be processed.

User Story S4 (Error Handling):
As a user, I want corrupted or password-protected PDFs flagged for manual 
review instead of failing silently.

Acceptance Criteria:
- Docling detects document boundaries
- Outputs separate documents with page ranges
- Preserves text content for each split document
- Handles single-document PDFs gracefully
- OCR runs on image-based pages with >95% accuracy target
- Error handling with clear status messages for unsupported files

Uses Docling to analyze layout and break a large PDF 
into logical sub-documents for processing.
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import cast, Type
from enum import Enum
from datetime import datetime

# Configure logger for splitter metrics (User Story S5)
logger = logging.getLogger(__name__)

from state import DealState

# Type alias for DocumentConverter - used for type checking when Docling is unavailable
DocumentConverterType = Any  # Will be properly typed when Docling is imported

# Conditional import for Docling (may not be available in test environment)
try:
    from docling.document_converter import DocumentConverter as _DocumentConverter
    from docling.datamodel.base_models import InputFormat as _InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions as _PdfPipelineOptions,
        OcrOptions as _OcrOptions,
        EasyOcrOptions as _EasyOcrOptions,
        TesseractOcrOptions as _TesseractOcrOptions,
        TesseractCliOcrOptions as _TesseractCliOcrOptions,
    )
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    DOCLING_AVAILABLE = True
    # Assign to Any-typed variables to satisfy type checker
    DocumentConverter: Any = _DocumentConverter
    InputFormat: Any = _InputFormat
    PdfPipelineOptions: Any = _PdfPipelineOptions
    OcrOptions: Any = _OcrOptions
    EasyOcrOptions: Any = _EasyOcrOptions
    TesseractOcrOptions: Any = _TesseractOcrOptions
    TesseractCliOcrOptions: Any = _TesseractCliOcrOptions
except ImportError:
    DOCLING_AVAILABLE = False
    DocumentConverter = None
    InputFormat = None
    PdfPipelineOptions = None
    OcrOptions = None
    EasyOcrOptions = None
    TesseractOcrOptions = None
    TesseractCliOcrOptions = None
    PyPdfiumDocumentBackend = None


# ============================================================================
# OCR Engine Types
# ============================================================================

class OcrEngine(Enum):
    """Available OCR engines for processing scanned documents."""
    EASYOCR = "easyocr"      # EasyOCR - good accuracy, GPU support
    TESSERACT = "tesseract"  # Tesseract via Python bindings
    TESSERACT_CLI = "tesseract_cli"  # Tesseract via CLI
    AUTO = "auto"            # Auto-detect best available


class PdfStatus(Enum):
    """
    Status of PDF processing (User Story S4).
    
    Used to track and report PDF processing outcomes,
    especially for error cases requiring manual review.
    """
    SUCCESS = "success"                    # PDF processed successfully
    PASSWORD_PROTECTED = "password_protected"  # PDF requires password
    CORRUPTED = "corrupted"                # PDF file is corrupted/invalid
    EMPTY = "empty"                        # PDF has no pages or content
    UNSUPPORTED_FORMAT = "unsupported_format"  # Not a valid PDF
    FILE_NOT_FOUND = "file_not_found"      # File doesn't exist
    PERMISSION_DENIED = "permission_denied"  # Can't read file
    PROCESSING_ERROR = "processing_error"  # Generic processing failure
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"  # Flagged for review


class PdfErrorSeverity(Enum):
    """Severity level for PDF processing errors."""
    INFO = "info"          # Informational, processing continued
    WARNING = "warning"    # Issue detected but processing completed
    ERROR = "error"        # Processing failed, needs attention
    CRITICAL = "critical"  # Cannot process, requires manual intervention


@dataclass
class PdfProcessingResult:
    """
    Result of attempting to process a PDF file (User Story S4).
    
    Provides detailed status and error information for PDFs
    that couldn't be processed normally.
    """
    status: PdfStatus
    success: bool
    message: str
    severity: PdfErrorSeverity = PdfErrorSeverity.INFO
    
    # Original file info
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    
    # Error details
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    
    # Suggestions for resolution
    suggested_action: Optional[str] = None
    requires_manual_review: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "success": self.success,
            "message": self.message,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "error_type": self.error_type,
            "error_details": self.error_details,
            "suggested_action": self.suggested_action,
            "requires_manual_review": self.requires_manual_review,
        }
    
    @staticmethod
    def success_result(file_path: str, message: str = "PDF processed successfully") -> "PdfProcessingResult":
        """Create a successful processing result."""
        return PdfProcessingResult(
            status=PdfStatus.SUCCESS,
            success=True,
            message=message,
            severity=PdfErrorSeverity.INFO,
            file_path=file_path,
        )
    
    @staticmethod
    def password_protected(file_path: str) -> "PdfProcessingResult":
        """Create result for password-protected PDF."""
        return PdfProcessingResult(
            status=PdfStatus.PASSWORD_PROTECTED,
            success=False,
            message="PDF is password-protected and cannot be processed automatically",
            severity=PdfErrorSeverity.ERROR,
            file_path=file_path,
            error_type="PasswordProtected",
            suggested_action="Please provide the PDF password or upload an unprotected version",
            requires_manual_review=True,
        )
    
    @staticmethod
    def corrupted(file_path: str, error_details: Optional[str] = None) -> "PdfProcessingResult":
        """Create result for corrupted PDF."""
        return PdfProcessingResult(
            status=PdfStatus.CORRUPTED,
            success=False,
            message="PDF file appears to be corrupted or invalid",
            severity=PdfErrorSeverity.CRITICAL,
            file_path=file_path,
            error_type="CorruptedFile",
            error_details=error_details,
            suggested_action="Please re-upload the PDF or provide an alternative copy",
            requires_manual_review=True,
        )
    
    @staticmethod
    def empty_pdf(file_path: str) -> "PdfProcessingResult":
        """Create result for empty PDF."""
        return PdfProcessingResult(
            status=PdfStatus.EMPTY,
            success=False,
            message="PDF file contains no pages or extractable content",
            severity=PdfErrorSeverity.ERROR,
            file_path=file_path,
            error_type="EmptyDocument",
            suggested_action="Please upload a PDF with content",
            requires_manual_review=True,
        )
    
    @staticmethod
    def not_a_pdf(file_path: str) -> "PdfProcessingResult":
        """Create result for non-PDF file."""
        return PdfProcessingResult(
            status=PdfStatus.UNSUPPORTED_FORMAT,
            success=False,
            message="File is not a valid PDF document",
            severity=PdfErrorSeverity.ERROR,
            file_path=file_path,
            error_type="InvalidFormat",
            suggested_action="Please upload a valid PDF file",
            requires_manual_review=True,
        )
    
    @staticmethod
    def file_not_found(file_path: str) -> "PdfProcessingResult":
        """Create result for missing file."""
        return PdfProcessingResult(
            status=PdfStatus.FILE_NOT_FOUND,
            success=False,
            message=f"PDF file not found: {file_path}",
            severity=PdfErrorSeverity.ERROR,
            file_path=file_path,
            error_type="FileNotFound",
            suggested_action="Please check the file path and try again",
            requires_manual_review=False,
        )
    
    @staticmethod
    def processing_error(file_path: str, error: Exception) -> "PdfProcessingResult":
        """Create result for generic processing error."""
        return PdfProcessingResult(
            status=PdfStatus.PROCESSING_ERROR,
            success=False,
            message=f"Error processing PDF: {str(error)}",
            severity=PdfErrorSeverity.ERROR,
            file_path=file_path,
            error_type=type(error).__name__,
            error_details=str(error),
            suggested_action="Please try again or contact support if the issue persists",
            requires_manual_review=True,
        )


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OcrConfig:
    """
    Configuration for OCR processing of scanned/image PDFs.
    
    User Story S2: Enable OCR for scanned documents with >95% accuracy target.
    """
    
    # Enable OCR processing
    enabled: bool = True
    
    # OCR engine to use
    engine: OcrEngine = field(default=OcrEngine.AUTO)
    
    # Languages for OCR (ISO 639-1 codes)
    languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Force OCR on all pages (even if text is detected)
    force_full_page_ocr: bool = False
    
    # Threshold for bitmap area to trigger OCR (0.0-1.0)
    # Lower = more aggressive OCR, Higher = only mostly-image pages
    bitmap_area_threshold: float = 0.05
    
    # Confidence threshold for OCR results (0.0-1.0)
    # Results below this are flagged as low-confidence
    confidence_threshold: float = 0.5
    
    # Use GPU acceleration if available (EasyOCR only)
    use_gpu: Optional[bool] = None  # None = auto-detect
    
    # Path to Tesseract binary (for TESSERACT_CLI engine)
    tesseract_cmd: Optional[str] = None
    
    # Tesseract PSM (Page Segmentation Mode)
    # 3 = Fully automatic page segmentation (default)
    # 6 = Assume uniform block of text
    # 11 = Sparse text, find as much text as possible
    tesseract_psm: Optional[int] = None


@dataclass 
class OcrResult:
    """Result of OCR processing for a page."""
    
    page_number: int
    text: str
    confidence: float  # 0.0 to 1.0
    is_scanned: bool   # True if page was primarily image-based
    ocr_engine_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def meets_accuracy_target(self) -> bool:
        """Check if OCR result meets the >95% accuracy target."""
        return self.confidence >= 0.95


@dataclass
class SplitterMetrics:
    """
    Performance metrics for PDF splitting operations.
    
    User Story S5: Track processing time, page count, split count per document.
    """
    
    # File information
    file_path: str = ""
    file_size_bytes: Optional[int] = None
    
    # Timing metrics (all in milliseconds)
    total_processing_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    boundary_detection_time_ms: float = 0.0
    splitting_time_ms: float = 0.0
    
    # Document metrics
    total_pages: int = 0
    documents_created: int = 0
    pages_per_document: List[int] = field(default_factory=list)
    
    # OCR metrics
    scanned_pages: int = 0
    ocr_time_ms: float = 0.0
    avg_ocr_confidence: float = 0.0
    pages_below_accuracy_target: int = 0
    
    # Performance indicators
    pages_per_second: float = 0.0
    is_slow: bool = False  # True if processing took longer than expected
    slow_threshold_ms: float = 5000.0  # 5 seconds default threshold
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics after timing data is collected."""
        if self.total_processing_time_ms > 0 and self.total_pages > 0:
            self.pages_per_second = (self.total_pages / self.total_processing_time_ms) * 1000
        
        self.is_slow = self.total_processing_time_ms > self.slow_threshold_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/serialization."""
        return {
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "validation_time_ms": round(self.validation_time_ms, 2),
            "extraction_time_ms": round(self.extraction_time_ms, 2),
            "boundary_detection_time_ms": round(self.boundary_detection_time_ms, 2),
            "splitting_time_ms": round(self.splitting_time_ms, 2),
            "total_pages": self.total_pages,
            "documents_created": self.documents_created,
            "pages_per_document": self.pages_per_document,
            "scanned_pages": self.scanned_pages,
            "ocr_time_ms": round(self.ocr_time_ms, 2),
            "avg_ocr_confidence": round(self.avg_ocr_confidence, 4),
            "pages_below_accuracy_target": self.pages_below_accuracy_target,
            "pages_per_second": round(self.pages_per_second, 2),
            "is_slow": self.is_slow,
            "timestamp": self.timestamp,
        }
    
    def log_summary(self) -> str:
        """Generate a human-readable summary for logging."""
        lines = [
            f"📊 Splitter Metrics Summary",
            f"   File: {os.path.basename(self.file_path) if self.file_path else 'N/A'}",
            f"   Total Time: {self.total_processing_time_ms:.1f}ms",
            f"   Pages: {self.total_pages} → {self.documents_created} document(s)",
        ]
        
        if self.pages_per_document:
            pages_str = ", ".join(str(p) for p in self.pages_per_document)
            lines.append(f"   Pages per doc: [{pages_str}]")
        
        if self.scanned_pages > 0:
            lines.append(f"   OCR: {self.scanned_pages} pages, {self.avg_ocr_confidence:.1%} avg confidence")
        
        lines.append(f"   Speed: {self.pages_per_second:.1f} pages/sec")
        
        if self.is_slow:
            lines.append(f"   ⚠️  SLOW: Processing exceeded {self.slow_threshold_ms}ms threshold")
        
        return "\n".join(lines)


def log_splitter_metrics(metrics: SplitterMetrics) -> None:
    """
    Log splitter metrics using the standard logger.
    
    User Story S5: Provide structured logging for admin monitoring.
    """
    # Calculate derived metrics first
    metrics.calculate_derived_metrics()
    
    # Log human-readable summary
    logger.info(metrics.log_summary())
    
    # Log structured data for programmatic access
    logger.debug(f"Splitter metrics: {metrics.to_dict()}")
    
    # Warn if processing was slow
    if metrics.is_slow:
        logger.warning(
            f"Slow PDF processing detected: {metrics.file_path} "
            f"took {metrics.total_processing_time_ms:.1f}ms "
            f"({metrics.total_pages} pages, {metrics.documents_created} docs)"
        )


@dataclass
class SplitterConfig:
    """Configuration for document splitting behavior."""
    
    # Document boundary detection patterns
    boundary_patterns: List[str] = field(default_factory=lambda: [
        r"^page\s+1\s+of\s+\d+",  # "Page 1 of X"
        r"^FORM\s+\w+",  # "FORM 123"
        r"^(?:RESIDENTIAL|COMMERCIAL)\s+.*(?:AGREEMENT|CONTRACT)",
        r"^DISCLOSURE\s+STATEMENT",
        r"^ADDENDUM",
        r"^COUNTER\s*OFFER",
        r"^AMENDMENT",
        r"^SELLER'S?\s+PROPERTY\s+DISCLOSURE",
        r"^LEAD[- ]BASED\s+PAINT",
        r"^BUYER'S?\s+INSPECTION",
    ])
    
    # Minimum pages to consider as separate document
    min_pages_per_doc: int = 1
    
    # Use mock mode when PDF not found or Docling unavailable
    use_mock_on_failure: bool = True
    
    # Enable page-level analysis
    analyze_page_headers: bool = True
    
    # OCR configuration for scanned documents
    ocr: OcrConfig = field(default_factory=OcrConfig)


# ============================================================================
# OCR Engine Detection & Configuration
# ============================================================================

def detect_available_ocr_engine() -> OcrEngine:
    """
    Detect the best available OCR engine on the system.
    
    Returns:
        OcrEngine: The recommended OCR engine to use
    """
    # Try EasyOCR first (best accuracy, GPU support)
    try:
        import easyocr
        return OcrEngine.EASYOCR
    except ImportError:
        pass
    
    # Try Tesseract Python bindings
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return OcrEngine.TESSERACT
    except (ImportError, Exception):
        pass
    
    # Try Tesseract CLI
    import shutil
    if shutil.which("tesseract"):
        return OcrEngine.TESSERACT_CLI
    
    # Fallback to EasyOCR (Docling's default, will install if needed)
    return OcrEngine.EASYOCR


def create_ocr_options(config: OcrConfig) -> Any:
    """
    Create Docling OCR options based on configuration.
    
    Args:
        config: OCR configuration
        
    Returns:
        Docling OcrOptions subclass instance
    """
    if not DOCLING_AVAILABLE:
        return None
    
    engine = config.engine
    if engine == OcrEngine.AUTO:
        engine = detect_available_ocr_engine()
    
    if engine == OcrEngine.EASYOCR:
        return EasyOcrOptions(
            lang=config.languages,
            force_full_page_ocr=config.force_full_page_ocr,
            bitmap_area_threshold=config.bitmap_area_threshold,
            confidence_threshold=config.confidence_threshold,
            use_gpu=config.use_gpu,
        )
    elif engine == OcrEngine.TESSERACT:
        return TesseractOcrOptions(
            lang=config.languages,
            force_full_page_ocr=config.force_full_page_ocr,
            bitmap_area_threshold=config.bitmap_area_threshold,
            psm=config.tesseract_psm,
        )
    elif engine == OcrEngine.TESSERACT_CLI:
        opts = TesseractCliOcrOptions(
            lang=config.languages,
            force_full_page_ocr=config.force_full_page_ocr,
            bitmap_area_threshold=config.bitmap_area_threshold,
            psm=config.tesseract_psm,
        )
        if config.tesseract_cmd:
            opts.tesseract_cmd = config.tesseract_cmd
        return opts
    
    # Fallback to base options
    return OcrOptions(
        lang=config.languages,
        force_full_page_ocr=config.force_full_page_ocr,
        bitmap_area_threshold=config.bitmap_area_threshold,
    )


def is_page_scanned(page_text: str, image_ratio: float = 0.0) -> bool:
    """
    Determine if a page appears to be scanned/image-based.
    
    Args:
        page_text: Extracted text from the page
        image_ratio: Ratio of page area covered by images (0.0-1.0)
        
    Returns:
        True if the page appears to be scanned
    """
    # If very little text, likely scanned
    if len(page_text.strip()) < 50:
        return True
    
    # If high image ratio, likely scanned
    if image_ratio > 0.8:
        return True
    
    # Check for OCR artifacts (common in poorly scanned docs)
    ocr_artifacts = [
        r'[|l1]{3,}',  # Repeated vertical bars (misread text)
        r'[_]{5,}',    # Long underscores
        r'[\[\]]{3,}', # Repeated brackets
    ]
    artifact_count = sum(
        len(re.findall(pattern, page_text)) 
        for pattern in ocr_artifacts
    )
    
    return artifact_count > 5


# ============================================================================
# Document Boundary Detection
# ============================================================================

@dataclass
class PageMapping:
    """
    Maps a page in a split document to its original location.
    
    User Story S3: Preserve page numbers after splitting for reference.
    
    Example:
        If a 10-page PDF is split into 2 documents:
        - Doc 1: pages 1-6 (original), mapped to pages 1-6 (local)
        - Doc 2: pages 7-10 (original), mapped to pages 1-4 (local)
    """
    original_page: int      # Page number in the original PDF (1-indexed)
    local_page: int         # Page number within this split document (1-indexed)
    has_content: bool = True  # Whether the page has meaningful content
    is_scanned: bool = False  # Whether this page was OCR'd
    ocr_confidence: Optional[float] = None  # OCR confidence if scanned
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_page": self.original_page,
            "local_page": self.local_page,
            "has_content": self.has_content,
            "is_scanned": self.is_scanned,
            "ocr_confidence": self.ocr_confidence,
        }


@dataclass
class DocumentBoundary:
    """Represents a detected document boundary within a PDF."""
    start_page: int  # 1-indexed
    end_page: int    # 1-indexed, inclusive
    detected_title: Optional[str] = None
    confidence: float = 1.0
    boundary_type: str = "header_pattern"  # header_pattern, page_break, form_number


def detect_document_boundaries(
    pages_text: List[str],
    config: SplitterConfig,
) -> List[DocumentBoundary]:
    """
    Analyze page text to detect document boundaries.
    
    Looks for patterns like:
    - "Page 1 of X" headers
    - Document type headers (AGREEMENT, DISCLOSURE, ADDENDUM)
    - Form numbers at top of pages
    
    Args:
        pages_text: List of text content for each page (0-indexed)
        config: Splitter configuration
    
    Returns:
        List of DocumentBoundary objects representing split points
    """
    boundaries: List[DocumentBoundary] = []
    current_start = 1  # 1-indexed
    current_title: Optional[str] = None
    
    compiled_patterns = [
        re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        for pattern in config.boundary_patterns
    ]
    
    for page_idx, page_text in enumerate(pages_text):
        page_num = page_idx + 1  # Convert to 1-indexed
        
        # Check first ~500 chars of page for boundary patterns
        header_text = page_text[:500] if page_text else ""
        
        is_new_document = False
        detected_title = None
        
        for pattern in compiled_patterns:
            match = pattern.search(header_text)
            if match:
                # Check if this is "Page 1" pattern specifically
                if "page" in pattern.pattern.lower() and "1" in match.group():
                    is_new_document = True
                    detected_title = _extract_document_title(page_text)
                    break
                # Other patterns indicate document start on any page
                elif page_num > 1:  # Don't split on first page
                    is_new_document = True
                    detected_title = match.group().strip()
                    break
        
        if is_new_document and page_num > current_start:
            # Close previous document
            boundaries.append(DocumentBoundary(
                start_page=current_start,
                end_page=page_num - 1,
                detected_title=current_title,
                confidence=0.9,
                boundary_type="header_pattern",
            ))
            current_start = page_num
            current_title = detected_title
        elif page_num == 1:
            current_title = detected_title or _extract_document_title(page_text)
    
    # Add final document
    if pages_text:
        boundaries.append(DocumentBoundary(
            start_page=current_start,
            end_page=len(pages_text),
            detected_title=current_title,
            confidence=0.9,
            boundary_type="header_pattern",
        ))
    
    return boundaries


def _extract_document_title(page_text: str) -> Optional[str]:
    """Extract a potential document title from the first few lines of a page."""
    if not page_text:
        return None
    
    lines = page_text.strip().split("\n")[:5]  # Check first 5 lines
    
    # Look for common title patterns
    title_patterns = [
        r"^(?:RESIDENTIAL|COMMERCIAL).*(?:AGREEMENT|CONTRACT)",
        r"^.*DISCLOSURE.*STATEMENT",
        r"^ADDENDUM.*",
        r"^COUNTER\s*OFFER.*",
        r"^AMENDMENT.*",
        r"^.*BUY.*SELL.*AGREEMENT",
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:  # Reasonable title length
            for pattern in title_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return line
    
    # Fallback: use first non-empty line if it looks like a title
    for line in lines:
        line = line.strip()
        if len(line) > 10 and len(line) < 80 and line.isupper():
            return line
    
    return None


# ============================================================================
# Page Mapping Utilities (User Story S3)
# ============================================================================

def create_page_mappings(
    boundary: DocumentBoundary,
    ocr_results: List[OcrResult],
    pages_text: List[str],
) -> List[PageMapping]:
    """
    Create detailed page mappings for a split document.
    
    Maps each page in the split document to its original location
    in the source PDF, including OCR metadata.
    
    Args:
        boundary: The document boundary defining page range
        ocr_results: OCR results for all pages
        pages_text: Text content for all pages (0-indexed)
        
    Returns:
        List of PageMapping objects for this document
    """
    mappings: List[PageMapping] = []
    
    # Create OCR lookup by page number
    ocr_by_page = {r.page_number: r for r in ocr_results}
    
    for original_page in range(boundary.start_page, boundary.end_page + 1):
        # Local page is 1-indexed within this split document
        local_page = original_page - boundary.start_page + 1
        
        # Get page text (0-indexed in pages_text)
        page_idx = original_page - 1
        page_text = pages_text[page_idx] if page_idx < len(pages_text) else ""
        
        # Check if page has meaningful content
        has_content = len(page_text.strip()) > 20
        
        # Get OCR info for this page
        ocr_result = ocr_by_page.get(original_page)
        is_scanned = ocr_result.is_scanned if ocr_result else False
        ocr_confidence = ocr_result.confidence if ocr_result and is_scanned else None
        
        mappings.append(PageMapping(
            original_page=original_page,
            local_page=local_page,
            has_content=has_content,
            is_scanned=is_scanned,
            ocr_confidence=ocr_confidence,
        ))
    
    return mappings


def format_page_reference(original_page: int, total_original_pages: int) -> str:
    """
    Format a page reference for display.
    
    Args:
        original_page: The original page number
        total_original_pages: Total pages in original document
        
    Returns:
        Formatted string like "Page 5 of 12 (original)"
    """
    return f"Page {original_page} of {total_original_pages} (original)"


def get_original_page_for_local(
    local_page: int,
    page_mappings: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Look up the original page number for a local page in a split document.
    
    Args:
        local_page: Page number within the split document (1-indexed)
        page_mappings: List of page mapping dictionaries
        
    Returns:
        Original page number, or None if not found
    """
    for mapping in page_mappings:
        if mapping.get("local_page") == local_page:
            return mapping.get("original_page")
    return None


def get_local_page_for_original(
    original_page: int,
    page_mappings: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Look up the local page number for an original page.
    
    Args:
        original_page: Page number in the original PDF (1-indexed)
        page_mappings: List of page mapping dictionaries
        
    Returns:
        Local page number within the split document, or None if not in this doc
    """
    for mapping in page_mappings:
        if mapping.get("original_page") == original_page:
            return mapping.get("local_page")
    return None


# ============================================================================
# PDF Validation and Error Handling (User Story S4)
# ============================================================================

def validate_pdf_file(file_path: str) -> PdfProcessingResult:
    """
    Validate a PDF file before processing.
    
    Checks for:
    - File existence
    - File permissions
    - Valid PDF format (magic bytes)
    - Password protection
    - Corruption
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        PdfProcessingResult indicating validation outcome
    """
    # Check file existence
    if not os.path.exists(file_path):
        return PdfProcessingResult.file_not_found(file_path)
    
    # Check file permissions
    if not os.access(file_path, os.R_OK):
        return PdfProcessingResult(
            status=PdfStatus.PERMISSION_DENIED,
            success=False,
            message=f"Cannot read file: {file_path}",
            severity=PdfErrorSeverity.ERROR,
            file_path=file_path,
            error_type="PermissionDenied",
            suggested_action="Check file permissions and try again",
            requires_manual_review=False,
        )
    
    # Get file size
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        file_size = None
    
    # Check if file is empty
    if file_size == 0:
        return PdfProcessingResult(
            status=PdfStatus.EMPTY,
            success=False,
            message="PDF file is empty (0 bytes)",
            severity=PdfErrorSeverity.ERROR,
            file_path=file_path,
            file_size_bytes=0,
            error_type="EmptyFile",
            suggested_action="Please upload a valid PDF file",
            requires_manual_review=True,
        )
    
    # Check PDF magic bytes
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            
            # PDF files should start with %PDF-
            if not header.startswith(b'%PDF-'):
                return PdfProcessingResult.not_a_pdf(file_path)
            
            # Check for encryption dictionary (basic password check)
            # Read more of the file to look for encryption markers
            f.seek(0)
            content = f.read(min(file_size or 10000, 10000))  # Read first 10KB
            
            # Look for encryption markers in PDF
            if b'/Encrypt' in content:
                return PdfProcessingResult.password_protected(file_path)
                
    except IOError as e:
        return PdfProcessingResult(
            status=PdfStatus.CORRUPTED,
            success=False,
            message=f"Cannot read PDF file: {str(e)}",
            severity=PdfErrorSeverity.CRITICAL,
            file_path=file_path,
            error_type="IOError",
            error_details=str(e),
            suggested_action="File may be corrupted. Please re-upload.",
            requires_manual_review=True,
        )
    
    # If we get here, basic validation passed
    result = PdfProcessingResult.success_result(file_path, "PDF validation passed")
    result.file_size_bytes = file_size
    return result


def check_pdf_with_docling(file_path: str) -> PdfProcessingResult:
    """
    Perform deeper PDF validation using Docling.
    
    This catches issues that basic validation might miss:
    - Corrupted internal structure
    - Password protection detected during parsing
    - Empty or unreadable pages
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        PdfProcessingResult with detailed status
    """
    if not DOCLING_AVAILABLE:
        return PdfProcessingResult.success_result(
            file_path, 
            "Docling not available, skipping deep validation"
        )
    
    try:
        # Try to create a converter and parse the PDF
        converter: Any = DocumentConverter()
        result = converter.convert(file_path)
        
        doc = result.document
        
        # Check if document has any content
        if not hasattr(doc, 'texts') or not doc.texts:
            # Check if it has any pages at all
            if hasattr(doc, 'pages') and not doc.pages:
                return PdfProcessingResult.empty_pdf(file_path)
        
        # Get page count
        page_count = len(doc.pages) if hasattr(doc, 'pages') and doc.pages else 0
        
        if page_count == 0:
            return PdfProcessingResult.empty_pdf(file_path)
        
        return PdfProcessingResult.success_result(
            file_path,
            f"PDF validated successfully ({page_count} pages)"
        )
        
    except Exception as e:
        error_str = str(e).lower()
        
        # Check for password-related errors
        if 'password' in error_str or 'encrypted' in error_str:
            return PdfProcessingResult.password_protected(file_path)
        
        # Check for corruption-related errors
        if any(term in error_str for term in ['corrupt', 'invalid', 'malformed', 'damaged']):
            return PdfProcessingResult.corrupted(file_path, str(e))
        
        # Generic processing error
        return PdfProcessingResult.processing_error(file_path, e)


def get_pdf_error_message(result: PdfProcessingResult) -> str:
    """
    Get a user-friendly error message for a PDF processing result.
    
    Args:
        result: The processing result
        
    Returns:
        Formatted error message string
    """
    if result.success:
        return result.message
    
    lines = [
        f"⚠️  PDF Processing Error: {result.message}",
    ]
    
    if result.error_type:
        lines.append(f"   Error Type: {result.error_type}")
    
    if result.error_details:
        lines.append(f"   Details: {result.error_details}")
    
    if result.suggested_action:
        lines.append(f"   Action: {result.suggested_action}")
    
    if result.requires_manual_review:
        lines.append("   ⚡ This file has been flagged for manual review")
    
    return "\n".join(lines)


# ============================================================================
# Docling Integration with OCR Support
# ============================================================================

@dataclass
class PageExtractionResult:
    """Result of extracting text from a single page."""
    page_number: int
    text: str
    is_scanned: bool = False
    ocr_confidence: Optional[float] = None
    ocr_engine: Optional[str] = None


def extract_pages_with_docling(
    pdf_path: str,
    ocr_config: Optional[OcrConfig] = None,
) -> Tuple[List[str], int, List[OcrResult]]:
    """
    Use Docling to extract text from each page of a PDF.
    
    Supports OCR for scanned/image-based PDFs (User Story S2).
    
    Args:
        pdf_path: Path to the PDF file
        ocr_config: Optional OCR configuration for scanned documents
    
    Returns:
        Tuple of (list of page texts, total page count, OCR results)
    
    Raises:
        FileNotFoundError: If PDF doesn't exist
        RuntimeError: If Docling fails to process
    """
    if not DOCLING_AVAILABLE:
        raise RuntimeError("Docling is not available")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Configure pipeline options with OCR if enabled
    pipeline_options = None
    ocr_engine_name = None
    
    if ocr_config and ocr_config.enabled:
        ocr_options = create_ocr_options(ocr_config)
        if ocr_options:
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                ocr_options=ocr_options,
            )
            # Determine engine name for reporting
            if ocr_config.engine == OcrEngine.AUTO:
                detected = detect_available_ocr_engine()
                ocr_engine_name = detected.value
            else:
                ocr_engine_name = ocr_config.engine.value
    
    # Initialize Docling converter with or without OCR
    if pipeline_options:
        converter: Any = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )
    else:
        converter = DocumentConverter()
    
    # Convert the PDF
    import time
    start_time = time.time()
    result = converter.convert(pdf_path)
    processing_time = (time.time() - start_time) * 1000  # ms
    
    # Extract text by page
    doc = result.document
    
    # Group text items by page number
    page_texts: Dict[int, List[str]] = defaultdict(list)
    
    # Track OCR results per page
    ocr_results: List[OcrResult] = []
    
    # doc.texts contains all TextItem objects with provenance info
    if hasattr(doc, 'texts') and doc.texts:
        for text_item in doc.texts:
            if hasattr(text_item, 'prov') and text_item.prov:
                for prov in text_item.prov:
                    if hasattr(prov, 'page_no') and hasattr(text_item, 'text'):
                        page_texts[prov.page_no].append(text_item.text)
    
    # Also check titles and section headers
    if hasattr(doc, 'iterate_items'):
        for item, level in doc.iterate_items():
            if hasattr(item, 'text') and hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        if item.text not in page_texts[prov.page_no]:
                            page_texts[prov.page_no].append(item.text)
    
    # Convert to ordered list and build OCR results
    pages_list: List[str] = []
    if page_texts:
        max_page = max(page_texts.keys())
        for i in range(1, max_page + 1):
            page_text = "\n".join(page_texts.get(i, []))
            pages_list.append(page_text)
            
            # Create OCR result for this page
            is_scanned = is_page_scanned(page_text)
            
            # Estimate confidence based on text characteristics
            confidence = estimate_ocr_confidence(page_text) if is_scanned else 1.0
            
            ocr_results.append(OcrResult(
                page_number=i,
                text=page_text,
                confidence=confidence,
                is_scanned=is_scanned,
                ocr_engine_used=ocr_engine_name if is_scanned else None,
                processing_time_ms=processing_time / max_page if is_scanned else None,
            ))
    else:
        # Fallback: treat entire document as single page
        full_text = doc.export_to_markdown()
        pages_list = [full_text]
        ocr_results.append(OcrResult(
            page_number=1,
            text=full_text,
            confidence=1.0,
            is_scanned=False,
        ))
    
    return pages_list, len(pages_list), ocr_results


def estimate_ocr_confidence(text: str) -> float:
    """
    Estimate OCR confidence based on text characteristics.
    
    This is a heuristic when actual confidence scores aren't available.
    
    Args:
        text: The extracted text
        
    Returns:
        Estimated confidence score (0.0 to 1.0)
    """
    if not text or len(text.strip()) < 10:
        return 0.5  # Very little text = uncertain
    
    # Start with high confidence
    confidence = 1.0
    
    # Penalize for common OCR errors
    penalties = [
        # Unusual character sequences
        (r'[|l1Il]{4,}', 0.1),  # Vertical bar confusion
        (r'[0Oo]{3,}', 0.05),   # Zero/O confusion  
        (r'[^a-zA-Z0-9\s.,;:!?\'"()\-$%@#&*/\\]{3,}', 0.15),  # Garbage chars
        
        # Word-level issues
        (r'\b[bcdfghjklmnpqrstvwxyz]{5,}\b', 0.08),  # Consonant-only "words"
        (r'\b[aeiou]{4,}\b', 0.05),  # Vowel-only sequences
        
        # Spacing issues
        (r'\s{3,}', 0.02),  # Excessive whitespace
        (r'[a-z][A-Z][a-z]', 0.03),  # Mid-word caps (OCR artifact)
    ]
    
    text_lower = text.lower()
    for pattern, penalty in penalties:
        matches = len(re.findall(pattern, text))
        confidence -= min(penalty * matches, 0.2)  # Cap penalty per pattern
    
    # Bonus for well-formed content
    bonuses = [
        (r'\b(the|and|of|to|in|for|is|on|that|by)\b', 0.01),  # Common words
        (r'\d{1,2}/\d{1,2}/\d{2,4}', 0.02),  # Dates
        (r'\$[\d,]+\.?\d*', 0.02),  # Currency
        (r'\b[A-Z][a-z]+\b', 0.005),  # Proper capitalization
    ]
    
    for pattern, bonus in bonuses:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        confidence += min(bonus * matches, 0.1)  # Cap bonus per pattern
    
    return max(0.0, min(1.0, confidence))


def extract_pages_with_ocr(
    pdf_path: str,
    config: SplitterConfig,
) -> Tuple[List[str], int, List[OcrResult]]:
    """
    Extract pages with full OCR support and detailed reporting.
    
    This is a convenience wrapper around extract_pages_with_docling
    that uses the OCR config from SplitterConfig.
    
    Args:
        pdf_path: Path to the PDF file
        config: Splitter configuration with OCR settings
        
    Returns:
        Tuple of (page texts, page count, OCR results)
    """
    return extract_pages_with_docling(pdf_path, config.ocr)


def split_pdf_with_docling(
    pdf_path: str,
    config: SplitterConfig,
) -> Tuple[List[Dict[str, Any]], List[OcrResult]]:
    """
    Split a PDF into logical documents using Docling with OCR support.
    
    Args:
        pdf_path: Path to the PDF file
        config: Splitter configuration
    
    Returns:
        Tuple of (list of split document dicts, OCR results)
    """
    print(f"   Processing PDF with Docling: {pdf_path}")
    
    # Extract pages with OCR if enabled
    pages_text, total_pages, ocr_results = extract_pages_with_docling(pdf_path, config.ocr)
    print(f"   Extracted {total_pages} pages")
    
    # Report OCR status
    scanned_pages = [r for r in ocr_results if r.is_scanned]
    if scanned_pages:
        avg_confidence = sum(r.confidence for r in scanned_pages) / len(scanned_pages)
        low_confidence = [r for r in scanned_pages if not r.meets_accuracy_target]
        print(f"   OCR processed {len(scanned_pages)} scanned page(s), avg confidence: {avg_confidence:.1%}")
        if low_confidence:
            print(f"   ⚠️  {len(low_confidence)} page(s) below 95% accuracy target")
    
    # Detect document boundaries
    boundaries = detect_document_boundaries(pages_text, config)
    print(f"   Detected {len(boundaries)} document(s)")
    
    # Build split documents with page mappings (User Story S3)
    split_docs = []
    for idx, boundary in enumerate(boundaries):
        # Combine text from pages in this document
        start_idx = boundary.start_page - 1  # Convert to 0-indexed
        end_idx = boundary.end_page  # end_page is inclusive, so no -1
        doc_text = "\n\n".join(pages_text[start_idx:end_idx])
        
        # Get OCR info for pages in this document
        doc_ocr_results = [r for r in ocr_results if boundary.start_page <= r.page_number <= boundary.end_page]
        has_scanned_pages = any(r.is_scanned for r in doc_ocr_results)
        avg_ocr_confidence = (
            sum(r.confidence for r in doc_ocr_results) / len(doc_ocr_results)
            if doc_ocr_results else 1.0
        )
        
        # Create detailed page mappings (S3)
        page_mappings = create_page_mappings(boundary, ocr_results, pages_text)
        
        # Calculate page statistics
        page_count = boundary.end_page - boundary.start_page + 1
        
        split_docs.append({
            "id": idx + 1,
            # Original page range (simple format)
            "page_range": [boundary.start_page, boundary.end_page],
            # Detailed page mapping (S3)
            "page_mappings": [pm.to_dict() for pm in page_mappings],
            # Page statistics
            "page_count": page_count,
            "original_page_start": boundary.start_page,
            "original_page_end": boundary.end_page,
            "total_original_pages": total_pages,
            # Content
            "raw_text": doc_text,
            "type": None,  # Will be set by classifier
            "detected_title": boundary.detected_title,
            "confidence": boundary.confidence,
            # OCR metadata
            "has_scanned_pages": has_scanned_pages,
            "ocr_confidence": avg_ocr_confidence,
            "meets_accuracy_target": avg_ocr_confidence >= 0.95,
        })
        
        title_display = boundary.detected_title or "(untitled)"
        ocr_indicator = " [OCR]" if has_scanned_pages else ""
        print(f"   Document {idx + 1}: Pages {boundary.start_page}-{boundary.end_page} of {total_pages} - {title_display}{ocr_indicator}")
    
    return split_docs, ocr_results


# ============================================================================
# Mock Data for Testing
# ============================================================================

MOCK_BUY_SELL_CONTENT = """
RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT

Property Address: 324 Muir Lane, Fort Benton, MT 59442

BUYER: John Smith and Jane Smith
SELLER: Robert Johnson

Purchase Price: $350,000.00
Earnest Money: $10,000.00
Earnest Money to be held by: First American Title Company

Offer Date: December 1, 2024
Closing Date: January 15, 2025
Inspection Date: December 10, 2024

TERMS AND CONDITIONS:
This agreement is contingent upon buyer obtaining financing...

_______________________________________________
Buyer's Signature                    Date

_______________________________________________
Buyer's Signature                    Date

_______________________________________________
Seller's Signature                   Date

_______________________________________________
Listing Agent Signature              Date

_______________________________________________
Buyer's Agent Signature              Date
"""

MOCK_DISCLOSURE_CONTENT = """
SELLER'S PROPERTY DISCLOSURE STATEMENT

Property: 324 Muir Lane, Fort Benton, MT 59442

The Seller makes the following disclosures regarding the property...

MOLD DISCLOSURE:
Seller is not aware of any mold or mildew issues...

LEAD PAINT DISCLOSURE:
This property was built after 1978 and is not subject to lead paint disclosure requirements.

Seller's Initials: _______  Date: _______
Buyer's Initials: _______   Date: _______
"""


def get_mock_split_docs() -> List[Dict[str, Any]]:
    """
    Return mock split documents for testing without a real PDF.
    
    Includes page mappings per User Story S3.
    """
    return [
        {
            "id": 1,
            "page_range": [1, 4],
            # Detailed page mappings (S3)
            "page_mappings": [
                {"original_page": 1, "local_page": 1, "has_content": True, "is_scanned": False, "ocr_confidence": None},
                {"original_page": 2, "local_page": 2, "has_content": True, "is_scanned": False, "ocr_confidence": None},
                {"original_page": 3, "local_page": 3, "has_content": True, "is_scanned": False, "ocr_confidence": None},
                {"original_page": 4, "local_page": 4, "has_content": True, "is_scanned": False, "ocr_confidence": None},
            ],
            "page_count": 4,
            "original_page_start": 1,
            "original_page_end": 4,
            "total_original_pages": 6,
            "raw_text": MOCK_BUY_SELL_CONTENT,
            "type": None,
            "detected_title": "RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT",
            "confidence": 1.0,
            "has_scanned_pages": False,
            "ocr_confidence": 1.0,
            "meets_accuracy_target": True,
        },
        {
            "id": 2,
            "page_range": [5, 6],
            # Detailed page mappings (S3)
            "page_mappings": [
                {"original_page": 5, "local_page": 1, "has_content": True, "is_scanned": False, "ocr_confidence": None},
                {"original_page": 6, "local_page": 2, "has_content": True, "is_scanned": False, "ocr_confidence": None},
            ],
            "page_count": 2,
            "original_page_start": 5,
            "original_page_end": 6,
            "total_original_pages": 6,
            "raw_text": MOCK_DISCLOSURE_CONTENT,
            "type": None,
            "detected_title": "SELLER'S PROPERTY DISCLOSURE STATEMENT",
            "confidence": 1.0,
            "has_scanned_pages": False,
            "ocr_confidence": 1.0,
            "meets_accuracy_target": True,
        },
    ]


# ============================================================================
# Main Node Function
# ============================================================================

def intelligent_splitter_node(state: DealState) -> dict:
    """
    Node B: Intelligent Splitter
    
    Uses Docling to analyze layout and break a large PDF into logical sub-documents.
    Supports OCR for scanned/image PDFs.
    
    User Story S1: Multi-document PDFs automatically split into individual contracts
    User Story S2: Scanned/image PDFs converted to searchable text via OCR
    User Story S4: Corrupted or password-protected PDFs flagged for manual review
    User Story S5: Performance metrics logged for admin monitoring
    
    Process:
    1. Validate PDF file (existence, permissions, format, encryption)
    2. Load PDF with Docling (with OCR if enabled)
    3. Extract text from each page (OCR for image-based pages)
    4. Detect document boundaries (headers, form changes)
    5. Split into separate logical documents
    6. Log performance metrics
    7. Return list of split documents with OCR metadata and processing status
    """
    print("--- NODE: Intelligent Splitter ---")
    
    # User Story S5: Start tracking metrics
    start_time = time.time()
    metrics = SplitterMetrics()
    
    pdf_path = state.get("raw_pdf_path", "")
    metrics.file_path = pdf_path
    
    # Get file size if available
    if pdf_path and os.path.exists(pdf_path):
        try:
            metrics.file_size_bytes = os.path.getsize(pdf_path)
        except OSError:
            pass
    
    # Build configuration from environment/state
    ocr_config = OcrConfig(
        enabled=os.getenv("OCR_ENABLED", "true").lower() == "true",
        force_full_page_ocr=os.getenv("OCR_FORCE_FULL_PAGE", "false").lower() == "true",
    )
    
    # Parse OCR engine from environment
    ocr_engine_str = os.getenv("OCR_ENGINE", "auto").lower()
    try:
        ocr_config.engine = OcrEngine(ocr_engine_str)
    except ValueError:
        ocr_config.engine = OcrEngine.AUTO
    
    config = SplitterConfig(ocr=ocr_config)
    
    # Check if we should use mock mode
    use_mock = os.getenv("USE_MOCK_SPLITTER", "true").lower() == "true"
    
    if use_mock:
        print("   Running in MOCK mode (set USE_MOCK_SPLITTER=false for real processing)")
        split_docs = get_mock_split_docs()
        print(f"   Split PDF into {len(split_docs)} sub-documents.")
        
        # S5: Record mock metrics
        metrics.total_processing_time_ms = (time.time() - start_time) * 1000
        metrics.documents_created = len(split_docs)
        metrics.pages_per_document = [doc.get("page_count", 0) for doc in split_docs]
        metrics.total_pages = sum(metrics.pages_per_document)
        log_splitter_metrics(metrics)
        
        return {
            "split_docs": split_docs, 
            "ocr_results": [],
            "pdf_processing_result": PdfProcessingResult.success_result(
                pdf_path or "mock", "Mock mode - no actual PDF processed"
            ).to_dict(),
            "splitter_metrics": metrics.to_dict(),
        }
    
    # Production mode: Use Docling
    if not DOCLING_AVAILABLE:
        print("   WARNING: Docling not available, using mock data")
        split_docs = get_mock_split_docs()
        
        metrics.total_processing_time_ms = (time.time() - start_time) * 1000
        metrics.documents_created = len(split_docs)
        log_splitter_metrics(metrics)
        
        return {
            "split_docs": split_docs, 
            "ocr_results": [],
            "pdf_processing_result": PdfProcessingResult(
                status=PdfStatus.PROCESSING_ERROR,
                success=False,
                message="Docling not available - using mock data",
                severity=PdfErrorSeverity.WARNING,
                error_type="DependencyMissing",
                suggested_action="Install Docling for production use",
                requires_manual_review=False,
            ).to_dict(),
            "splitter_metrics": metrics.to_dict(),
        }
    
    # User Story S4: Validate PDF before processing
    if not pdf_path:
        print("   WARNING: No PDF path provided, using mock data")
        metrics.total_processing_time_ms = (time.time() - start_time) * 1000
        log_splitter_metrics(metrics)
        
        if config.use_mock_on_failure:
            split_docs = get_mock_split_docs()
            return {
                "split_docs": split_docs, 
                "ocr_results": [],
                "pdf_processing_result": PdfProcessingResult(
                    status=PdfStatus.FILE_NOT_FOUND,
                    success=False,
                    message="No PDF path provided",
                    severity=PdfErrorSeverity.ERROR,
                    error_type="MissingInput",
                    suggested_action="Provide a PDF file path",
                    requires_manual_review=False,
                ).to_dict(),
                "splitter_metrics": metrics.to_dict(),
            }
        else:
            return {
                "split_docs": [], 
                "ocr_results": [],
                "pdf_processing_result": PdfProcessingResult.file_not_found(pdf_path or "").to_dict(),
                "splitter_metrics": metrics.to_dict(),
            }
    
    # Validate PDF file (S4: Check for corruption, password protection, etc.)
    # S5: Time validation
    validation_start = time.time()
    print(f"   Validating PDF: {pdf_path}")
    validation_result = validate_pdf_file(pdf_path)
    metrics.validation_time_ms = (time.time() - validation_start) * 1000
    
    if not validation_result.success:
        # PDF validation failed - flag for manual review if needed
        print(f"   {get_pdf_error_message(validation_result)}")
        
        if validation_result.requires_manual_review:
            print("   ⚡ PDF flagged for manual review")
        
        metrics.total_processing_time_ms = (time.time() - start_time) * 1000
        log_splitter_metrics(metrics)
        
        if config.use_mock_on_failure:
            print("   Falling back to mock data")
            split_docs = get_mock_split_docs()
            return {
                "split_docs": split_docs, 
                "ocr_results": [],
                "pdf_processing_result": validation_result.to_dict(),
                "splitter_metrics": metrics.to_dict(),
            }
        else:
            return {
                "split_docs": [], 
                "ocr_results": [],
                "pdf_processing_result": validation_result.to_dict(),
                "splitter_metrics": metrics.to_dict(),
            }
    
    try:
        # S5: Time extraction and splitting
        extraction_start = time.time()
        split_docs, ocr_results = split_pdf_with_docling(pdf_path, config)
        metrics.extraction_time_ms = (time.time() - extraction_start) * 1000
        
        # S5: Collect document metrics
        metrics.total_pages = len(ocr_results) if ocr_results else 0
        metrics.documents_created = len(split_docs)
        metrics.pages_per_document = [doc.get("page_count", 0) for doc in split_docs]
        
        # Report OCR summary and collect OCR metrics
        scanned_count = sum(1 for r in ocr_results if r.is_scanned)
        metrics.scanned_pages = scanned_count
        
        if scanned_count > 0:
            below_target = sum(1 for r in ocr_results if r.is_scanned and not r.meets_accuracy_target)
            metrics.pages_below_accuracy_target = below_target
            metrics.avg_ocr_confidence = sum(
                r.confidence for r in ocr_results if r.is_scanned
            ) / scanned_count
            metrics.ocr_time_ms = sum(
                r.processing_time_ms or 0 for r in ocr_results if r.is_scanned
            )
            print(f"   OCR Summary: {scanned_count} scanned pages, {below_target} below 95% target")
        
        if not split_docs:
            # Single-page or undetectable boundaries - treat as one document
            print("   No boundaries detected, treating as single document")
            split_docs = [{
                "id": 1,
                "page_range": [1, 1],
                "raw_text": "",
                "type": None,
                "detected_title": None,
                "confidence": 0.5,
                "has_scanned_pages": scanned_count > 0,
                "ocr_confidence": sum(r.confidence for r in ocr_results) / len(ocr_results) if ocr_results else 1.0,
                "meets_accuracy_target": all(r.meets_accuracy_target for r in ocr_results if r.is_scanned),
            }]
            metrics.documents_created = 1
            metrics.pages_per_document = [metrics.total_pages or 1]
        
        print(f"   Split PDF into {len(split_docs)} sub-documents.")
        
        # Convert OcrResult objects to dicts for state serialization
        ocr_results_dicts = [
            {
                "page_number": r.page_number,
                "is_scanned": r.is_scanned,
                "confidence": r.confidence,
                "meets_accuracy_target": r.meets_accuracy_target,
                "ocr_engine": r.ocr_engine_used,
                "warnings": r.warnings,
            }
            for r in ocr_results
        ]
        
        # Success - include processing result
        success_result = PdfProcessingResult.success_result(
            pdf_path, 
            f"Successfully processed PDF into {len(split_docs)} document(s)"
        )
        
        # S5: Finalize and log metrics
        metrics.total_processing_time_ms = (time.time() - start_time) * 1000
        log_splitter_metrics(metrics)
        
        return {
            "split_docs": split_docs, 
            "ocr_results": ocr_results_dicts,
            "pdf_processing_result": success_result.to_dict(),
            "splitter_metrics": metrics.to_dict(),
        }
        
    except Exception as e:
        # User Story S4: Handle processing errors with clear status messages
        print(f"   ERROR processing PDF: {e}")
        
        # S5: Record metrics even on failure
        metrics.total_processing_time_ms = (time.time() - start_time) * 1000
        log_splitter_metrics(metrics)
        
        # Analyze the error to determine type
        error_str = str(e).lower()
        
        if 'password' in error_str or 'encrypted' in error_str:
            error_result = PdfProcessingResult.password_protected(pdf_path)
        elif any(term in error_str for term in ['corrupt', 'invalid', 'malformed', 'damaged']):
            error_result = PdfProcessingResult.corrupted(pdf_path, str(e))
        else:
            error_result = PdfProcessingResult.processing_error(pdf_path, e)
        
        print(f"   {get_pdf_error_message(error_result)}")
        
        if error_result.requires_manual_review:
            print("   ⚡ PDF flagged for manual review")
        
        if config.use_mock_on_failure:
            print("   Falling back to mock data")
            split_docs = get_mock_split_docs()
            return {
                "split_docs": split_docs, 
                "ocr_results": [],
                "pdf_processing_result": error_result.to_dict(),
                "splitter_metrics": metrics.to_dict(),
            }
        else:
            return {
                "split_docs": [], 
                "ocr_results": [],
                "pdf_processing_result": error_result.to_dict(),
                "splitter_metrics": metrics.to_dict(),
            }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test the splitter standalone.
    
    Usage:
        python nodes/splitter.py [path/to/pdf]
    """
    import sys
    
    test_state: DealState = {
        "deal_id": "test-001",
        "status": "",
        "email_metadata": {},
        "raw_pdf_path": sys.argv[1] if len(sys.argv) > 1 else "",
        "split_docs": [],
        "property_address": None,
        "property_details": None,
        "buyers": [],
        "sellers": [],
        "participants": [],
        "financials": {},
        "financial_details": None,
        "contract_dates": None,
        "signature_fields": [],
        "signature_mapping": {},
        "missing_docs": [],
        "human_approval_status": "Pending",
        "target_system": "dotloop",
        "dotloop_payload": None,
    }
    
    # If no PDF provided, use mock mode
    if not test_state["raw_pdf_path"]:
        os.environ["USE_MOCK_SPLITTER"] = "true"
    else:
        os.environ["USE_MOCK_SPLITTER"] = "false"
    
    result = intelligent_splitter_node(test_state)
    
    print("\n--- Result ---")
    print(f"  Split into {len(result['split_docs'])} documents:")
    for doc in result["split_docs"]:
        print(f"    Doc {doc['id']}: Pages {doc['page_range']}")
        if doc.get("detected_title"):
            print(f"      Title: {doc['detected_title']}")
        print(f"      Text preview: {doc['raw_text'][:100]}...")
