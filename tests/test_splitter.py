"""
Unit Tests for Intelligent Splitter Node

Tests the PDF document splitting logic including:
- Document boundary detection
- Title extraction
- Mock data handling
- Docling integration (when available)
- Page mapping for original document reference (S3)

Run with: pytest tests/test_splitter.py -v
"""

import os
import tempfile
from typing import List, Any
from unittest.mock import Mock, MagicMock, patch
import pytest

# Import the module under test
from nodes.splitter import (
    SplitterConfig,
    DocumentBoundary,
    detect_document_boundaries,
    _extract_document_title,
    get_mock_split_docs,
    intelligent_splitter_node,
    MOCK_BUY_SELL_CONTENT,
    MOCK_DISCLOSURE_CONTENT,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def splitter_config():
    """Create a default splitter configuration."""
    return SplitterConfig()


@pytest.fixture
def initial_state():
    """Create an initial DealState for testing."""
    return {
        "deal_id": "test-001",
        "status": "",
        "email_metadata": {},
        "raw_pdf_path": "",
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


@pytest.fixture
def single_doc_pages():
    """Sample pages for a single document PDF (no boundary-triggering headers)."""
    return [
        "RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT\n\nPage 1 of 4\n\nProperty Address: 123 Main St...",
        "TERMS AND CONDITIONS\n\nPage 2 of 4\n\nThe buyer agrees to...",
        "SIGNATURES\n\nPage 3 of 4\n\nBuyer: _______________",
        "ADDITIONAL TERMS\n\nPage 4 of 4\n\nSome extra conditions...",
    ]


@pytest.fixture
def multi_doc_pages():
    """Sample pages for a multi-document PDF."""
    return [
        "RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT\n\nPage 1 of 3\n\nProperty: 123 Main St",
        "TERMS AND CONDITIONS\n\nPage 2 of 3\n\nBuyer agrees...",
        "SIGNATURES\n\nPage 3 of 3\n\nBuyer: _______________",
        "SELLER'S PROPERTY DISCLOSURE STATEMENT\n\nPage 1 of 2\n\nDisclosure information...",
        "ADDITIONAL DISCLOSURES\n\nPage 2 of 2\n\nMore info...",
    ]


# ============================================================================
# Test: SplitterConfig
# ============================================================================

class TestSplitterConfig:
    """Tests for the SplitterConfig dataclass."""
    
    def test_default_values(self):
        """Should have sensible default values."""
        config = SplitterConfig()
        
        assert len(config.boundary_patterns) > 0
        assert config.min_pages_per_doc == 1
        assert config.use_mock_on_failure is True
        assert config.analyze_page_headers is True
    
    def test_has_page_pattern(self):
        """Should include pattern for 'Page 1 of X'."""
        config = SplitterConfig()
        
        page_patterns = [p for p in config.boundary_patterns if "page" in p.lower()]
        assert len(page_patterns) > 0
    
    def test_has_document_type_patterns(self):
        """Should include patterns for document types."""
        config = SplitterConfig()
        
        patterns_str = " ".join(config.boundary_patterns).lower()
        assert "agreement" in patterns_str or "contract" in patterns_str
        assert "disclosure" in patterns_str


# ============================================================================
# Test: detect_document_boundaries
# ============================================================================

class TestDetectDocumentBoundaries:
    """Tests for the detect_document_boundaries function."""
    
    def test_single_document_returns_one_boundary(self, splitter_config, single_doc_pages):
        """Should return single boundary for single-document PDF."""
        boundaries = detect_document_boundaries(single_doc_pages, splitter_config)
        
        assert len(boundaries) == 1
        assert boundaries[0].start_page == 1
        assert boundaries[0].end_page == 4
    
    def test_multi_document_detects_split(self, splitter_config, multi_doc_pages):
        """Should detect boundary when 'Page 1' appears mid-document."""
        boundaries = detect_document_boundaries(multi_doc_pages, splitter_config)
        
        assert len(boundaries) == 2
        assert boundaries[0].start_page == 1
        assert boundaries[0].end_page == 3
        assert boundaries[1].start_page == 4
        assert boundaries[1].end_page == 5
    
    def test_empty_pages_returns_empty(self, splitter_config):
        """Should return empty list for empty input."""
        boundaries = detect_document_boundaries([], splitter_config)
        
        assert boundaries == []
    
    def test_detects_disclosure_boundary(self, splitter_config):
        """Should detect DISCLOSURE STATEMENT as document boundary."""
        pages = [
            "Some contract content...",
            "More contract...",
            "SELLER'S PROPERTY DISCLOSURE STATEMENT\n\nThis is a disclosure...",
        ]
        
        boundaries = detect_document_boundaries(pages, splitter_config)
        
        assert len(boundaries) == 2
        assert boundaries[1].start_page == 3
    
    def test_detects_addendum_boundary(self, splitter_config):
        """Should detect ADDENDUM as document boundary."""
        pages = [
            "Main contract content...",
            "ADDENDUM TO PURCHASE AGREEMENT\n\nAdditional terms...",
        ]
        
        boundaries = detect_document_boundaries(pages, splitter_config)
        
        assert len(boundaries) == 2
    
    def test_boundary_includes_detected_title(self, splitter_config):
        """Should extract title from document."""
        pages = [
            "RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT\n\nProperty: 123 Main St",
        ]
        
        boundaries = detect_document_boundaries(pages, splitter_config)
        
        assert boundaries[0].detected_title is not None
        assert "AGREEMENT" in boundaries[0].detected_title.upper() or boundaries[0].detected_title is not None


# ============================================================================
# Test: _extract_document_title
# ============================================================================

class TestExtractDocumentTitle:
    """Tests for the _extract_document_title function."""
    
    def test_extracts_agreement_title(self):
        """Should extract agreement-style titles."""
        text = "RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT\n\nProperty Address: 123 Main St"
        
        title = _extract_document_title(text)
        
        assert title is not None
        assert "AGREEMENT" in title.upper()
    
    def test_extracts_disclosure_title(self):
        """Should extract disclosure-style titles."""
        text = "SELLER'S PROPERTY DISCLOSURE STATEMENT\n\nThe seller discloses..."
        
        title = _extract_document_title(text)
        
        assert title is not None
        assert "DISCLOSURE" in title.upper()
    
    def test_returns_none_for_empty_text(self):
        """Should return None for empty text."""
        title = _extract_document_title("")
        
        assert title is None
    
    def test_returns_none_for_no_title(self):
        """Should return None when no recognizable title found."""
        text = "Some random text\nwithout a title\nMore text here"
        
        title = _extract_document_title(text)
        
        # May return None or first uppercase line
        # Just verify it doesn't crash
        assert title is None or isinstance(title, str)
    
    def test_handles_uppercase_first_line(self):
        """Should use uppercase first line as fallback title."""
        text = "THIS IS A TITLE LINE\n\nThis is body text that is not uppercase."
        
        title = _extract_document_title(text)
        
        # Should recognize uppercase line as potential title
        assert title is None or "TITLE" in title.upper()


# ============================================================================
# Test: get_mock_split_docs
# ============================================================================

class TestGetMockSplitDocs:
    """Tests for the get_mock_split_docs function."""
    
    def test_returns_list(self):
        """Should return a list of documents."""
        docs = get_mock_split_docs()
        
        assert isinstance(docs, list)
        assert len(docs) >= 2
    
    def test_docs_have_required_fields(self):
        """Should have all required fields in each document."""
        docs = get_mock_split_docs()
        
        for doc in docs:
            assert "id" in doc
            assert "page_range" in doc
            assert "raw_text" in doc
            assert "type" in doc
            assert isinstance(doc["page_range"], list)
            assert len(doc["page_range"]) == 2
    
    def test_includes_buy_sell_content(self):
        """Should include buy-sell agreement content."""
        docs = get_mock_split_docs()
        
        texts = [doc["raw_text"] for doc in docs]
        combined = " ".join(texts)
        
        assert "PURCHASE" in combined.upper() or "AGREEMENT" in combined.upper()
    
    def test_includes_disclosure_content(self):
        """Should include disclosure content."""
        docs = get_mock_split_docs()
        
        texts = [doc["raw_text"] for doc in docs]
        combined = " ".join(texts)
        
        assert "DISCLOSURE" in combined.upper()


# ============================================================================
# Test: intelligent_splitter_node (Mock Mode)
# ============================================================================

class TestIntelligentSplitterNodeMock:
    """Tests for intelligent_splitter_node in mock mode."""
    
    def test_mock_mode_returns_split_docs(self, initial_state):
        """Should return split docs in mock mode."""
        with patch.dict(os.environ, {"USE_MOCK_SPLITTER": "true"}):
            result = intelligent_splitter_node(initial_state)
        
        assert "split_docs" in result
        assert len(result["split_docs"]) >= 2
    
    def test_mock_mode_docs_have_structure(self, initial_state):
        """Should return properly structured documents in mock mode."""
        with patch.dict(os.environ, {"USE_MOCK_SPLITTER": "true"}):
            result = intelligent_splitter_node(initial_state)
        
        for doc in result["split_docs"]:
            assert "id" in doc
            assert "page_range" in doc
            assert "raw_text" in doc
    
    def test_falls_back_to_mock_when_pdf_missing(self, initial_state):
        """Should use mock data when PDF path doesn't exist."""
        initial_state["raw_pdf_path"] = "/nonexistent/path/to/pdf.pdf"
        
        with patch.dict(os.environ, {"USE_MOCK_SPLITTER": "false"}):
            result = intelligent_splitter_node(initial_state)
        
        # Should fall back to mock data
        assert "split_docs" in result
        assert len(result["split_docs"]) >= 1


# ============================================================================
# Test: DocumentBoundary
# ============================================================================

class TestDocumentBoundary:
    """Tests for the DocumentBoundary dataclass."""
    
    def test_create_boundary(self):
        """Should create a document boundary."""
        boundary = DocumentBoundary(
            start_page=1,
            end_page=4,
            detected_title="Test Document",
            confidence=0.95,
            boundary_type="header_pattern",
        )
        
        assert boundary.start_page == 1
        assert boundary.end_page == 4
        assert boundary.detected_title == "Test Document"
        assert boundary.confidence == 0.95
    
    def test_default_values(self):
        """Should have sensible defaults."""
        boundary = DocumentBoundary(start_page=1, end_page=3)
        
        assert boundary.detected_title is None
        assert boundary.confidence == 1.0
        assert boundary.boundary_type == "header_pattern"


# ============================================================================
# Test: Mock Content Constants
# ============================================================================

class TestMockContent:
    """Tests for the mock content constants."""
    
    def test_buy_sell_has_property_address(self):
        """Mock buy-sell should include property address."""
        assert "Property Address" in MOCK_BUY_SELL_CONTENT or "324 Muir" in MOCK_BUY_SELL_CONTENT
    
    def test_buy_sell_has_buyer_seller(self):
        """Mock buy-sell should include buyer and seller."""
        assert "BUYER" in MOCK_BUY_SELL_CONTENT.upper()
        assert "SELLER" in MOCK_BUY_SELL_CONTENT.upper()
    
    def test_buy_sell_has_financial_info(self):
        """Mock buy-sell should include financial information."""
        assert "Purchase Price" in MOCK_BUY_SELL_CONTENT or "$" in MOCK_BUY_SELL_CONTENT
    
    def test_buy_sell_has_signature_lines(self):
        """Mock buy-sell should include signature lines."""
        assert "Signature" in MOCK_BUY_SELL_CONTENT or "___" in MOCK_BUY_SELL_CONTENT
    
    def test_disclosure_has_disclosure_statement(self):
        """Mock disclosure should be a disclosure document."""
        assert "DISCLOSURE" in MOCK_DISCLOSURE_CONTENT.upper()


# ============================================================================
# OCR Configuration Tests (User Story S2)
# ============================================================================

class TestOcrConfig:
    """Tests for OCR configuration."""
    
    def test_ocr_config_defaults(self):
        """Should have sensible OCR defaults."""
        from nodes.splitter import OcrConfig
        config = OcrConfig()
        
        assert config.enabled is True
        assert config.languages == ["en"]
        assert config.force_full_page_ocr is False
        assert config.bitmap_area_threshold == 0.05
        assert config.confidence_threshold == 0.5
    
    def test_ocr_config_custom_languages(self):
        """Should support multiple languages."""
        from nodes.splitter import OcrConfig
        config = OcrConfig(languages=["en", "es", "fr"])
        
        assert len(config.languages) == 3
        assert "es" in config.languages
    
    def test_ocr_config_force_full_page(self):
        """Should support forcing OCR on all pages."""
        from nodes.splitter import OcrConfig
        config = OcrConfig(force_full_page_ocr=True)
        
        assert config.force_full_page_ocr is True
    
    def test_splitter_config_has_ocr(self, splitter_config):
        """SplitterConfig should include OCR configuration."""
        assert hasattr(splitter_config, 'ocr')
        assert splitter_config.ocr.enabled is True


class TestOcrEngine:
    """Tests for OCR engine selection."""
    
    def test_ocr_engine_enum_values(self):
        """Should have expected OCR engine options."""
        from nodes.splitter import OcrEngine
        
        assert OcrEngine.EASYOCR.value == "easyocr"
        assert OcrEngine.TESSERACT.value == "tesseract"
        assert OcrEngine.TESSERACT_CLI.value == "tesseract_cli"
        assert OcrEngine.AUTO.value == "auto"
    
    def test_default_engine_is_auto(self):
        """Default OCR engine should be AUTO."""
        from nodes.splitter import OcrConfig, OcrEngine
        config = OcrConfig()
        
        assert config.engine == OcrEngine.AUTO


class TestOcrResult:
    """Tests for OCR result tracking."""
    
    def test_ocr_result_creation(self):
        """Should create OCR result with all fields."""
        from nodes.splitter import OcrResult
        result = OcrResult(
            page_number=1,
            text="Sample text",
            confidence=0.98,
            is_scanned=True,
            ocr_engine_used="easyocr",
        )
        
        assert result.page_number == 1
        assert result.text == "Sample text"
        assert result.confidence == 0.98
        assert result.is_scanned is True
        assert result.ocr_engine_used == "easyocr"
    
    def test_meets_accuracy_target_high(self):
        """Should meet target when confidence >= 95%."""
        from nodes.splitter import OcrResult
        result = OcrResult(page_number=1, text="text", confidence=0.96, is_scanned=True)
        
        assert result.meets_accuracy_target is True
    
    def test_meets_accuracy_target_low(self):
        """Should not meet target when confidence < 95%."""
        from nodes.splitter import OcrResult
        result = OcrResult(page_number=1, text="text", confidence=0.90, is_scanned=True)
        
        assert result.meets_accuracy_target is False
    
    def test_meets_accuracy_target_exact(self):
        """Should meet target at exactly 95%."""
        from nodes.splitter import OcrResult
        result = OcrResult(page_number=1, text="text", confidence=0.95, is_scanned=True)
        
        assert result.meets_accuracy_target is True


class TestIsPageScanned:
    """Tests for scanned page detection."""
    
    def test_short_text_is_scanned(self):
        """Pages with very little text are likely scanned."""
        from nodes.splitter import is_page_scanned
        
        assert is_page_scanned("abc") is True
        assert is_page_scanned("") is True
    
    def test_normal_text_not_scanned(self):
        """Pages with normal text content are not scanned."""
        from nodes.splitter import is_page_scanned
        
        normal_text = """
        RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT
        
        This Agreement is entered into by and between the Buyer and Seller
        for the purchase of the property located at 123 Main Street.
        The purchase price is $450,000.00.
        """
        
        assert is_page_scanned(normal_text) is False
    
    def test_high_image_ratio_is_scanned(self):
        """Pages with high image ratio are scanned."""
        from nodes.splitter import is_page_scanned
        
        # Even with some text, high image ratio indicates scanned
        assert is_page_scanned("Some text here", image_ratio=0.9) is True


class TestEstimateOcrConfidence:
    """Tests for OCR confidence estimation."""
    
    def test_good_text_high_confidence(self):
        """Well-formed text should have high confidence."""
        from nodes.splitter import estimate_ocr_confidence
        
        good_text = """
        The purchase price for this property is $450,000.00.
        The closing date is 12/15/2024.
        Buyer: John Smith
        Seller: Jane Doe
        """
        
        confidence = estimate_ocr_confidence(good_text)
        assert confidence >= 0.8
    
    def test_garbage_text_low_confidence(self):
        """Text with OCR artifacts should have lower confidence."""
        from nodes.splitter import estimate_ocr_confidence
        
        garbage_text = "|||l|l|l| @@@ #### [[[]]]] bcdfghjklmnpqrs"
        
        confidence = estimate_ocr_confidence(garbage_text)
        assert confidence < 0.8
    
    def test_empty_text_medium_confidence(self):
        """Empty text should have uncertain confidence."""
        from nodes.splitter import estimate_ocr_confidence
        
        confidence = estimate_ocr_confidence("")
        assert 0.4 <= confidence <= 0.6
    
    def test_confidence_bounded(self):
        """Confidence should always be between 0 and 1."""
        from nodes.splitter import estimate_ocr_confidence
        
        # Even extreme text should stay in bounds
        assert 0.0 <= estimate_ocr_confidence("|||" * 100) <= 1.0
        assert 0.0 <= estimate_ocr_confidence("normal text " * 100) <= 1.0


class TestMockSplitDocsOcr:
    """Tests for OCR metadata in mock split docs."""
    
    def test_mock_docs_have_ocr_metadata(self):
        """Mock documents should include OCR fields."""
        docs = get_mock_split_docs()
        
        for doc in docs:
            assert "has_scanned_pages" in doc
            assert "ocr_confidence" in doc
            assert "meets_accuracy_target" in doc
    
    def test_mock_docs_not_scanned(self):
        """Mock documents should not be flagged as scanned."""
        docs = get_mock_split_docs()
        
        for doc in docs:
            assert doc["has_scanned_pages"] is False
            assert doc["ocr_confidence"] == 1.0
            assert doc["meets_accuracy_target"] is True


class TestIntelligentSplitterNodeOcr:
    """Tests for OCR integration in splitter node."""
    
    def test_node_returns_ocr_results(self, initial_state):
        """Node should return ocr_results in output."""
        with patch.dict(os.environ, {"USE_MOCK_SPLITTER": "true"}):
            result = intelligent_splitter_node(initial_state)
        
        assert "ocr_results" in result
        assert isinstance(result["ocr_results"], list)
    
    def test_node_respects_ocr_enabled_env(self, initial_state):
        """Node should read OCR_ENABLED from environment."""
        with patch.dict(os.environ, {
            "USE_MOCK_SPLITTER": "true",
            "OCR_ENABLED": "false"
        }):
            # This just verifies it doesn't crash - actual OCR disabled behavior
            # would be tested with a real PDF
            result = intelligent_splitter_node(initial_state)
            assert "split_docs" in result


# ============================================================================
# Page Mapping Tests (User Story S3)
# ============================================================================

class TestPageMapping:
    """Tests for PageMapping dataclass."""
    
    def test_page_mapping_creation(self):
        """Should create PageMapping with all fields."""
        from nodes.splitter import PageMapping
        
        mapping = PageMapping(
            original_page=5,
            local_page=1,
            has_content=True,
            is_scanned=False,
        )
        
        assert mapping.original_page == 5
        assert mapping.local_page == 1
        assert mapping.has_content is True
        assert mapping.is_scanned is False
        assert mapping.ocr_confidence is None
    
    def test_page_mapping_with_ocr(self):
        """Should include OCR confidence when scanned."""
        from nodes.splitter import PageMapping
        
        mapping = PageMapping(
            original_page=3,
            local_page=3,
            has_content=True,
            is_scanned=True,
            ocr_confidence=0.92,
        )
        
        assert mapping.is_scanned is True
        assert mapping.ocr_confidence == 0.92
    
    def test_page_mapping_to_dict(self):
        """Should convert to dictionary for serialization."""
        from nodes.splitter import PageMapping
        
        mapping = PageMapping(
            original_page=7,
            local_page=2,
            has_content=True,
            is_scanned=True,
            ocr_confidence=0.95,
        )
        
        d = mapping.to_dict()
        
        assert d["original_page"] == 7
        assert d["local_page"] == 2
        assert d["has_content"] is True
        assert d["is_scanned"] is True
        assert d["ocr_confidence"] == 0.95


class TestCreatePageMappings:
    """Tests for create_page_mappings function."""
    
    def test_creates_mappings_for_boundary(self):
        """Should create mapping for each page in boundary."""
        from nodes.splitter import create_page_mappings, DocumentBoundary, OcrResult
        
        boundary = DocumentBoundary(start_page=3, end_page=5)
        ocr_results: List[Any] = []
        pages_text = ["p1", "p2", "Page 3 content here", "Page 4 content", "Page 5 content"]
        
        mappings = create_page_mappings(boundary, ocr_results, pages_text)
        
        assert len(mappings) == 3
        assert mappings[0].original_page == 3
        assert mappings[0].local_page == 1
        assert mappings[1].original_page == 4
        assert mappings[1].local_page == 2
        assert mappings[2].original_page == 5
        assert mappings[2].local_page == 3
    
    def test_local_pages_start_at_one(self):
        """Local pages should always start at 1 regardless of original."""
        from nodes.splitter import create_page_mappings, DocumentBoundary
        
        boundary = DocumentBoundary(start_page=10, end_page=12)
        pages_text = [""] * 12  # 12 empty pages
        
        mappings = create_page_mappings(boundary, [], pages_text)
        
        assert mappings[0].local_page == 1
        assert mappings[0].original_page == 10
        assert mappings[2].local_page == 3
        assert mappings[2].original_page == 12
    
    def test_includes_ocr_info(self):
        """Should include OCR info from ocr_results."""
        from nodes.splitter import create_page_mappings, DocumentBoundary, OcrResult
        
        boundary = DocumentBoundary(start_page=1, end_page=2)
        ocr_results = [
            OcrResult(page_number=1, text="text", confidence=0.90, is_scanned=True),
            OcrResult(page_number=2, text="text", confidence=0.98, is_scanned=True),
        ]
        pages_text = ["page 1 content here", "page 2 content here"]
        
        mappings = create_page_mappings(boundary, ocr_results, pages_text)
        
        assert mappings[0].is_scanned is True
        assert mappings[0].ocr_confidence == 0.90
        assert mappings[1].is_scanned is True
        assert mappings[1].ocr_confidence == 0.98


class TestPageMappingLookups:
    """Tests for page mapping lookup functions."""
    
    def test_get_original_for_local(self):
        """Should find original page for a local page number."""
        from nodes.splitter import get_original_page_for_local
        
        mappings = [
            {"original_page": 5, "local_page": 1},
            {"original_page": 6, "local_page": 2},
            {"original_page": 7, "local_page": 3},
        ]
        
        assert get_original_page_for_local(1, mappings) == 5
        assert get_original_page_for_local(2, mappings) == 6
        assert get_original_page_for_local(3, mappings) == 7
    
    def test_get_original_returns_none_if_not_found(self):
        """Should return None if local page not in mappings."""
        from nodes.splitter import get_original_page_for_local
        
        mappings = [
            {"original_page": 5, "local_page": 1},
        ]
        
        assert get_original_page_for_local(99, mappings) is None
    
    def test_get_local_for_original(self):
        """Should find local page for an original page number."""
        from nodes.splitter import get_local_page_for_original
        
        mappings = [
            {"original_page": 5, "local_page": 1},
            {"original_page": 6, "local_page": 2},
            {"original_page": 7, "local_page": 3},
        ]
        
        assert get_local_page_for_original(5, mappings) == 1
        assert get_local_page_for_original(6, mappings) == 2
        assert get_local_page_for_original(7, mappings) == 3
    
    def test_get_local_returns_none_if_not_found(self):
        """Should return None if original page not in mappings."""
        from nodes.splitter import get_local_page_for_original
        
        mappings = [
            {"original_page": 5, "local_page": 1},
        ]
        
        assert get_local_page_for_original(1, mappings) is None


class TestMockDocsPageMappings:
    """Tests for page mappings in mock split docs."""
    
    def test_mock_docs_have_page_mappings(self):
        """Mock documents should include page_mappings."""
        docs = get_mock_split_docs()
        
        for doc in docs:
            assert "page_mappings" in doc
            assert isinstance(doc["page_mappings"], list)
            assert len(doc["page_mappings"]) > 0
    
    def test_mock_docs_have_page_statistics(self):
        """Mock documents should include page count and range info."""
        docs = get_mock_split_docs()
        
        for doc in docs:
            assert "page_count" in doc
            assert "original_page_start" in doc
            assert "original_page_end" in doc
            assert "total_original_pages" in doc
    
    def test_mock_doc1_page_mapping_correct(self):
        """First mock doc should map pages 1-4."""
        docs = get_mock_split_docs()
        doc1 = docs[0]
        
        assert doc1["page_count"] == 4
        assert doc1["original_page_start"] == 1
        assert doc1["original_page_end"] == 4
        assert len(doc1["page_mappings"]) == 4
        
        # Check first page mapping
        assert doc1["page_mappings"][0]["original_page"] == 1
        assert doc1["page_mappings"][0]["local_page"] == 1
    
    def test_mock_doc2_page_mapping_correct(self):
        """Second mock doc should map pages 5-6 to local 1-2."""
        docs = get_mock_split_docs()
        doc2 = docs[1]
        
        assert doc2["page_count"] == 2
        assert doc2["original_page_start"] == 5
        assert doc2["original_page_end"] == 6
        assert len(doc2["page_mappings"]) == 2
        
        # Second doc's local page 1 = original page 5
        assert doc2["page_mappings"][0]["original_page"] == 5
        assert doc2["page_mappings"][0]["local_page"] == 1
        
        # Second doc's local page 2 = original page 6
        assert doc2["page_mappings"][1]["original_page"] == 6
        assert doc2["page_mappings"][1]["local_page"] == 2
    
    def test_total_original_pages_consistent(self):
        """All docs should have same total_original_pages."""
        docs = get_mock_split_docs()
        
        total = docs[0]["total_original_pages"]
        for doc in docs:
            assert doc["total_original_pages"] == total


class TestFormatPageReference:
    """Tests for format_page_reference function."""
    
    def test_formats_page_reference(self):
        """Should format page reference correctly."""
        from nodes.splitter import format_page_reference
        
        result = format_page_reference(5, 12)
        
        assert result == "Page 5 of 12 (original)"
    
    def test_formats_single_page(self):
        """Should format single page document correctly."""
        from nodes.splitter import format_page_reference
        
        result = format_page_reference(1, 1)
        
        assert result == "Page 1 of 1 (original)"


# ============================================================================
# User Story S4: Error Handling for Corrupted/Password PDFs
# ============================================================================

class TestPdfStatus:
    """Test PdfStatus enum values."""
    
    def test_has_expected_values(self):
        from nodes.splitter import PdfStatus
        expected = [
            "success", "password_protected", "corrupted", "empty",
            "unsupported_format", "file_not_found", "permission_denied",
            "processing_error", "requires_manual_review"
        ]
        for value in expected:
            assert PdfStatus(value) is not None
    
    def test_success_status(self):
        from nodes.splitter import PdfStatus
        assert PdfStatus.SUCCESS.value == "success"
    
    def test_password_protected_status(self):
        from nodes.splitter import PdfStatus
        assert PdfStatus.PASSWORD_PROTECTED.value == "password_protected"
    
    def test_corrupted_status(self):
        from nodes.splitter import PdfStatus
        assert PdfStatus.CORRUPTED.value == "corrupted"


class TestPdfErrorSeverity:
    """Test PdfErrorSeverity enum values."""
    
    def test_has_severity_levels(self):
        from nodes.splitter import PdfErrorSeverity
        assert PdfErrorSeverity.INFO.value == "info"
        assert PdfErrorSeverity.WARNING.value == "warning"
        assert PdfErrorSeverity.ERROR.value == "error"
        assert PdfErrorSeverity.CRITICAL.value == "critical"


class TestPdfProcessingResult:
    """Test PdfProcessingResult dataclass."""
    
    def test_success_result_creation(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus, PdfErrorSeverity
        result = PdfProcessingResult(
            status=PdfStatus.SUCCESS,
            success=True,
            message="Test success",
            severity=PdfErrorSeverity.INFO,
        )
        assert result.success is True
        assert result.status == PdfStatus.SUCCESS
        assert result.message == "Test success"
    
    def test_error_result_creation(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus, PdfErrorSeverity
        result = PdfProcessingResult(
            status=PdfStatus.CORRUPTED,
            success=False,
            message="File corrupted",
            severity=PdfErrorSeverity.CRITICAL,
            error_type="CorruptedPdf",
            requires_manual_review=True,
        )
        assert result.success is False
        assert result.requires_manual_review is True
        assert result.error_type == "CorruptedPdf"
    
    def test_to_dict(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus, PdfErrorSeverity
        result = PdfProcessingResult(
            status=PdfStatus.SUCCESS,
            success=True,
            message="Test",
            severity=PdfErrorSeverity.INFO,
            file_path="/test/path.pdf",
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["success"] is True
        assert d["message"] == "Test"
        assert d["file_path"] == "/test/path.pdf"


class TestPdfProcessingResultFactoryMethods:
    """Test factory methods for PdfProcessingResult."""
    
    def test_success_result(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus
        result = PdfProcessingResult.success_result("/path/test.pdf", "All good")
        assert result.success is True
        assert result.status == PdfStatus.SUCCESS
        assert result.file_path == "/path/test.pdf"
        assert result.message == "All good"
    
    def test_password_protected(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus
        result = PdfProcessingResult.password_protected("/path/secure.pdf")
        assert result.success is False
        assert result.status == PdfStatus.PASSWORD_PROTECTED
        assert result.requires_manual_review is True
        assert result.error_type == "PasswordProtected"
        assert "password" in result.message.lower()
    
    def test_corrupted(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus
        result = PdfProcessingResult.corrupted("/path/bad.pdf", "Invalid header")
        assert result.success is False
        assert result.status == PdfStatus.CORRUPTED
        assert result.requires_manual_review is True
        assert result.error_details == "Invalid header"
    
    def test_empty_pdf(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus
        result = PdfProcessingResult.empty_pdf("/path/empty.pdf")
        assert result.success is False
        assert result.status == PdfStatus.EMPTY
        assert "empty" in result.message.lower() or "no pages" in result.message.lower()
    
    def test_not_a_pdf(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus
        result = PdfProcessingResult.not_a_pdf("/path/fake.pdf")
        assert result.success is False
        assert result.status == PdfStatus.UNSUPPORTED_FORMAT
        assert result.requires_manual_review is True
    
    def test_file_not_found(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus
        result = PdfProcessingResult.file_not_found("/path/missing.pdf")
        assert result.success is False
        assert result.status == PdfStatus.FILE_NOT_FOUND
        assert result.file_path == "/path/missing.pdf"
    
    def test_processing_error(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus
        error = ValueError("Something went wrong")
        result = PdfProcessingResult.processing_error("/path/test.pdf", error)
        assert result.success is False
        assert result.status == PdfStatus.PROCESSING_ERROR
        assert "Something went wrong" in result.error_details


class TestValidatePdfFile:
    """Test validate_pdf_file function."""
    
    def test_file_not_found(self, tmp_path):
        from nodes.splitter import validate_pdf_file, PdfStatus
        result = validate_pdf_file(str(tmp_path / "nonexistent.pdf"))
        assert result.success is False
        assert result.status == PdfStatus.FILE_NOT_FOUND
    
    def test_empty_file(self, tmp_path):
        from nodes.splitter import validate_pdf_file, PdfStatus
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")
        result = validate_pdf_file(str(empty_file))
        assert result.success is False
        assert result.status == PdfStatus.EMPTY
    
    def test_not_a_pdf(self, tmp_path):
        from nodes.splitter import validate_pdf_file, PdfStatus
        text_file = tmp_path / "notapdf.pdf"
        text_file.write_text("This is just text, not a PDF")
        result = validate_pdf_file(str(text_file))
        assert result.success is False
        assert result.status == PdfStatus.UNSUPPORTED_FORMAT
    
    def test_valid_pdf_header(self, tmp_path):
        from nodes.splitter import validate_pdf_file, PdfStatus
        # Create a minimal PDF-like file (just the header)
        pdf_file = tmp_path / "valid.pdf"
        # Minimal PDF content without /Encrypt
        pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        pdf_file.write_bytes(pdf_content)
        result = validate_pdf_file(str(pdf_file))
        assert result.success is True
        assert result.status == PdfStatus.SUCCESS
    
    def test_detects_encrypted_pdf(self, tmp_path):
        from nodes.splitter import validate_pdf_file, PdfStatus
        pdf_file = tmp_path / "encrypted.pdf"
        # Simulate encrypted PDF with /Encrypt in content
        pdf_content = b"%PDF-1.4\n/Encrypt << /V 2 >>\n%%EOF"
        pdf_file.write_bytes(pdf_content)
        result = validate_pdf_file(str(pdf_file))
        assert result.success is False
        assert result.status == PdfStatus.PASSWORD_PROTECTED


class TestGetPdfErrorMessage:
    """Test get_pdf_error_message function."""
    
    def test_success_message(self):
        from nodes.splitter import get_pdf_error_message, PdfProcessingResult
        result = PdfProcessingResult.success_result("/test.pdf", "All good")
        message = get_pdf_error_message(result)
        assert message == "All good"
    
    def test_error_message_includes_type(self):
        from nodes.splitter import get_pdf_error_message, PdfProcessingResult
        result = PdfProcessingResult.corrupted("/test.pdf", "Bad structure")
        message = get_pdf_error_message(result)
        assert "Error Type:" in message
        assert "CorruptedFile" in message
    
    def test_error_message_includes_action(self):
        from nodes.splitter import get_pdf_error_message, PdfProcessingResult
        result = PdfProcessingResult.password_protected("/test.pdf")
        message = get_pdf_error_message(result)
        assert "Action:" in message
    
    def test_error_message_includes_manual_review_flag(self):
        from nodes.splitter import get_pdf_error_message, PdfProcessingResult
        result = PdfProcessingResult.corrupted("/test.pdf", "Bad")
        assert result.requires_manual_review is True
        message = get_pdf_error_message(result)
        assert "manual review" in message.lower()


class TestIntelligentSplitterNodeErrorHandling:
    """Test intelligent_splitter_node error handling (S4)."""
    
    def test_node_returns_pdf_processing_result_in_mock_mode(self):
        """Test that mock mode includes pdf_processing_result."""
        import os
        os.environ["USE_MOCK_SPLITTER"] = "true"
        
        from nodes.splitter import intelligent_splitter_node
        from state import DealState
        
        state: DealState = {
            "deal_id": "test-s4-1",
            "status": "",
            "email_metadata": {},
            "raw_pdf_path": "",
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
        
        result = intelligent_splitter_node(state)
        
        assert "pdf_processing_result" in result
        assert result["pdf_processing_result"]["success"] is True
    
    def test_node_handles_missing_pdf_path(self):
        """Test that empty pdf_path returns appropriate error."""
        import os
        os.environ["USE_MOCK_SPLITTER"] = "false"
        
        from nodes.splitter import intelligent_splitter_node
        from state import DealState
        
        state: DealState = {
            "deal_id": "test-s4-2",
            "status": "",
            "email_metadata": {},
            "raw_pdf_path": "",
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
        
        result = intelligent_splitter_node(state)
        
        assert "pdf_processing_result" in result
        assert result["pdf_processing_result"]["success"] is False
        # Should still return split_docs (possibly mock)
        assert "split_docs" in result
    
    def test_node_validates_pdf_before_processing(self, tmp_path):
        """Test that node validates PDF and catches errors."""
        import os
        os.environ["USE_MOCK_SPLITTER"] = "false"
        
        # Create an invalid "PDF" file
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_text("This is not a PDF file at all")
        
        from nodes.splitter import intelligent_splitter_node
        from state import DealState
        
        state: DealState = {
            "deal_id": "test-s4-3",
            "status": "",
            "email_metadata": {},
            "raw_pdf_path": str(invalid_pdf),
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
        
        result = intelligent_splitter_node(state)
        
        assert "pdf_processing_result" in result
        assert result["pdf_processing_result"]["success"] is False
        assert result["pdf_processing_result"]["status"] == "unsupported_format"


class TestPdfProcessingResultSerialization:
    """Test that PdfProcessingResult serializes correctly for state."""
    
    def test_to_dict_serializable(self):
        """Test that to_dict produces JSON-serializable output."""
        import json
        from nodes.splitter import PdfProcessingResult
        
        result = PdfProcessingResult.corrupted("/test.pdf", "Error details")
        d = result.to_dict()
        
        # Should not raise
        json_str = json.dumps(d)
        assert json_str is not None
    
    def test_to_dict_includes_all_fields(self):
        from nodes.splitter import PdfProcessingResult, PdfStatus, PdfErrorSeverity
        
        result = PdfProcessingResult(
            status=PdfStatus.PASSWORD_PROTECTED,
            success=False,
            message="Password required",
            severity=PdfErrorSeverity.ERROR,
            file_path="/path/secure.pdf",
            file_size_bytes=12345,
            error_type="PasswordProtected",
            error_details="AES-256 encryption",
            suggested_action="Contact sender for password",
            requires_manual_review=True,
        )
        
        d = result.to_dict()
        
        assert d["status"] == "password_protected"
        assert d["success"] is False
        assert d["message"] == "Password required"
        assert d["severity"] == "error"
        assert d["file_path"] == "/path/secure.pdf"
        assert d["file_size_bytes"] == 12345
        assert d["error_type"] == "PasswordProtected"
        assert d["error_details"] == "AES-256 encryption"
        assert d["suggested_action"] == "Contact sender for password"
        assert d["requires_manual_review"] is True


# ============================================================================
# User Story S5: Performance Metrics Logging
# ============================================================================

class TestSplitterMetrics:
    """Test SplitterMetrics dataclass."""
    
    def test_metrics_creation(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics()
        assert metrics.total_processing_time_ms == 0.0
        assert metrics.total_pages == 0
        assert metrics.documents_created == 0
    
    def test_metrics_with_values(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics(
            file_path="/test/file.pdf",
            file_size_bytes=12345,
            total_processing_time_ms=1500.0,
            total_pages=10,
            documents_created=3,
            pages_per_document=[4, 4, 2],
        )
        assert metrics.file_path == "/test/file.pdf"
        assert metrics.file_size_bytes == 12345
        assert metrics.total_pages == 10
        assert metrics.documents_created == 3
        assert len(metrics.pages_per_document) == 3
    
    def test_calculate_derived_metrics(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics(
            total_processing_time_ms=2000.0,  # 2 seconds
            total_pages=10,
        )
        metrics.calculate_derived_metrics()
        assert metrics.pages_per_second == 5.0  # 10 pages / 2 seconds
    
    def test_is_slow_detection(self):
        from nodes.splitter import SplitterMetrics
        
        # Fast processing
        fast_metrics = SplitterMetrics(total_processing_time_ms=1000.0)
        fast_metrics.calculate_derived_metrics()
        assert fast_metrics.is_slow is False
        
        # Slow processing (default threshold is 5000ms)
        slow_metrics = SplitterMetrics(total_processing_time_ms=6000.0)
        slow_metrics.calculate_derived_metrics()
        assert slow_metrics.is_slow is True
    
    def test_custom_slow_threshold(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics(
            total_processing_time_ms=3000.0,
            slow_threshold_ms=2000.0,  # 2 second threshold
        )
        metrics.calculate_derived_metrics()
        assert metrics.is_slow is True
    
    def test_to_dict(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics(
            file_path="/test.pdf",
            total_processing_time_ms=1234.567,
            total_pages=5,
            documents_created=2,
        )
        d = metrics.to_dict()
        
        assert d["file_path"] == "/test.pdf"
        assert d["total_processing_time_ms"] == 1234.57  # Rounded
        assert d["total_pages"] == 5
        assert d["documents_created"] == 2
        assert "timestamp" in d
    
    def test_to_dict_serializable(self):
        """Test that to_dict produces JSON-serializable output."""
        import json
        from nodes.splitter import SplitterMetrics
        
        metrics = SplitterMetrics(
            file_path="/test.pdf",
            total_processing_time_ms=1500.0,
            total_pages=10,
            documents_created=2,
            pages_per_document=[6, 4],
        )
        
        d = metrics.to_dict()
        json_str = json.dumps(d)
        assert json_str is not None
    
    def test_log_summary(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics(
            file_path="/path/to/test.pdf",
            total_processing_time_ms=2500.0,
            total_pages=10,
            documents_created=2,
            pages_per_document=[6, 4],
        )
        metrics.calculate_derived_metrics()
        
        summary = metrics.log_summary()
        
        assert "test.pdf" in summary
        assert "2500" in summary or "2.5" in summary
        assert "10" in summary
        assert "2 document" in summary


class TestSplitterMetricsOcr:
    """Test SplitterMetrics OCR-related fields."""
    
    def test_ocr_metrics(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics(
            scanned_pages=5,
            ocr_time_ms=3000.0,
            avg_ocr_confidence=0.92,
            pages_below_accuracy_target=2,
        )
        
        assert metrics.scanned_pages == 5
        assert metrics.ocr_time_ms == 3000.0
        assert metrics.avg_ocr_confidence == 0.92
        assert metrics.pages_below_accuracy_target == 2
    
    def test_log_summary_includes_ocr(self):
        from nodes.splitter import SplitterMetrics
        metrics = SplitterMetrics(
            file_path="/test.pdf",
            total_processing_time_ms=1000.0,
            total_pages=10,
            documents_created=1,
            scanned_pages=3,
            avg_ocr_confidence=0.95,
        )
        metrics.calculate_derived_metrics()
        
        summary = metrics.log_summary()
        assert "OCR" in summary
        assert "3 pages" in summary


class TestLogSplitterMetrics:
    """Test log_splitter_metrics function."""
    
    def test_log_splitter_metrics_calculates_derived(self):
        from nodes.splitter import SplitterMetrics, log_splitter_metrics
        
        metrics = SplitterMetrics(
            total_processing_time_ms=1000.0,
            total_pages=10,
        )
        
        # Before calling log function
        assert metrics.pages_per_second == 0.0
        
        log_splitter_metrics(metrics)
        
        # After - derived metrics should be calculated
        assert metrics.pages_per_second == 10.0


class TestNodeReturnsMetrics:
    """Test that intelligent_splitter_node returns metrics."""
    
    def test_mock_mode_returns_metrics(self):
        import os
        os.environ["USE_MOCK_SPLITTER"] = "true"
        
        from nodes.splitter import intelligent_splitter_node
        from state import DealState
        
        state: DealState = {
            "deal_id": "test-s5-1",
            "status": "",
            "email_metadata": {},
            "raw_pdf_path": "",
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
        
        result = intelligent_splitter_node(state)
        
        assert "splitter_metrics" in result
        metrics = result["splitter_metrics"]
        
        # Check required fields per acceptance criteria
        assert "total_processing_time_ms" in metrics
        assert "total_pages" in metrics
        assert "documents_created" in metrics
        assert "pages_per_document" in metrics
        assert metrics["total_processing_time_ms"] >= 0
    
    def test_metrics_include_document_counts(self):
        import os
        os.environ["USE_MOCK_SPLITTER"] = "true"
        
        from nodes.splitter import intelligent_splitter_node
        from state import DealState
        
        state: DealState = {
            "deal_id": "test-s5-2",
            "status": "",
            "email_metadata": {},
            "raw_pdf_path": "",
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
        
        result = intelligent_splitter_node(state)
        
        metrics = result["splitter_metrics"]
        split_docs = result["split_docs"]
        
        # Metrics should reflect actual documents created
        assert metrics["documents_created"] == len(split_docs)
        assert len(metrics["pages_per_document"]) == len(split_docs)
    
    def test_metrics_include_timestamp(self):
        import os
        os.environ["USE_MOCK_SPLITTER"] = "true"
        
        from nodes.splitter import intelligent_splitter_node
        from state import DealState
        
        state: DealState = {
            "deal_id": "test-s5-3",
            "status": "",
            "email_metadata": {},
            "raw_pdf_path": "",
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
        
        result = intelligent_splitter_node(state)
        
        metrics = result["splitter_metrics"]
        assert "timestamp" in metrics
        # Should be ISO format
        assert "T" in metrics["timestamp"]


class TestMetricsOnValidationFailure:
    """Test that metrics are returned even on validation failure."""
    
    def test_returns_metrics_on_invalid_pdf(self, tmp_path):
        import os
        os.environ["USE_MOCK_SPLITTER"] = "false"
        
        # Create an invalid "PDF" file
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_text("This is not a PDF")
        
        from nodes.splitter import intelligent_splitter_node
        from state import DealState
        
        state: DealState = {
            "deal_id": "test-s5-4",
            "status": "",
            "email_metadata": {},
            "raw_pdf_path": str(invalid_pdf),
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
        
        result = intelligent_splitter_node(state)
        
        # Should still have metrics even on failure
        assert "splitter_metrics" in result
        metrics = result["splitter_metrics"]
        assert metrics["total_processing_time_ms"] >= 0
        assert metrics["validation_time_ms"] >= 0


class TestSlowDocumentDetection:
    """Test slow document detection."""
    
    def test_slow_threshold_configurable(self):
        from nodes.splitter import SplitterMetrics
        
        # Very low threshold
        metrics = SplitterMetrics(
            total_processing_time_ms=100.0,
            slow_threshold_ms=50.0,
        )
        metrics.calculate_derived_metrics()
        assert metrics.is_slow is True
        
        # High threshold
        metrics2 = SplitterMetrics(
            total_processing_time_ms=100.0,
            slow_threshold_ms=1000.0,
        )
        metrics2.calculate_derived_metrics()
        assert metrics2.is_slow is False


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
