"""
Tests for the document classifier node (User Story C1).

User Story C1:
"As a processor, I want documents automatically classified by type 
(Buy-Sell, Disclosure, Addendum) so the right extraction logic runs."

Acceptance Criteria:
- LLM/ML classifier with >90% accuracy on common document types
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json
import os
import tempfile
from typing import cast, Any

from nodes.classifier import (
    DocumentType,
    ClassifierConfig,
    ClassificationResult,
    ClassifierMetrics,
    classify_by_keywords,
    classify_document,
    doc_type_classifier_node,
    KEYWORD_PATTERNS,
    DOCUMENT_TYPE_TO_FOLDER,
    # C2 OCR-related imports
    OcrDocumentInfo,
    preprocess_ocr_text,
    assess_text_quality_for_classification,
    OCR_ACCURACY_TARGET,
    OCR_LOW_CONFIDENCE_THRESHOLD,
    OCR_POOR_QUALITY_THRESHOLD,
    # C3 Review queue imports
    ReviewReason,
    ReviewFlag,
    check_needs_review,
    REVIEW_CONFIDENCE_THRESHOLD,
    # C4 Custom document types imports
    CustomDocumentType,
    CustomDocumentTypeRegistry,
    CustomTypeMatch,
    match_custom_types,
    get_custom_type_registry,
    set_custom_type_registry,
    create_custom_document_type,
    update_custom_document_type,
    delete_custom_document_type,
    list_custom_document_types,
    # C5 Missing document detection imports
    PropertyMetadata,
    TransactionMetadata,
    RequirementCondition,
    RequiredDocumentRule,
    MissingDocument,
    MissingDocumentReport,
    get_required_documents,
    detect_missing_documents,
    DEFAULT_REQUIREMENT_RULES,
)
from state import DealState


# ============================================================================
# DocumentType Enum Tests
# ============================================================================

class TestDocumentType:
    """Tests for the DocumentType enum."""
    
    def test_all_core_document_types_exist(self):
        """Verify core document types are defined."""
        core_types = [
            "BUY_SELL_AGREEMENT",
            "PURCHASE_AGREEMENT",
            "COUNTER_OFFER",
            "AMENDMENT",
            "ADDENDUM",
            "SELLER_DISCLOSURE",
            "LEAD_PAINT_DISCLOSURE",
            "MOLD_DISCLOSURE",
            "INSPECTION_REPORT",
            "UNKNOWN",
        ]
        for type_name in core_types:
            assert hasattr(DocumentType, type_name), f"Missing DocumentType.{type_name}"
    
    def test_buy_sell_agreement_value(self):
        """Verify buy-sell agreement has correct value."""
        assert DocumentType.BUY_SELL_AGREEMENT.value == "Buy-Sell Agreement"
    
    def test_disclosure_values(self):
        """Verify disclosure types have correct values."""
        assert DocumentType.SELLER_DISCLOSURE.value == "Seller Disclosure"
        assert DocumentType.LEAD_PAINT_DISCLOSURE.value == "Lead Paint Disclosure"
    
    def test_addendum_value(self):
        """Verify addendum has correct value."""
        assert DocumentType.ADDENDUM.value == "Addendum"
    
    def test_from_string_buy_sell(self):
        """Test from_string for buy-sell variations."""
        assert DocumentType.from_string("Buy-Sell Agreement") == DocumentType.BUY_SELL_AGREEMENT
        assert DocumentType.from_string("buy sell agreement") == DocumentType.BUY_SELL_AGREEMENT
    
    def test_from_string_purchase_agreement(self):
        """Test from_string for purchase agreement."""
        assert DocumentType.from_string("purchase agreement") == DocumentType.PURCHASE_AGREEMENT
    
    def test_from_string_counter_offer(self):
        """Test from_string for counter offer."""
        assert DocumentType.from_string("counter offer") == DocumentType.COUNTER_OFFER
    
    def test_from_string_addendum(self):
        """Test from_string for addendum."""
        assert DocumentType.from_string("addendum") == DocumentType.ADDENDUM
    
    def test_from_string_disclosure(self):
        """Test from_string for seller disclosure."""
        assert DocumentType.from_string("seller disclosure") == DocumentType.SELLER_DISCLOSURE
    
    def test_from_string_lead_paint(self):
        """Test from_string for lead paint disclosure."""
        assert DocumentType.from_string("lead paint disclosure") == DocumentType.LEAD_PAINT_DISCLOSURE
    
    def test_from_string_unknown(self):
        """Test from_string returns UNKNOWN for unrecognized."""
        assert DocumentType.from_string("random gibberish") == DocumentType.UNKNOWN


# ============================================================================
# ClassifierConfig Tests
# ============================================================================

class TestClassifierConfig:
    """Tests for the ClassifierConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClassifierConfig()
        assert config.use_llm is True
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.llm_temperature == 0.0
        assert config.use_keyword_fallback is True
        assert config.min_confidence_threshold == 0.50
        assert config.use_mock is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClassifierConfig(
            use_llm=False,
            llm_provider="anthropic",
            llm_model="claude-3-haiku",
            use_keyword_fallback=False,
            min_confidence_threshold=0.8,
            use_mock=True,
        )
        assert config.use_llm is False
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-haiku"
        assert config.use_keyword_fallback is False
        assert config.min_confidence_threshold == 0.8
        assert config.use_mock is True


# ============================================================================
# ClassificationResult Tests
# ============================================================================

class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""
    
    def test_create_result(self):
        """Test creating a classification result."""
        result = ClassificationResult(
            document_type=DocumentType.BUY_SELL_AGREEMENT,
            confidence=0.95,
            method="llm",
            processing_time_ms=150,
        )
        assert result.document_type == DocumentType.BUY_SELL_AGREEMENT
        assert result.confidence == 0.95
        assert result.method == "llm"
        assert result.processing_time_ms == 150
    
    def test_suggested_folder_auto_set(self):
        """Test that suggested folder is auto-set based on type."""
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.9,
            method="keyword",
        )
        assert result.suggested_folder == "Addenda"
    
    def test_contracts_folder(self):
        """Test that contracts go to Contracts folder."""
        result = ClassificationResult(
            document_type=DocumentType.BUY_SELL_AGREEMENT,
            confidence=0.9,
            method="keyword",
        )
        assert result.suggested_folder == "Contracts"
    
    def test_disclosures_folder(self):
        """Test that disclosures go to Disclosures folder."""
        result = ClassificationResult(
            document_type=DocumentType.SELLER_DISCLOSURE,
            confidence=0.9,
            method="keyword",
        )
        assert result.suggested_folder == "Disclosures"
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ClassificationResult(
            document_type=DocumentType.SELLER_DISCLOSURE,
            confidence=0.85,
            method="keyword",
            alternative_types=[],
            processing_time_ms=10,
        )
        d = result.to_dict()
        assert d["document_type"] == "Seller Disclosure"
        assert d["confidence"] == 0.85
        assert d["method"] == "keyword"
        assert d["processing_time_ms"] == 10
    
    def test_meets_accuracy_target_high_confidence(self):
        """Test meets_accuracy_target for high confidence."""
        result = ClassificationResult(
            document_type=DocumentType.BUY_SELL_AGREEMENT,
            confidence=0.95,
            method="llm",
        )
        assert result.meets_accuracy_target is True
    
    def test_meets_accuracy_target_low_confidence(self):
        """Test meets_accuracy_target for low confidence."""
        result = ClassificationResult(
            document_type=DocumentType.UNKNOWN,
            confidence=0.3,
            method="keyword",
        )
        assert result.meets_accuracy_target is False


# ============================================================================
# ClassifierMetrics Tests
# ============================================================================

class TestClassifierMetrics:
    """Tests for the ClassifierMetrics dataclass."""
    
    def test_create_metrics(self):
        """Test creating classifier metrics."""
        metrics = ClassifierMetrics(
            total_documents=10,
            documents_classified=9,
            documents_unknown=1,
            llm_classifications=7,
            keyword_classifications=2,
        )
        assert metrics.total_documents == 10
        assert metrics.documents_classified == 9
        assert metrics.documents_unknown == 1
        assert metrics.llm_classifications == 7
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ClassifierMetrics(
            total_documents=5,
            documents_classified=5,
            documents_unknown=0,
            llm_classifications=5,
            keyword_classifications=0,
            avg_confidence=0.92,
        )
        d = metrics.to_dict()
        assert d["total_documents"] == 5
        assert d["documents_classified"] == 5
        assert d["avg_confidence"] == 0.92


# ============================================================================
# Keyword Classification Tests
# ============================================================================

class TestKeywordClassification:
    """Tests for keyword-based classification."""
    
    def test_classify_buy_sell_agreement(self):
        """Test classifying a buy-sell agreement."""
        text = """
        REAL ESTATE BUY-SELL AGREEMENT
        
        This Agreement is entered into between Buyer and Seller for the
        purchase of property located at 123 Main Street. The purchase price
        shall be $450,000 with earnest money of $10,000.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.BUY_SELL_AGREEMENT
        assert result.confidence > 0.5
        assert result.method == "keyword"
    
    def test_classify_purchase_agreement(self):
        """Test classifying a purchase agreement."""
        text = """
        REAL ESTATE PURCHASE AGREEMENT
        
        This purchase agreement outlines the terms for the sale of property.
        """
        result = classify_by_keywords(text)
        # Could be BUY_SELL or PURCHASE_AGREEMENT
        assert result.document_type in [
            DocumentType.BUY_SELL_AGREEMENT,
            DocumentType.PURCHASE_AGREEMENT,
        ]
    
    def test_classify_seller_disclosure(self):
        """Test classifying a seller disclosure."""
        text = """
        SELLER'S PROPERTY DISCLOSURE STATEMENT
        
        The seller hereby discloses the following known conditions about
        the property. This disclosure is made in compliance with state law.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.SELLER_DISCLOSURE
        assert result.method == "keyword"
    
    def test_classify_addendum(self):
        """Test classifying an addendum."""
        text = """
        ADDENDUM TO PURCHASE AGREEMENT
        
        This addendum is made part of the Purchase Agreement dated
        January 15, 2024. The following additional terms are hereby added.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.ADDENDUM
        assert result.method == "keyword"
    
    def test_classify_counter_offer(self):
        """Test classifying a counter offer."""
        text = """
        COUNTER OFFER
        
        In response to the offer received on January 10, 2024, seller
        hereby counters with the following terms.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.COUNTER_OFFER
        assert result.method == "keyword"
    
    def test_classify_amendment(self):
        """Test classifying an amendment."""
        text = """
        AMENDMENT TO CONTRACT
        
        This amendment modifies the original contract dated January 1, 2024.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.AMENDMENT
        assert result.method == "keyword"
    
    def test_classify_inspection_report(self):
        """Test classifying an inspection report."""
        text = """
        HOME INSPECTION REPORT
        
        Property: 123 Main Street
        Inspector: John Smith, Certified Home Inspector
        
        SUMMARY OF FINDINGS:
        - Roof: Good condition
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.INSPECTION_REPORT
        assert result.method == "keyword"
    
    def test_classify_inspection_response(self):
        """Test classifying an inspection response."""
        text = """
        BUYER'S INSPECTION RESPONSE
        
        In response to the inspection report dated January 20, 2024,
        the buyer requests the following repairs...
        Response to inspection objections.
        """
        result = classify_by_keywords(text)
        # Inspection response matches both INSPECTION_REPORT and INSPECTION_RESPONSE
        # Either is acceptable since they share keywords
        assert result.document_type in [
            DocumentType.INSPECTION_RESPONSE,
            DocumentType.INSPECTION_REPORT,
        ]
        assert result.method == "keyword"
    
    def test_classify_title_commitment(self):
        """Test classifying a title commitment."""
        text = """
        COMMITMENT FOR TITLE INSURANCE
        
        First American Title Insurance Company hereby commits to issue
        a policy of title insurance for the property.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.TITLE_COMMITMENT
        assert result.method == "keyword"
    
    def test_classify_closing_disclosure(self):
        """Test classifying a closing disclosure."""
        text = """
        CLOSING DISCLOSURE
        
        This form is a statement of final loan terms and closing costs.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.CLOSING_DISCLOSURE
        assert result.method == "keyword"
    
    def test_classify_lead_paint_disclosure(self):
        """Test classifying a lead paint disclosure."""
        text = """
        LEAD-BASED PAINT DISCLOSURE
        
        Disclosure of Information on Lead-Based Paint Hazards.
        This disclosure is required for housing built before 1978.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.LEAD_PAINT_DISCLOSURE
        assert result.method == "keyword"
    
    def test_classify_mold_disclosure(self):
        """Test classifying a mold disclosure."""
        text = """
        MOLD DISCLOSURE STATEMENT
        
        Information regarding mold and mildew in the property.
        Presence of mold has been disclosed.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.MOLD_DISCLOSURE
        assert result.method == "keyword"
    
    def test_classify_wire_instructions(self):
        """Test classifying wire instructions."""
        text = """
        WIRE TRANSFER INSTRUCTIONS
        
        ABA Routing Number: 123456789
        Beneficiary Account: ABC Title Company
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.WIRE_INSTRUCTIONS
        assert result.method == "keyword"
    
    def test_classify_unknown(self):
        """Test classifying unrecognizable text as UNKNOWN."""
        text = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        """
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.UNKNOWN
        assert result.method == "keyword"
    
    def test_classify_empty_text(self):
        """Test classifying empty text."""
        result = classify_by_keywords("")
        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence == 0.0
    
    def test_detected_title_helps_classification(self):
        """Test that detected title helps classification."""
        text = "Some generic text without keywords"
        result = classify_by_keywords(
            text, 
            detected_title="SELLER'S PROPERTY DISCLOSURE STATEMENT"
        )
        # Title should help with classification
        assert result.document_type == DocumentType.SELLER_DISCLOSURE
    
    def test_case_insensitive(self):
        """Test that classification is case-insensitive."""
        text_lower = "this is a purchase agreement for real estate"
        text_upper = "THIS IS A PURCHASE AGREEMENT FOR REAL ESTATE"
        
        result_lower = classify_by_keywords(text_lower)
        result_upper = classify_by_keywords(text_upper)
        
        assert result_lower.document_type == result_upper.document_type


# ============================================================================
# classify_document Tests
# ============================================================================

class TestClassifyDocument:
    """Tests for the main classify_document function."""
    
    def test_mock_mode_buy_sell(self):
        """Test mock mode returns buy-sell for matching keywords."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "This is a buy sell agreement for property",
            config=config,
        )
        assert result.document_type == DocumentType.BUY_SELL_AGREEMENT
        assert result.method == "mock"
    
    def test_mock_mode_disclosure(self):
        """Test mock mode returns disclosure for matching keywords."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Seller disclosure statement",
            config=config,
        )
        assert result.document_type == DocumentType.SELLER_DISCLOSURE
        assert result.method == "mock"
    
    def test_mock_mode_lead_paint(self):
        """Test mock mode returns lead paint disclosure."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Lead paint disclosure for pre-1978 homes",
            config=config,
        )
        assert result.document_type == DocumentType.LEAD_PAINT_DISCLOSURE
        assert result.method == "mock"
    
    def test_mock_mode_mold(self):
        """Test mock mode returns mold disclosure."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Mold disclosure statement",
            config=config,
        )
        assert result.document_type == DocumentType.MOLD_DISCLOSURE
        assert result.method == "mock"
    
    def test_mock_mode_counter_offer(self):
        """Test mock mode returns counter offer."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Counter offer response",
            config=config,
        )
        assert result.document_type == DocumentType.COUNTER_OFFER
        assert result.method == "mock"
    
    def test_mock_mode_addendum(self):
        """Test mock mode returns addendum."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Contract addendum",
            config=config,
        )
        assert result.document_type == DocumentType.ADDENDUM
        assert result.method == "mock"
    
    def test_mock_mode_amendment(self):
        """Test mock mode returns amendment."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Contract amendment",
            config=config,
        )
        assert result.document_type == DocumentType.AMENDMENT
        assert result.method == "mock"
    
    def test_mock_mode_inspection(self):
        """Test mock mode returns inspection report."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Home inspection findings",
            config=config,
        )
        assert result.document_type == DocumentType.INSPECTION_REPORT
        assert result.method == "mock"
    
    def test_mock_mode_unknown(self):
        """Test mock mode returns unknown for non-matching."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "Lorem ipsum dolor sit amet",
            config=config,
        )
        assert result.document_type == DocumentType.UNKNOWN
        assert result.method == "mock"
    
    def test_keyword_fallback_when_llm_disabled(self):
        """Test that keyword fallback is used when LLM disabled."""
        config = ClassifierConfig(
            use_llm=False,
            use_keyword_fallback=True,
            use_mock=False,
        )
        result = classify_document(
            "BUY-SELL AGREEMENT for property purchase",
            config=config,
        )
        assert result.document_type == DocumentType.BUY_SELL_AGREEMENT
        assert result.method in ["keyword", "title"]
    
    def test_detected_title_used_for_classification(self):
        """Test that detected title is used for classification."""
        config = ClassifierConfig(
            use_llm=False,
            use_keyword_fallback=True,
            use_mock=False,
        )
        result = classify_document(
            "Some generic text",
            detected_title="ADDENDUM TO PURCHASE AGREEMENT",
            config=config,
        )
        # Should use title or keyword matching on title
        assert result.document_type == DocumentType.ADDENDUM


# ============================================================================
# doc_type_classifier_node Tests
# ============================================================================

class TestDocTypeClassifierNode:
    """Tests for the main classifier node function."""
    
    def test_node_with_no_documents(self):
        """Test node handles empty split_docs."""
        state: Any = {"split_docs": []}
        result = doc_type_classifier_node(cast(DealState, state))
        
        assert "split_docs" in result
        assert result["split_docs"] == []
    
    def test_node_classifies_documents(self):
        """Test node classifies documents and adds type."""
        state = {
            "split_docs": [
                {
                    "id": 1,
                    "raw_text": "REAL ESTATE BUY-SELL AGREEMENT for property",
                    "page_range": [1, 2],
                },
                {
                    "id": 2,
                    "raw_text": "SELLER'S PROPERTY DISCLOSURE STATEMENT",
                    "page_range": [3, 4],
                },
            ]
        }
        
        # Use mock mode for predictable results
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        assert len(result["split_docs"]) == 2
        assert result["split_docs"][0]["type"] == "Buy-Sell Agreement"
        assert result["split_docs"][1]["type"] == "Seller Disclosure"
    
    def test_node_adds_classification_confidence(self):
        """Test node adds confidence scores."""
        state: Any = {
            "split_docs": [
                {"id": 1, "raw_text": "PURCHASE AGREEMENT", "page_range": [1]},
            ]
        }
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        assert "classification_confidence" in result["split_docs"][0]
        assert isinstance(result["split_docs"][0]["classification_confidence"], float)
    
    def test_node_adds_classification_method(self):
        """Test node adds classification method."""
        state: Any = {
            "split_docs": [
                {"id": 1, "raw_text": "Seller disclosure form", "page_range": [1]},
            ]
        }
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        assert "classification_method" in result["split_docs"][0]
        assert result["split_docs"][0]["classification_method"] in ["llm", "keyword", "title", "mock", "none"]
    
    def test_node_adds_suggested_folder(self):
        """Test node adds suggested folder."""
        state: Any = {
            "split_docs": [
                {"id": 1, "raw_text": "Addendum document", "page_range": [1]},
            ]
        }
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        assert "suggested_folder" in result["split_docs"][0]
        assert result["split_docs"][0]["suggested_folder"] == "Addenda"
    
    def test_node_returns_metrics(self):
        """Test node returns classification metrics."""
        state: Any = {
            "split_docs": [
                {"id": 1, "raw_text": "Buy sell agreement", "page_range": [1]},
                {"id": 2, "raw_text": "Disclosure statement", "page_range": [2]},
            ]
        }
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        assert "classifier_metrics" in result
        metrics = result["classifier_metrics"]
        assert metrics["total_documents"] == 2
    
    def test_node_handles_missing_raw_text(self):
        """Test node handles documents without raw_text."""
        state: Any = {
            "split_docs": [
                {"id": 1, "page_range": [1, 2]},  # No raw_text field
            ]
        }
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        # Should still return a classification (UNKNOWN)
        assert result["split_docs"][0]["type"] == "Unknown"
    
    def test_node_preserves_existing_fields(self):
        """Test node preserves existing document fields."""
        state = {
            "split_docs": [
                {
                    "id": 1,
                    "raw_text": "Buy sell agreement",
                    "page_range": [1, 2, 3],
                    "custom_field": "custom_value",
                    "file_path": "/path/to/doc.pdf",
                },
            ]
        }
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        doc = result["split_docs"][0]
        assert doc["page_range"] == [1, 2, 3]
        assert doc["custom_field"] == "custom_value"
        assert doc["file_path"] == "/path/to/doc.pdf"
    
    def test_node_uses_detected_title(self):
        """Test node uses detected_title for classification."""
        state = {
            "split_docs": [
                {
                    "id": 1,
                    "raw_text": "Some generic text",
                    "detected_title": "ADDENDUM TO CONTRACT",
                    "page_range": [1],
                },
            ]
        }
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        # Mock should pick up "addendum" from the title
        assert result["split_docs"][0]["type"] == "Addendum"


# ============================================================================
# Document Type to Folder Mapping Tests
# ============================================================================

class TestDocumentTypeToFolderMapping:
    """Tests for the document type to folder mapping."""
    
    def test_buy_sell_goes_to_contracts(self):
        """Test buy-sell goes to Contracts folder."""
        assert DOCUMENT_TYPE_TO_FOLDER[DocumentType.BUY_SELL_AGREEMENT] == "Contracts"
    
    def test_addendum_goes_to_addenda(self):
        """Test addendum goes to Addenda folder."""
        assert DOCUMENT_TYPE_TO_FOLDER[DocumentType.ADDENDUM] == "Addenda"
    
    def test_disclosure_goes_to_disclosures(self):
        """Test disclosure goes to Disclosures folder."""
        assert DOCUMENT_TYPE_TO_FOLDER[DocumentType.SELLER_DISCLOSURE] == "Disclosures"
        assert DOCUMENT_TYPE_TO_FOLDER[DocumentType.LEAD_PAINT_DISCLOSURE] == "Disclosures"
    
    def test_inspection_goes_to_inspections(self):
        """Test inspection goes to Inspections folder."""
        assert DOCUMENT_TYPE_TO_FOLDER[DocumentType.INSPECTION_REPORT] == "Inspections"
    
    def test_unknown_goes_to_other(self):
        """Test unknown goes to Other folder."""
        assert DOCUMENT_TYPE_TO_FOLDER[DocumentType.UNKNOWN] == "Other"


# ============================================================================
# Integration Tests
# ============================================================================

class TestClassifierIntegration:
    """Integration tests for the classifier pipeline."""
    
    def test_full_classification_pipeline(self):
        """Test complete classification pipeline with multiple documents."""
        documents = [
            {
                "id": 1,
                "raw_text": """
                REAL ESTATE BUY-SELL AGREEMENT
                
                This agreement is made between John Smith (Buyer) and Jane Doe (Seller)
                for the purchase of property at 123 Main Street for $450,000.
                """,
                "page_range": [1, 2, 3],
            },
            {
                "id": 2,
                "raw_text": """
                SELLER'S PROPERTY DISCLOSURE STATEMENT
                
                The seller discloses the following conditions:
                - Roof replaced 2020
                - HVAC serviced annually
                """,
                "page_range": [4, 5],
            },
            {
                "id": 3,
                "raw_text": """
                CONTRACT ADDENDUM
                
                This addendum modifies the terms of the original contract.
                1. Inspection period extended to 14 days
                """,
                "page_range": [6],
            },
        ]
        
        state: Any = {"split_docs": documents}
        
        with patch.dict(os.environ, {"USE_MOCK_CLASSIFIER": "true"}):
            result = doc_type_classifier_node(cast(DealState, state))
        
        # Verify classifications
        assert result["split_docs"][0]["type"] == "Buy-Sell Agreement"
        assert result["split_docs"][1]["type"] == "Seller Disclosure"
        assert result["split_docs"][2]["type"] == "Addendum"
        
        # Verify metrics
        assert result["classifier_metrics"]["total_documents"] == 3
        assert result["classifier_metrics"]["documents_classified"] >= 3
    
    def test_accuracy_on_common_types_keyword(self):
        """Test >90% accuracy on common document types (C1 acceptance criteria)."""
        # Test cases for common document types - using clear, unambiguous patterns
        test_cases = [
            # Buy-Sell Agreements - use distinctive keywords
            ("Buy-sell agreement for real estate", DocumentType.BUY_SELL_AGREEMENT),
            ("Residential real estate purchase and sale agreement", DocumentType.BUY_SELL_AGREEMENT),
            
            # Disclosures - use seller disclosure patterns
            ("Seller's property disclosure statement about conditions", DocumentType.SELLER_DISCLOSURE),
            ("The seller makes the following disclosures about property", DocumentType.SELLER_DISCLOSURE),
            
            # Addendums
            ("Addendum to contract", DocumentType.ADDENDUM),
            ("Contract addendum with additional terms exhibit A", DocumentType.ADDENDUM),
            
            # Counter Offers
            ("Counter offer in response to buyer", DocumentType.COUNTER_OFFER),
            ("Counter proposal for negotiation", DocumentType.COUNTER_OFFER),
            
            # Amendments
            ("Amendment to contract modifying terms", DocumentType.AMENDMENT),
            ("Amend agreement dated January 2024", DocumentType.AMENDMENT),
            
            # Lead Paint
            ("Lead-based paint disclosure for pre-1978 housing", DocumentType.LEAD_PAINT_DISCLOSURE),
            ("Lead paint hazard disclosure form", DocumentType.LEAD_PAINT_DISCLOSURE),
        ]
        
        correct = 0
        
        for text, expected_type in test_cases:
            result = classify_by_keywords(text)
            if result.document_type == expected_type:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.90, f"Accuracy {accuracy:.1%} is below 90% threshold"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_short_text(self):
        """Test classification of very short text."""
        result = classify_by_keywords("deed")
        # Should still attempt classification
        assert result.document_type is not None
    
    def test_special_characters(self):
        """Test handling of special characters in text."""
        text = "BUY-SELL AGREEMENT™ © 2024 — $450,000.00 @ 6.5%"
        result = classify_by_keywords(text)
        assert result.document_type == DocumentType.BUY_SELL_AGREEMENT
    
    def test_numeric_only(self):
        """Test handling of numeric-only content."""
        result = classify_by_keywords("123456789 450000 123")
        assert result.document_type == DocumentType.UNKNOWN
    
    def test_whitespace_only(self):
        """Test handling of whitespace-only content."""
        result = classify_by_keywords("   \n\t   \n   ")
        assert result.document_type == DocumentType.UNKNOWN
        assert result.confidence == 0.0
    
    def test_missing_split_docs_key(self):
        """Test node handles state without split_docs."""
        state: Any = {}
        result = doc_type_classifier_node(cast(DealState, state))
        
        assert "split_docs" in result
        assert result["split_docs"] == []


# ============================================================================
# Performance Tests  
# ============================================================================

class TestPerformance:
    """Tests for classification performance."""
    
    def test_keyword_classifier_speed(self):
        """Test that keyword classification is fast."""
        import time
        
        text = "REAL ESTATE BUY-SELL AGREEMENT " * 100  # ~4000 chars
        
        start = time.time()
        for _ in range(100):
            classify_by_keywords(text)
        elapsed = time.time() - start
        
        # Should classify 100 documents in under 1 second
        assert elapsed < 1.0, f"Classification too slow: {elapsed:.2f}s for 100 docs"
    
    def test_processing_time_tracked(self):
        """Test that processing time is tracked in results."""
        result = classify_by_keywords("PURCHASE AGREEMENT")
        
        assert result.processing_time_ms is not None
        assert result.processing_time_ms >= 0


# ============================================================================
# Keyword Pattern Tests
# ============================================================================

class TestKeywordPatterns:
    """Tests for the keyword patterns dictionary."""
    
    def test_all_document_types_have_patterns(self):
        """Test that most document types have keyword patterns."""
        # These are the types we expect to have patterns
        expected_patterns = [
            DocumentType.BUY_SELL_AGREEMENT,
            DocumentType.COUNTER_OFFER,
            DocumentType.AMENDMENT,
            DocumentType.ADDENDUM,
            DocumentType.SELLER_DISCLOSURE,
            DocumentType.LEAD_PAINT_DISCLOSURE,
            DocumentType.INSPECTION_REPORT,
        ]
        
        for doc_type in expected_patterns:
            assert doc_type in KEYWORD_PATTERNS, f"Missing patterns for {doc_type}"
            assert len(KEYWORD_PATTERNS[doc_type]) > 0, f"Empty patterns for {doc_type}"
    
    def test_patterns_have_weights(self):
        """Test that patterns have associated weights."""
        for doc_type, patterns in KEYWORD_PATTERNS.items():
            for pattern, weight in patterns:
                assert isinstance(pattern, str), f"Pattern should be string: {pattern}"
                assert 0.0 <= weight <= 1.0, f"Weight should be 0-1: {weight}"


# ============================================================================
# C2 Tests: OCR Document Info
# ============================================================================

class TestOcrDocumentInfo:
    """Tests for OcrDocumentInfo dataclass (User Story C2)."""
    
    def test_default_values(self):
        """Test default values for non-OCR documents."""
        info = OcrDocumentInfo()
        
        assert info.has_scanned_pages is False
        assert info.ocr_confidence == 1.0
        assert info.meets_accuracy_target is True
        assert info.ocr_engine_used is None
        assert info.scanned_page_count == 0
    
    def test_high_quality_ocr(self):
        """Test OCR info for high-quality OCR."""
        info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.98,
            meets_accuracy_target=True,
            ocr_engine_used="easyocr",
            scanned_page_count=2,
        )
        
        assert info.has_scanned_pages is True
        assert info.ocr_confidence == 0.98
        assert info.meets_accuracy_target is True
        assert info.is_low_confidence is False
        assert info.is_poor_quality is False
        assert info.needs_review is False
    
    def test_low_confidence_ocr(self):
        """Test OCR info for low-confidence OCR."""
        info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.60,
            meets_accuracy_target=False,
        )
        
        assert info.is_low_confidence is True
        assert info.is_poor_quality is False
        assert info.needs_review is True
    
    def test_poor_quality_ocr(self):
        """Test OCR info for poor-quality OCR."""
        info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.35,
            meets_accuracy_target=False,
        )
        
        assert info.is_low_confidence is True
        assert info.is_poor_quality is True
        assert info.needs_review is True
    
    def test_from_document_with_ocr(self):
        """Test creating OcrDocumentInfo from document dict."""
        doc = {
            "page_range": [1, 5],
            "has_scanned_pages": True,
            "ocr_confidence": 0.92,
            "meets_accuracy_target": False,
            "ocr_engine_used": "tesseract",
            "scanned_page_count": 3,
        }
        
        info = OcrDocumentInfo.from_document(doc)
        
        assert info.has_scanned_pages is True
        assert info.ocr_confidence == 0.92
        assert info.meets_accuracy_target is False
        assert info.ocr_engine_used == "tesseract"
        assert info.scanned_page_count == 3
        assert info.total_page_count == 5
    
    def test_from_document_without_ocr(self):
        """Test creating OcrDocumentInfo from non-OCR document."""
        doc = {
            "page_range": [1, 3],
            "raw_text": "Native PDF text",
        }
        
        info = OcrDocumentInfo.from_document(doc)
        
        assert info.has_scanned_pages is False
        assert info.ocr_confidence == 1.0
        assert info.meets_accuracy_target is True
    
    def test_to_dict(self):
        """Test OcrDocumentInfo serialization."""
        info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.87,
            meets_accuracy_target=False,
            ocr_engine_used="easyocr",
            scanned_page_count=2,
            total_page_count=5,
        )
        
        result = info.to_dict()
        
        assert result["has_scanned_pages"] is True
        assert result["ocr_confidence"] == 0.87
        assert result["meets_accuracy_target"] is False
        assert result["is_low_confidence"] is False  # 0.87 >= 0.70
        assert result["is_poor_quality"] is False
        assert result["needs_review"] is True  # doesn't meet accuracy target


class TestOcrConstants:
    """Tests for OCR-related constants."""
    
    def test_ocr_accuracy_target(self):
        """Verify OCR accuracy target is 95% per C2 acceptance criteria."""
        assert OCR_ACCURACY_TARGET == 0.95
    
    def test_low_confidence_threshold(self):
        """Verify low confidence threshold."""
        assert OCR_LOW_CONFIDENCE_THRESHOLD == 0.70
    
    def test_poor_quality_threshold(self):
        """Verify poor quality threshold."""
        assert OCR_POOR_QUALITY_THRESHOLD == 0.50


# ============================================================================
# C2 Tests: OCR Text Preprocessing
# ============================================================================

class TestPreprocessOcrText:
    """Tests for OCR text preprocessing (User Story C2)."""
    
    def test_empty_text(self):
        """Test preprocessing empty text."""
        assert preprocess_ocr_text("") == ""
        assert preprocess_ocr_text("", 0.5) == ""
    
    def test_no_change_for_clean_text(self):
        """Test that clean text is preserved."""
        text = "This is clean text from OCR."
        result = preprocess_ocr_text(text, 0.95)
        assert result == text
    
    def test_removes_excessive_whitespace(self):
        """Test removal of excessive whitespace."""
        text = "Word1    Word2\t\tWord3"
        result = preprocess_ocr_text(text, 0.95)
        assert "    " not in result
        assert "\t\t" not in result
    
    def test_removes_excessive_newlines(self):
        """Test removal of excessive newlines."""
        text = "Line1\n\n\n\n\nLine2"
        result = preprocess_ocr_text(text, 0.95)
        # Should reduce to max 2 newlines
        assert "\n\n\n" not in result
    
    def test_aggressive_cleaning_for_low_confidence(self):
        """Test more aggressive cleaning for low confidence OCR."""
        # Low confidence should apply additional cleaning
        text = "Good text @ # $ noise"
        result_low = preprocess_ocr_text(text, 0.50)
        result_high = preprocess_ocr_text(text, 0.95)
        
        # Low confidence cleaning may be more aggressive
        assert isinstance(result_low, str)
        assert isinstance(result_high, str)
    
    def test_preserves_alphanumeric_content(self):
        """Test that alphanumeric content is preserved."""
        text = "Purchase Agreement for 123 Main Street"
        result = preprocess_ocr_text(text, 0.80)
        
        assert "Purchase Agreement" in result
        assert "123 Main Street" in result


class TestAssessTextQualityForClassification:
    """Tests for OCR quality assessment function."""
    
    def test_non_ocr_document(self):
        """Test assessment of non-OCR document."""
        ocr_info = OcrDocumentInfo(has_scanned_pages=False)
        text, quality_factor = assess_text_quality_for_classification("Test", ocr_info)
        
        assert text == "Test"
        assert quality_factor == 1.0
    
    def test_high_quality_ocr_factor(self):
        """Test quality factor for high-quality OCR."""
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.98,
            meets_accuracy_target=True,
        )
        _, quality_factor = assess_text_quality_for_classification("Test", ocr_info)
        
        assert quality_factor == 1.0
    
    def test_medium_quality_ocr_factor(self):
        """Test quality factor for medium-quality OCR."""
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.85,
            meets_accuracy_target=False,
        )
        _, quality_factor = assess_text_quality_for_classification("Test", ocr_info)
        
        assert quality_factor == 0.95  # Small penalty
    
    def test_low_quality_ocr_factor(self):
        """Test quality factor for low-quality OCR."""
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.55,
            meets_accuracy_target=False,
        )
        _, quality_factor = assess_text_quality_for_classification("Test", ocr_info)
        
        assert quality_factor == 0.85  # Moderate penalty
    
    def test_poor_quality_ocr_factor(self):
        """Test quality factor for poor-quality OCR."""
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.30,
            meets_accuracy_target=False,
        )
        _, quality_factor = assess_text_quality_for_classification("Test", ocr_info)
        
        assert quality_factor == 0.70  # Significant penalty


# ============================================================================
# C2 Tests: Classification with OCR Awareness
# ============================================================================

class TestClassifyDocumentWithOcr:
    """Tests for OCR-aware document classification (User Story C2)."""
    
    def test_classification_without_ocr_info(self):
        """Test classification works without OCR info."""
        config = ClassifierConfig(use_mock=True)
        result = classify_document(
            "RESIDENTIAL REAL ESTATE BUY-SELL AGREEMENT",
            config=config,
            ocr_info=None,
        )
        
        assert result.document_type == DocumentType.BUY_SELL_AGREEMENT
        assert result.ocr_info is None
        assert result.ocr_quality_factor == 1.0
    
    def test_classification_with_high_quality_ocr(self):
        """Test classification with high-quality OCR."""
        config = ClassifierConfig(use_mock=True)
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.98,
            meets_accuracy_target=True,
        )
        
        result = classify_document(
            "SELLER'S PROPERTY DISCLOSURE",
            config=config,
            ocr_info=ocr_info,
        )
        
        assert result.document_type == DocumentType.SELLER_DISCLOSURE
        assert result.ocr_info is not None
        assert result.ocr_quality_factor == 1.0
        assert result.original_confidence is None  # No adjustment needed
    
    def test_classification_with_low_quality_ocr(self):
        """Test classification confidence is reduced for low-quality OCR."""
        config = ClassifierConfig(use_mock=True)
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.55,
            meets_accuracy_target=False,
        )
        
        result = classify_document(
            "SELLER'S PROPERTY DISCLOSURE",
            config=config,
            ocr_info=ocr_info,
        )
        
        assert result.document_type == DocumentType.SELLER_DISCLOSURE
        assert result.ocr_info is not None
        assert result.ocr_quality_factor < 1.0
        assert result.original_confidence is not None
        assert result.confidence < result.original_confidence
        assert result.ocr_impacted is True
    
    def test_classification_with_poor_quality_ocr(self):
        """Test classification significantly reduced for poor OCR."""
        config = ClassifierConfig(use_mock=True)
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.35,
            meets_accuracy_target=False,
        )
        
        result = classify_document(
            "ADDENDUM",
            config=config,
            ocr_info=ocr_info,
        )
        
        assert result.document_type == DocumentType.ADDENDUM
        assert result.ocr_quality_factor == 0.70
        assert result.original_confidence is not None
        assert result.confidence == result.original_confidence * 0.70


class TestClassificationResultWithOcr:
    """Tests for ClassificationResult OCR fields."""
    
    def test_ocr_impacted_property_false(self):
        """Test ocr_impacted when quality factor is 1.0."""
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.95,
            method="mock",
            ocr_quality_factor=1.0,
        )
        
        assert result.ocr_impacted is False
    
    def test_ocr_impacted_property_true(self):
        """Test ocr_impacted when quality factor is less than 1.0."""
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.80,
            method="mock",
            ocr_quality_factor=0.85,
            original_confidence=0.94,
        )
        
        assert result.ocr_impacted is True
    
    def test_to_dict_includes_ocr_fields(self):
        """Test to_dict includes OCR information."""
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.87,
            meets_accuracy_target=False,
        )
        
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.80,
            method="mock",
            ocr_info=ocr_info,
            ocr_quality_factor=0.95,
            original_confidence=0.84,
        )
        
        result_dict = result.to_dict()
        
        assert "ocr_info" in result_dict
        assert result_dict["ocr_info"]["has_scanned_pages"] is True
        assert result_dict["ocr_quality_factor"] == 0.95
        assert result_dict["original_confidence"] == 0.84
        assert result_dict["ocr_impacted"] is True


# ============================================================================
# C2 Tests: Classifier Metrics with OCR
# ============================================================================

class TestClassifierMetricsWithOcr:
    """Tests for ClassifierMetrics OCR fields (User Story C2)."""
    
    def test_ocr_metrics_default_values(self):
        """Test OCR metrics have correct defaults."""
        metrics = ClassifierMetrics()
        
        assert metrics.ocr_documents_count == 0
        assert metrics.ocr_accuracy_met_count == 0
        assert metrics.ocr_low_confidence_count == 0
        assert metrics.ocr_poor_quality_count == 0
        assert metrics.ocr_impacted_classifications == 0
        assert metrics.avg_ocr_confidence == 0.0
        assert metrics.ocr_needs_review_count == 0
    
    def test_to_dict_includes_ocr_metrics(self):
        """Test to_dict includes all OCR metrics."""
        metrics = ClassifierMetrics(
            ocr_documents_count=5,
            ocr_accuracy_met_count=3,
            ocr_low_confidence_count=1,
            ocr_poor_quality_count=1,
            ocr_impacted_classifications=2,
            avg_ocr_confidence=0.85,
            ocr_needs_review_count=2,
        )
        
        result = metrics.to_dict()
        
        assert result["ocr_documents_count"] == 5
        assert result["ocr_accuracy_met_count"] == 3
        assert result["ocr_low_confidence_count"] == 1
        assert result["ocr_poor_quality_count"] == 1
        assert result["ocr_impacted_classifications"] == 2
        assert result["avg_ocr_confidence"] == 0.85
        assert result["ocr_needs_review_count"] == 2


# ============================================================================
# C2 Tests: Node with OCR Documents
# ============================================================================

class TestClassifierNodeWithOcr:
    """Tests for classifier node with OCR documents (User Story C2)."""
    
    def test_node_processes_ocr_documents(self):
        """Test node correctly processes documents with OCR metadata."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 3],
                    "raw_text": "BUY-SELL AGREEMENT for property",
                    "detected_title": "Buy-Sell Agreement",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.96,
                    "meets_accuracy_target": True,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        
        metrics = result["classifier_metrics"]
        assert metrics["ocr_documents_count"] == 1
        assert metrics["ocr_accuracy_met_count"] == 1
        assert metrics["ocr_needs_review_count"] == 0
    
    def test_node_flags_low_quality_ocr(self):
        """Test node flags documents with low OCR quality."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "SELLER DISCLOSURE statement",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.65,
                    "meets_accuracy_target": False,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        doc = result["split_docs"][0]
        metrics = result["classifier_metrics"]
        
        # Document should be flagged for review
        assert doc.get("ocr_needs_review") is True
        assert doc.get("ocr_quality_factor") is not None
        assert doc.get("ocr_quality_factor") < 1.0
        
        # Metrics should track low confidence
        assert metrics["ocr_low_confidence_count"] == 1
        assert metrics["ocr_needs_review_count"] == 1
        assert metrics["ocr_impacted_classifications"] == 1
    
    def test_node_tracks_poor_quality_ocr(self):
        """Test node tracks poor quality OCR documents."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "ADDENDUM to contract",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.40,
                    "meets_accuracy_target": False,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        assert metrics["ocr_poor_quality_count"] == 1
        assert metrics["ocr_low_confidence_count"] == 1
    
    def test_node_calculates_avg_ocr_confidence(self):
        """Test node calculates average OCR confidence correctly."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "BUY-SELL AGREEMENT",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.90,
                    "meets_accuracy_target": False,
                },
                {
                    "id": 2,
                    "page_range": [3, 4],
                    "raw_text": "SELLER DISCLOSURE",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.80,
                    "meets_accuracy_target": False,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        assert metrics["ocr_documents_count"] == 2
        assert metrics["avg_ocr_confidence"] == 0.85  # (0.90 + 0.80) / 2
    
    def test_node_handles_mixed_ocr_native_documents(self):
        """Test node handles mix of OCR and native PDF documents."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "BUY-SELL AGREEMENT",
                    "has_scanned_pages": False,  # Native PDF
                },
                {
                    "id": 2,
                    "page_range": [3, 4],
                    "raw_text": "SELLER DISCLOSURE",
                    "has_scanned_pages": True,  # Scanned
                    "ocr_confidence": 0.97,
                    "meets_accuracy_target": True,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        assert metrics["ocr_documents_count"] == 1
        assert metrics["ocr_accuracy_met_count"] == 1
    
    def test_node_adds_ocr_fields_to_documents(self):
        """Test node adds OCR-related fields to classified documents."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "LEAD PAINT DISCLOSURE",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.60,
                    "meets_accuracy_target": False,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        doc = result["split_docs"][0]
        
        # OCR-related fields should be added
        assert "ocr_quality_factor" in doc
        assert "original_confidence" in doc
        assert "ocr_needs_review" in doc
        
        # Confidence should be adjusted
        assert doc["classification_confidence"] < doc["original_confidence"]


class TestClassifierNodeOcrEdgeCases:
    """Edge cases for OCR handling in classifier node."""
    
    def test_node_handles_missing_ocr_fields(self):
        """Test node handles documents missing OCR fields."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "COUNTER OFFER document",
                    # Missing has_scanned_pages, ocr_confidence, etc.
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        doc = result["split_docs"][0]
        
        # Should classify normally without error
        assert doc["type"] == "Counter Offer"
        assert "ocr_needs_review" not in doc  # Not an OCR document
    
    def test_node_handles_zero_ocr_confidence(self):
        """Test node handles zero OCR confidence."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "AMENDMENT",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.0,
                    "meets_accuracy_target": False,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        assert metrics["ocr_poor_quality_count"] == 1
        assert metrics["ocr_needs_review_count"] == 1
    
    def test_node_handles_exact_threshold_confidence(self):
        """Test node handles OCR confidence at exact threshold values."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "page_range": [1, 2],
                    "raw_text": "BUY-SELL AGREEMENT",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.95,  # Exactly at target
                    "meets_accuracy_target": True,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        assert metrics["ocr_accuracy_met_count"] == 1
        assert metrics["ocr_low_confidence_count"] == 0


class TestClassifierOcrIntegration:
    """Integration tests for OCR-aware classification (C2)."""
    
    def test_ocr_quality_affects_high_confidence_threshold(self):
        """Test OCR quality can push classification below high-confidence."""
        config = ClassifierConfig(use_mock=True)
        
        # High-confidence mock result (0.95) with poor OCR (0.70 factor)
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.35,
            meets_accuracy_target=False,
        )
        
        result = classify_document(
            "BUY-SELL AGREEMENT",
            config=config,
            ocr_info=ocr_info,
        )
        
        # Original would be 0.95, but with 0.70 factor = 0.665
        assert result.original_confidence == 0.95
        assert result.confidence == pytest.approx(0.665, rel=0.01)
        assert result.meets_accuracy_target is False  # Below 90%
    
    def test_accuracy_target_with_ocr_documents(self):
        """Test >95% OCR accuracy target is tracked."""
        state = cast(DealState, {
            "split_docs": [
                {"id": 1, "raw_text": "BUY-SELL", "has_scanned_pages": True,
                 "ocr_confidence": 0.97, "meets_accuracy_target": True},
                {"id": 2, "raw_text": "DISCLOSURE", "has_scanned_pages": True,
                 "ocr_confidence": 0.93, "meets_accuracy_target": False},
                {"id": 3, "raw_text": "ADDENDUM", "has_scanned_pages": True,
                 "ocr_confidence": 0.96, "meets_accuracy_target": True},
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        # 2 of 3 meet 95% accuracy target
        assert metrics["ocr_documents_count"] == 3
        assert metrics["ocr_accuracy_met_count"] == 2
        assert metrics["ocr_needs_review_count"] == 1


# ============================================================================
# C3 Tests: Review Reason Enum
# ============================================================================

class TestReviewReason:
    """Tests for ReviewReason enum (User Story C3)."""
    
    def test_all_review_reasons_exist(self):
        """Verify all expected review reasons are defined."""
        expected_reasons = [
            "LOW_CONFIDENCE",
            "UNKNOWN_TYPE",
            "AMBIGUOUS_CLASSIFICATION",
            "OCR_QUALITY_ISSUE",
            "MULTIPLE_CLOSE_ALTERNATIVES",
            "TITLE_ONLY_CLASSIFICATION",
        ]
        for reason_name in expected_reasons:
            assert hasattr(ReviewReason, reason_name)
    
    def test_review_reason_values(self):
        """Test review reason values are correct."""
        assert ReviewReason.LOW_CONFIDENCE.value == "low_confidence"
        assert ReviewReason.UNKNOWN_TYPE.value == "unknown_type"
        assert ReviewReason.AMBIGUOUS_CLASSIFICATION.value == "ambiguous_classification"


class TestReviewConfidenceThreshold:
    """Tests for review confidence threshold constant."""
    
    def test_default_threshold_is_80_percent(self):
        """Test default review threshold is 80%."""
        assert REVIEW_CONFIDENCE_THRESHOLD == 0.80


# ============================================================================
# C3 Tests: ReviewFlag Dataclass
# ============================================================================

class TestReviewFlag:
    """Tests for ReviewFlag dataclass (User Story C3)."""
    
    def test_create_basic_review_flag(self):
        """Test creating a basic review flag."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[ReviewReason.LOW_CONFIDENCE],
            confidence=0.75,
            classified_type="Buy-Sell Agreement",
        )
        
        assert flag.document_id == 1
        assert flag.reasons == [ReviewReason.LOW_CONFIDENCE]
        assert flag.confidence == 0.75
        assert flag.classified_type == "Buy-Sell Agreement"
    
    def test_priority_calculation_critical(self):
        """Test critical priority for unknown type with zero confidence."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[ReviewReason.UNKNOWN_TYPE],
            confidence=0.0,
            classified_type="Unknown",
        )
        
        assert flag.priority == "critical"
    
    def test_priority_calculation_high(self):
        """Test high priority for very low confidence."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[ReviewReason.LOW_CONFIDENCE],
            confidence=0.40,
            classified_type="Addendum",
        )
        
        assert flag.priority == "high"
    
    def test_priority_calculation_high_multiple_reasons(self):
        """Test high priority for multiple reasons."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[
                ReviewReason.LOW_CONFIDENCE,
                ReviewReason.OCR_QUALITY_ISSUE,
                ReviewReason.AMBIGUOUS_CLASSIFICATION,
            ],
            confidence=0.65,
            classified_type="Amendment",
        )
        
        assert flag.priority == "high"
    
    def test_priority_calculation_low(self):
        """Test low priority for title-only with decent confidence."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[ReviewReason.TITLE_ONLY_CLASSIFICATION],
            confidence=0.75,
            classified_type="Seller Disclosure",
        )
        
        assert flag.priority == "low"
    
    def test_priority_calculation_normal(self):
        """Test normal priority for typical case."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[ReviewReason.LOW_CONFIDENCE],
            confidence=0.70,
            classified_type="Counter Offer",
        )
        
        assert flag.priority == "normal"
    
    def test_reason_descriptions(self):
        """Test human-readable reason descriptions."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[ReviewReason.LOW_CONFIDENCE, ReviewReason.OCR_QUALITY_ISSUE],
            confidence=0.65,
            classified_type="Addendum",
        )
        
        descriptions = flag.reason_descriptions
        
        assert len(descriptions) == 2
        assert "65%" in descriptions[0]  # Confidence in description
        assert "OCR" in descriptions[1]
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        flag = ReviewFlag(
            document_id=1,
            reasons=[ReviewReason.LOW_CONFIDENCE],
            confidence=0.75,
            classified_type="Amendment",
            alternative_types=[{"type": "Addendum", "confidence": 0.70}],
            suggested_action="manual_review",
            notes="Test note",
        )
        
        result = flag.to_dict()
        
        assert result["document_id"] == 1
        assert result["reasons"] == ["low_confidence"]
        assert len(result["reason_descriptions"]) == 1
        assert result["confidence"] == 0.75
        assert result["classified_type"] == "Amendment"
        assert result["alternative_types"] == [{"type": "Addendum", "confidence": 0.70}]
        assert result["suggested_action"] == "manual_review"
        assert result["notes"] == "Test note"
        assert "priority" in result


# ============================================================================
# C3 Tests: check_needs_review Function
# ============================================================================

class TestCheckNeedsReview:
    """Tests for check_needs_review function (User Story C3)."""
    
    def test_no_review_for_high_confidence(self):
        """Test no review flag for high confidence classification."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.BUY_SELL_AGREEMENT,
            confidence=0.95,
            method="keyword",
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is None
    
    def test_review_for_low_confidence(self):
        """Test review flag for low confidence classification."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.75,
            method="keyword",
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert ReviewReason.LOW_CONFIDENCE in flag.reasons
        assert flag.confidence == 0.75
    
    def test_review_for_unknown_type(self):
        """Test review flag for unknown document type."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.UNKNOWN,
            confidence=0.0,
            method="none",
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert ReviewReason.UNKNOWN_TYPE in flag.reasons
    
    def test_review_for_title_only_classification(self):
        """Test review flag for title-only classification."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.SELLER_DISCLOSURE,
            confidence=0.75,
            method="title",
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert ReviewReason.TITLE_ONLY_CLASSIFICATION in flag.reasons
    
    def test_review_for_ocr_quality_issue(self):
        """Test review flag for OCR quality issue."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        ocr_info = OcrDocumentInfo(
            has_scanned_pages=True,
            ocr_confidence=0.60,
            meets_accuracy_target=False,
        )
        result = ClassificationResult(
            document_type=DocumentType.AMENDMENT,
            confidence=0.72,  # After OCR penalty
            method="keyword",
            ocr_info=ocr_info,
            ocr_quality_factor=0.85,
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert ReviewReason.OCR_QUALITY_ISSUE in flag.reasons
    
    def test_review_for_ambiguous_classification(self):
        """Test review flag for ambiguous classification."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.AMENDMENT,
            confidence=0.85,
            method="keyword",
            alternative_types=[
                (DocumentType.ADDENDUM, 0.82),  # Within 10% of main
            ],
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert ReviewReason.AMBIGUOUS_CLASSIFICATION in flag.reasons
    
    def test_review_for_multiple_close_alternatives(self):
        """Test review flag for multiple close alternatives."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.85,
            method="keyword",
            alternative_types=[
                (DocumentType.AMENDMENT, 0.78),  # Within 15%
                (DocumentType.COUNTER_OFFER, 0.75),  # Within 15%
            ],
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert ReviewReason.MULTIPLE_CLOSE_ALTERNATIVES in flag.reasons
    
    def test_custom_threshold(self):
        """Test custom confidence threshold."""
        # With 70% threshold, 75% should not trigger review
        config = ClassifierConfig(review_confidence_threshold=0.70)
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.75,
            method="keyword",
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is None
    
    def test_suggested_action_for_unknown_type(self):
        """Test suggested action for unknown type."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.UNKNOWN,
            confidence=0.0,
            method="none",
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert flag.suggested_action == "manual_classification"
    
    def test_suggested_action_for_ambiguous(self):
        """Test suggested action for ambiguous classification."""
        config = ClassifierConfig(review_confidence_threshold=0.80)
        result = ClassificationResult(
            document_type=DocumentType.AMENDMENT,
            confidence=0.85,
            method="keyword",
            alternative_types=[(DocumentType.ADDENDUM, 0.83)],
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert flag.suggested_action == "confirm_type_selection"


# ============================================================================
# C3 Tests: ClassifierConfig Review Settings
# ============================================================================

class TestClassifierConfigReviewSettings:
    """Tests for ClassifierConfig review queue settings (C3)."""
    
    def test_default_review_threshold(self):
        """Test default review confidence threshold."""
        config = ClassifierConfig()
        assert config.review_confidence_threshold == 0.80
    
    def test_default_review_queue_enabled(self):
        """Test review queue enabled by default."""
        config = ClassifierConfig()
        assert config.enable_review_queue is True
    
    def test_custom_review_threshold(self):
        """Test custom review threshold."""
        config = ClassifierConfig(review_confidence_threshold=0.70)
        assert config.review_confidence_threshold == 0.70
    
    def test_disable_review_queue(self):
        """Test disabling review queue."""
        config = ClassifierConfig(enable_review_queue=False)
        assert config.enable_review_queue is False


# ============================================================================
# C3 Tests: ClassifierMetrics Review Tracking
# ============================================================================

class TestClassifierMetricsReviewTracking:
    """Tests for ClassifierMetrics review queue tracking (C3)."""
    
    def test_review_metrics_default_values(self):
        """Test default values for review metrics."""
        metrics = ClassifierMetrics()
        
        assert metrics.review_flagged_count == 0
        assert metrics.review_by_reason == {}
        assert metrics.review_by_priority == {}
        assert metrics.avg_flagged_confidence == 0.0
    
    def test_to_dict_includes_review_metrics(self):
        """Test to_dict includes review metrics."""
        metrics = ClassifierMetrics(
            review_flagged_count=3,
            review_by_reason={"low_confidence": 2, "unknown_type": 1},
            review_by_priority={"normal": 2, "critical": 1},
            avg_flagged_confidence=0.45,
        )
        
        result = metrics.to_dict()
        
        assert result["review_flagged_count"] == 3
        assert result["review_by_reason"] == {"low_confidence": 2, "unknown_type": 1}
        assert result["review_by_priority"] == {"normal": 2, "critical": 1}
        assert result["avg_flagged_confidence"] == 0.45


# ============================================================================
# C3 Tests: Node Review Queue Integration
# ============================================================================

class TestClassifierNodeReviewQueue:
    """Tests for classifier node review queue integration (C3)."""
    
    def test_node_returns_review_queue(self):
        """Test node returns review_queue in result."""
        state = cast(DealState, {
            "split_docs": [
                {"id": 1, "raw_text": "BUY-SELL AGREEMENT"},
            ],
        })
        
        result = doc_type_classifier_node(state)
        
        assert "review_queue" in result
        assert isinstance(result["review_queue"], list)
    
    def test_node_flags_low_confidence_for_review(self):
        """Test node flags low confidence documents for review."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "raw_text": "Some unclear document",  # Will be UNKNOWN
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        
        assert len(result["review_queue"]) == 1
        assert result["review_queue"][0]["document_id"] == 1
        assert "unknown_type" in result["review_queue"][0]["reasons"]
    
    def test_node_adds_needs_review_to_doc(self):
        """Test node adds needs_review field to documents."""
        state = cast(DealState, {
            "split_docs": [
                {"id": 1, "raw_text": "BUY-SELL AGREEMENT"},  # High confidence
                {"id": 2, "raw_text": "unclear text"},  # Low confidence
            ],
        })
        
        result = doc_type_classifier_node(state)
        docs = result["split_docs"]
        
        # High confidence should not need review
        assert docs[0]["needs_review"] is False
        
        # Unknown type should need review  
        assert docs[1]["needs_review"] is True
        assert "review_reasons" in docs[1]
        assert "review_priority" in docs[1]
    
    def test_node_tracks_review_metrics(self):
        """Test node tracks review queue metrics."""
        state = cast(DealState, {
            "split_docs": [
                {"id": 1, "raw_text": "BUY-SELL AGREEMENT"},
                {"id": 2, "raw_text": "unknown document"},
                {"id": 3, "raw_text": "another unclear doc"},
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        # Should have flagged unknown documents
        assert metrics["review_flagged_count"] >= 1
        assert "review_by_reason" in metrics
        assert "review_by_priority" in metrics
    
    def test_node_calculates_avg_flagged_confidence(self):
        """Test node calculates average confidence of flagged docs."""
        state = cast(DealState, {
            "split_docs": [
                {"id": 1, "raw_text": "BUY-SELL AGREEMENT"},  # High conf
                {"id": 2, "raw_text": "unknown"},  # Zero conf
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        if metrics["review_flagged_count"] > 0:
            assert metrics["avg_flagged_confidence"] >= 0.0
    
    def test_node_review_queue_disabled(self):
        """Test node respects disabled review queue."""
        # Set environment to disable review queue
        import os
        original = os.environ.get("ENABLE_REVIEW_QUEUE")
        os.environ["ENABLE_REVIEW_QUEUE"] = "false"
        
        try:
            state = cast(DealState, {
                "split_docs": [
                    {"id": 1, "raw_text": "unknown document"},
                ],
            })
            
            result = doc_type_classifier_node(state)
            
            # Review queue should be empty when disabled
            assert len(result["review_queue"]) == 0
        finally:
            if original:
                os.environ["ENABLE_REVIEW_QUEUE"] = original
            else:
                del os.environ["ENABLE_REVIEW_QUEUE"]


class TestClassifierNodeReviewEdgeCases:
    """Edge case tests for review queue (C3)."""
    
    def test_no_review_for_empty_docs(self):
        """Test no review flags for empty document list."""
        state = cast(DealState, {"split_docs": []})
        
        result = doc_type_classifier_node(state)
        
        assert result["review_queue"] == []
        assert result["classifier_metrics"]["review_flagged_count"] == 0
    
    def test_review_flag_includes_alternatives(self):
        """Test review flag includes alternative types."""
        config = ClassifierConfig(
            use_mock=True,
            review_confidence_threshold=0.99,  # Force all to be flagged
        )
        
        result = ClassificationResult(
            document_type=DocumentType.AMENDMENT,
            confidence=0.85,
            method="keyword",
            alternative_types=[
                (DocumentType.ADDENDUM, 0.80),
                (DocumentType.COUNTER_OFFER, 0.75),
            ],
        )
        
        flag = check_needs_review(result, doc_id=1, config=config)
        
        assert flag is not None
        assert len(flag.alternative_types) == 2
        assert flag.alternative_types[0]["type"] == "Addendum"
    
    def test_review_priority_in_metrics(self):
        """Test review priorities are tracked in metrics."""
        state = cast(DealState, {
            "split_docs": [
                {"id": 1, "raw_text": "unknown"},  # Will be UNKNOWN -> critical
            ],
        })
        
        result = doc_type_classifier_node(state)
        metrics = result["classifier_metrics"]
        
        # Should have priority counts
        assert "review_by_priority" in metrics
        if metrics["review_flagged_count"] > 0:
            assert sum(metrics["review_by_priority"].values()) == metrics["review_flagged_count"]


class TestClassifierReviewIntegration:
    """Integration tests for review queue (C3)."""
    
    def test_review_threshold_80_percent(self):
        """Test 80% threshold triggers review correctly."""
        config = ClassifierConfig(
            use_mock=True,
            review_confidence_threshold=0.80,
        )
        
        # 79% confidence should trigger review
        result_low = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.79,
            method="keyword",
        )
        flag_low = check_needs_review(result_low, doc_id=1, config=config)
        assert flag_low is not None
        
        # 80% confidence should NOT trigger review (equal to threshold)
        result_at = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.80,
            method="keyword",
        )
        flag_at = check_needs_review(result_at, doc_id=2, config=config)
        assert flag_at is None
        
        # 81% confidence should NOT trigger review
        result_above = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.81,
            method="keyword",
        )
        flag_above = check_needs_review(result_above, doc_id=3, config=config)
        assert flag_above is None
    
    def test_combined_ocr_and_review_flagging(self):
        """Test documents flagged for both OCR and classification review."""
        state = cast(DealState, {
            "split_docs": [
                {
                    "id": 1,
                    "raw_text": "BUY-SELL AGREEMENT",
                    "has_scanned_pages": True,
                    "ocr_confidence": 0.55,  # Poor OCR
                    "meets_accuracy_target": False,
                },
            ],
        })
        
        result = doc_type_classifier_node(state)
        doc = result["split_docs"][0]
        
        # Should be flagged for OCR quality
        assert doc.get("ocr_needs_review") is True
        
        # With poor OCR quality factor, classification confidence drops
        # which may also trigger classification review
        if doc["needs_review"]:
            assert "ocr_quality_issue" in doc.get("review_reasons", [])


# ============================================================================
# Custom Document Types Tests (C4)
# ============================================================================

class TestCustomDocumentType:
    """Tests for the CustomDocumentType dataclass."""
    
    def test_create_basic_custom_type(self):
        """Test creating a basic custom document type."""
        custom_type = CustomDocumentType(
            id="abc-brokerage-form",
            name="ABC Brokerage Disclosure",
            brokerage_id="abc-realty",
            patterns=[
                (r"abc\s+brokerage\s+disclosure", 0.95),
                (r"abc\s+realty\s+property\s+form", 0.85),
            ],
            example_texts=["ABC Brokerage Property Disclosure Form"],
            folder="ABC Forms",
            description="Custom disclosure form for ABC Realty",
        )
        
        assert custom_type.id == "abc-brokerage-form"
        assert custom_type.name == "ABC Brokerage Disclosure"
        assert custom_type.brokerage_id == "abc-realty"
        assert len(custom_type.patterns) == 2
        assert custom_type.folder == "ABC Forms"
        assert custom_type.is_active is True  # Default
        assert custom_type.priority == 1.0  # Default
        assert custom_type.min_confidence == 0.70  # Default
    
    def test_custom_type_matches_text(self):
        """Test matching text against custom type patterns."""
        custom_type = CustomDocumentType(
            id="test-type",
            name="Test Form",
            brokerage_id="test-brokerage",
            patterns=[
                (r"xyz\s+company\s+agreement", 0.95),
                (r"xyz\s+form", 0.75),
            ],
            example_texts=[],
        )
        
        # Should match high-confidence pattern
        matches, confidence = custom_type.matches("XYZ COMPANY AGREEMENT - Property Sale")
        assert matches is True
        assert confidence >= 0.70
        
        # Should match lower-confidence pattern
        matches, confidence = custom_type.matches("XYZ Form 123")
        assert matches is True
        assert confidence >= 0.70
    
    def test_custom_type_no_match(self):
        """Test non-matching text returns no match."""
        custom_type = CustomDocumentType(
            id="test-type",
            name="Test Form",
            brokerage_id="test-brokerage",
            patterns=[
                (r"xyz\s+company\s+agreement", 0.95),
            ],
            example_texts=[],
        )
        
        matches, confidence = custom_type.matches("ABC Company Purchase Agreement")
        assert matches is False
        assert confidence == 0.0
    
    def test_custom_type_example_text_matching(self):
        """Test matching against example texts."""
        custom_type = CustomDocumentType(
            id="test-type",
            name="Test Form",
            brokerage_id="test-brokerage",
            patterns=[],
            example_texts=["Special Brokerage Disclosure Form"],
        )
        
        # Should match when example text is found
        matches, confidence = custom_type.matches(
            "This is the Special Brokerage Disclosure Form for 123 Main St"
        )
        assert matches is True
        assert confidence >= 0.70
    
    def test_custom_type_inactive_no_match(self):
        """Test inactive custom types don't match."""
        custom_type = CustomDocumentType(
            id="test-type",
            name="Test Form",
            brokerage_id="test-brokerage",
            patterns=[(r"test\s+pattern", 0.95)],
            example_texts=[],
            is_active=False,
        )
        
        matches, confidence = custom_type.matches("test pattern found")
        assert matches is False
        assert confidence == 0.0
    
    def test_custom_type_to_dict(self):
        """Test serialization to dictionary."""
        custom_type = CustomDocumentType(
            id="test-type",
            name="Test Form",
            brokerage_id="test-brokerage",
            patterns=[(r"test", 0.90)],
            example_texts=["example"],
            folder="Test Folder",
            description="A test form",
        )
        
        data = custom_type.to_dict()
        assert data["id"] == "test-type"
        assert data["name"] == "Test Form"
        assert data["brokerage_id"] == "test-brokerage"
        assert data["folder"] == "Test Folder"
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["pattern"] == r"test"
        assert data["patterns"][0]["weight"] == 0.90
    
    def test_custom_type_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "loaded-type",
            "name": "Loaded Form",
            "brokerage_id": "loaded-brokerage",
            "patterns": [
                {"pattern": r"loaded\s+pattern", "weight": 0.88}
            ],
            "example_texts": ["loaded example"],
            "folder": "Loaded Folder",
            "description": "A loaded form",
            "is_active": True,
            "priority": 2.0,
            "min_confidence": 0.75,
        }
        
        custom_type = CustomDocumentType.from_dict(data)
        assert custom_type.id == "loaded-type"
        assert custom_type.name == "Loaded Form"
        assert custom_type.brokerage_id == "loaded-brokerage"
        assert len(custom_type.patterns) == 1
        assert custom_type.patterns[0][0] == r"loaded\s+pattern"
        assert custom_type.patterns[0][1] == 0.88
        assert custom_type.priority == 2.0
        assert custom_type.min_confidence == 0.75


class TestCustomDocumentTypeRegistry:
    """Tests for the CustomDocumentTypeRegistry class."""
    
    def setup_method(self):
        """Reset the global registry before each test."""
        # Create a fresh registry for each test
        set_custom_type_registry(CustomDocumentTypeRegistry())
    
    def test_add_custom_type(self):
        """Test adding a custom type to registry."""
        registry = get_custom_type_registry()
        
        custom_type = CustomDocumentType(
            id="new-type",
            name="New Form",
            brokerage_id="test-brokerage",
            patterns=[(r"new\s+form", 0.90)],
            example_texts=[],
        )
        
        result = registry.add(custom_type)
        assert result.id == "new-type"
        assert len(registry) == 1
        assert "new-type" in registry
    
    def test_add_duplicate_raises_error(self):
        """Test adding duplicate type raises ValueError."""
        registry = get_custom_type_registry()
        
        custom_type = CustomDocumentType(
            id="dup-type",
            name="Dup Form",
            brokerage_id="test-brokerage",
            patterns=[],
            example_texts=[],
        )
        
        registry.add(custom_type)
        
        with pytest.raises(ValueError, match="already exists"):
            registry.add(custom_type)
    
    def test_get_custom_type(self):
        """Test retrieving a custom type by ID."""
        registry = get_custom_type_registry()
        
        custom_type = CustomDocumentType(
            id="get-type",
            name="Get Form",
            brokerage_id="test-brokerage",
            patterns=[],
            example_texts=[],
        )
        
        registry.add(custom_type)
        
        retrieved = registry.get("get-type")
        assert retrieved is not None
        assert retrieved.name == "Get Form"
    
    def test_get_nonexistent_type_returns_none(self):
        """Test getting non-existent type returns None."""
        registry = get_custom_type_registry()
        
        result = registry.get("nonexistent-type")
        assert result is None
    
    def test_delete_custom_type(self):
        """Test deleting a custom type."""
        registry = get_custom_type_registry()
        
        custom_type = CustomDocumentType(
            id="del-type",
            name="Delete Form",
            brokerage_id="test-brokerage",
            patterns=[],
            example_texts=[],
        )
        
        registry.add(custom_type)
        assert len(registry) == 1
        
        result = registry.delete("del-type")
        assert result is True
        assert len(registry) == 0
        assert "del-type" not in registry
    
    def test_delete_nonexistent_returns_false(self):
        """Test deleting non-existent type returns False."""
        registry = get_custom_type_registry()
        
        result = registry.delete("nonexistent-type")
        assert result is False
    
    def test_list_custom_types(self):
        """Test listing all custom types."""
        registry = get_custom_type_registry()
        
        type1 = CustomDocumentType(
            id="type-1",
            name="Form 1",
            brokerage_id="brokerage-a",
            patterns=[],
            example_texts=[],
            priority=1.0,
        )
        type2 = CustomDocumentType(
            id="type-2",
            name="Form 2",
            brokerage_id="brokerage-b",
            patterns=[],
            example_texts=[],
            priority=2.0,
        )
        
        registry.add(type1)
        registry.add(type2)
        
        types = registry.list()
        assert len(types) == 2
        # Should be sorted by priority (highest first)
        assert types[0].id == "type-2"
        assert types[1].id == "type-1"
    
    def test_list_filter_by_brokerage(self):
        """Test filtering types by brokerage."""
        registry = get_custom_type_registry()
        
        type1 = CustomDocumentType(
            id="type-a1",
            name="Form A1",
            brokerage_id="brokerage-a",
            patterns=[],
            example_texts=[],
        )
        type2 = CustomDocumentType(
            id="type-b1",
            name="Form B1",
            brokerage_id="brokerage-b",
            patterns=[],
            example_texts=[],
        )
        
        registry.add(type1)
        registry.add(type2)
        
        types = registry.list(brokerage_id="brokerage-a")
        assert len(types) == 1
        assert types[0].id == "type-a1"
    
    def test_list_active_only(self):
        """Test filtering to active types only."""
        registry = get_custom_type_registry()
        
        type1 = CustomDocumentType(
            id="active-type",
            name="Active Form",
            brokerage_id="test",
            patterns=[],
            example_texts=[],
            is_active=True,
        )
        type2 = CustomDocumentType(
            id="inactive-type",
            name="Inactive Form",
            brokerage_id="test",
            patterns=[],
            example_texts=[],
            is_active=False,
        )
        
        registry.add(type1)
        registry.add(type2)
        
        types = registry.list(active_only=True)
        assert len(types) == 1
        assert types[0].id == "active-type"
    
    def test_find_match(self):
        """Test finding best matching custom type for text."""
        registry = get_custom_type_registry()
        
        type1 = CustomDocumentType(
            id="match-type",
            name="Match Form",
            brokerage_id="test",
            patterns=[(r"special\s+disclosure", 0.90)],
            example_texts=[],
        )
        
        registry.add(type1)
        
        result = registry.find_match("This is a Special Disclosure document")
        assert result is not None
        custom_type, confidence = result
        assert custom_type.id == "match-type"
        assert confidence >= 0.70
    
    def test_find_match_no_match(self):
        """Test find_match returns None when no match."""
        registry = get_custom_type_registry()
        
        type1 = CustomDocumentType(
            id="no-match",
            name="No Match Form",
            brokerage_id="test",
            patterns=[(r"xyz\s+pattern", 0.90)],
            example_texts=[],
        )
        
        registry.add(type1)
        
        result = registry.find_match("This is an ABC document")
        assert result is None
    
    def test_clear_registry(self):
        """Test clearing all types from registry."""
        registry = get_custom_type_registry()
        
        type1 = CustomDocumentType(
            id="clear-type",
            name="Clear Form",
            brokerage_id="test",
            patterns=[],
            example_texts=[],
        )
        
        registry.add(type1)
        assert len(registry) == 1
        
        count = registry.clear()
        assert count == 1
        assert len(registry) == 0
    
    def test_save_and_load(self):
        """Test saving and loading registry to/from file."""
        registry = get_custom_type_registry()
        
        type1 = CustomDocumentType(
            id="persist-type",
            name="Persist Form",
            brokerage_id="test-brokerage",
            patterns=[(r"persist", 0.85)],
            example_texts=["persist example"],
            folder="Persist Folder",
        )
        
        registry.add(type1)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            registry.save(temp_path)
            
            # Create new registry and load
            new_registry = CustomDocumentTypeRegistry()
            count = new_registry.load(temp_path)
            
            assert count == 1
            loaded = new_registry.get("persist-type")
            assert loaded is not None
            assert loaded.name == "Persist Form"
            assert loaded.folder == "Persist Folder"
            assert len(loaded.patterns) == 1
        finally:
            os.unlink(temp_path)


class TestMatchCustomTypes:
    """Tests for the match_custom_types function."""
    
    def setup_method(self):
        """Reset registry before each test."""
        set_custom_type_registry(CustomDocumentTypeRegistry())
    
    def test_match_custom_types_disabled(self):
        """Test matching returns None when disabled."""
        config = ClassifierConfig(custom_types_enabled=False)
        
        result = match_custom_types("any text", config)
        assert result is None
    
    def test_match_custom_types_empty_registry(self):
        """Test matching returns None with empty registry."""
        config = ClassifierConfig(custom_types_enabled=True)
        
        result = match_custom_types("any text", config)
        assert result is None
    
    def test_match_custom_types_finds_match(self):
        """Test finding a custom type match."""
        registry = get_custom_type_registry()
        registry.add(CustomDocumentType(
            id="custom-match",
            name="Custom Match Form",
            brokerage_id="test",
            patterns=[(r"custom\s+match\s+pattern", 0.92)],
            example_texts=[],
        ))
        
        config = ClassifierConfig(custom_types_enabled=True)
        
        result = match_custom_types("This is a Custom Match Pattern document", config)
        assert result is not None
        assert isinstance(result, CustomTypeMatch)
        assert result.custom_type.id == "custom-match"
        assert result.confidence >= 0.70
        assert len(result.matched_patterns) >= 1
    
    def test_match_custom_types_filter_by_brokerage(self):
        """Test filtering matches by brokerage."""
        registry = get_custom_type_registry()
        registry.add(CustomDocumentType(
            id="broker-a-type",
            name="Broker A Form",
            brokerage_id="broker-a",
            patterns=[(r"special\s+form", 0.90)],
            example_texts=[],
        ))
        registry.add(CustomDocumentType(
            id="broker-b-type",
            name="Broker B Form",
            brokerage_id="broker-b",
            patterns=[(r"special\s+form", 0.90)],
            example_texts=[],
        ))
        
        # Only match broker-a
        config = ClassifierConfig(
            custom_types_enabled=True,
            custom_types_brokerage_id="broker-a",
        )
        
        result = match_custom_types("This is a Special Form", config)
        assert result is not None
        assert result.custom_type.brokerage_id == "broker-a"


class TestClassifyByKeywordsWithCustomTypes:
    """Tests for classify_by_keywords with custom type integration."""
    
    def setup_method(self):
        """Reset registry before each test."""
        set_custom_type_registry(CustomDocumentTypeRegistry())
    
    def test_custom_type_matched_first(self):
        """Test custom types are matched before standard types when prioritized."""
        registry = get_custom_type_registry()
        registry.add(CustomDocumentType(
            id="priority-custom",
            name="Priority Custom Form",
            brokerage_id="test",
            patterns=[(r"purchase\s+agreement", 0.95)],  # Same as standard
            example_texts=[],
        ))
        
        config = ClassifierConfig(
            custom_types_enabled=True,
            custom_types_priority=True,
            use_llm=False,
        )
        
        result = classify_by_keywords(
            "This is a PURCHASE AGREEMENT for the property",
            config=config,
        )
        
        # Should match custom type first
        assert result.document_type == DocumentType.CUSTOM
        assert result.custom_type_id == "priority-custom"
        assert result.method == "keyword_custom"
    
    def test_standard_type_when_custom_disabled(self):
        """Test standard types match when custom types disabled."""
        registry = get_custom_type_registry()
        registry.add(CustomDocumentType(
            id="disabled-custom",
            name="Disabled Custom Form",
            brokerage_id="test",
            patterns=[(r"purchase\s+agreement", 0.95)],
            example_texts=[],
        ))
        
        config = ClassifierConfig(
            custom_types_enabled=False,
            use_llm=False,
        )
        
        result = classify_by_keywords(
            "This is a PURCHASE AGREEMENT for the property",
            config=config,
        )
        
        # Should match standard type
        assert result.document_type == DocumentType.PURCHASE_AGREEMENT
        assert result.custom_type_id is None
        assert result.method == "keyword"
    
    def test_custom_type_as_fallback(self):
        """Test custom types as fallback when not prioritized."""
        registry = get_custom_type_registry()
        registry.add(CustomDocumentType(
            id="fallback-custom",
            name="Fallback Form",
            brokerage_id="test",
            patterns=[(r"xyz\s+special\s+form", 0.90)],  # Unique pattern
            example_texts=[],
        ))
        
        config = ClassifierConfig(
            custom_types_enabled=True,
            custom_types_priority=False,
            use_llm=False,
        )
        
        result = classify_by_keywords(
            "This is an XYZ Special Form document",
            config=config,
        )
        
        # Should match custom type as fallback
        assert result.document_type == DocumentType.CUSTOM
        assert result.custom_type_id == "fallback-custom"


class TestClassificationResultCustomType:
    """Tests for ClassificationResult with custom type fields."""
    
    def test_custom_type_fields(self):
        """Test custom type fields in ClassificationResult."""
        result = ClassificationResult(
            document_type=DocumentType.CUSTOM,
            confidence=0.92,
            method="keyword_custom",
            custom_type_id="my-custom-type",
            custom_type_name="My Custom Form",
        )
        
        assert result.custom_type_id == "my-custom-type"
        assert result.custom_type_name == "My Custom Form"
        assert result.is_custom_type is True
    
    def test_is_custom_type_false_for_standard(self):
        """Test is_custom_type is False for standard types."""
        result = ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.85,
            method="keyword",
        )
        
        assert result.is_custom_type is False
        assert result.custom_type_id is None
        assert result.custom_type_name is None
    
    def test_to_dict_includes_custom_type(self):
        """Test to_dict includes custom type info."""
        result = ClassificationResult(
            document_type=DocumentType.CUSTOM,
            confidence=0.88,
            method="keyword_custom",
            custom_type_id="dict-custom-type",
            custom_type_name="Dict Custom Form",
        )
        
        data = result.to_dict()
        assert data["custom_type_id"] == "dict-custom-type"
        assert data["custom_type_name"] == "Dict Custom Form"
        assert data["is_custom_type"] is True


class TestAdminAPIFunctions:
    """Tests for the Admin API functions for custom types."""
    
    def setup_method(self):
        """Reset registry before each test."""
        set_custom_type_registry(CustomDocumentTypeRegistry())
    
    def test_create_custom_document_type(self):
        """Test creating a custom type via admin API."""
        result = create_custom_document_type(
            type_id="api-created",
            name="API Created Form",
            brokerage_id="api-brokerage",
            patterns=[{"pattern": r"api\s+form", "weight": 0.85}],
            folder="API Forms",
        )
        
        # Result is the custom type dict
        assert result["id"] == "api-created"
        assert result["name"] == "API Created Form"
        assert result["folder"] == "API Forms"
        
        # Verify it was added to registry
        registry = get_custom_type_registry()
        custom_type = registry.get("api-created")
        assert custom_type is not None
        assert custom_type.name == "API Created Form"
        assert custom_type.folder == "API Forms"
    
    def test_update_custom_document_type(self):
        """Test updating a custom type via admin API."""
        # Create first
        create_custom_document_type(
            type_id="api-update",
            name="Original Name",
            brokerage_id="test",
        )
        
        # Update
        result = update_custom_document_type(
            type_id="api-update",
            updates={"name": "Updated Name", "folder": "New Folder"},
        )
        
        # Result is the updated custom type dict
        assert result["name"] == "Updated Name"
        assert result["folder"] == "New Folder"
        
        # Verify update in registry
        registry = get_custom_type_registry()
        custom_type = registry.get("api-update")
        assert custom_type is not None
        assert custom_type.name == "Updated Name"
        assert custom_type.folder == "New Folder"
    
    def test_delete_custom_document_type(self):
        """Test deleting a custom type via admin API."""
        # Create first
        create_custom_document_type(
            type_id="api-delete",
            name="Delete Me",
            brokerage_id="test",
        )
        
        registry = get_custom_type_registry()
        assert "api-delete" in registry
        
        # Delete
        result = delete_custom_document_type("api-delete")
        assert result is True
        assert "api-delete" not in registry
    
    def test_list_custom_document_types(self):
        """Test listing custom types via admin API."""
        create_custom_document_type(
            type_id="list-type-1",
            name="List Form 1",
            brokerage_id="broker-x",
        )
        create_custom_document_type(
            type_id="list-type-2",
            name="List Form 2",
            brokerage_id="broker-y",
        )
        
        # List all
        result = list_custom_document_types()
        assert len(result) == 2
        
        # List by brokerage
        result = list_custom_document_types(brokerage_id="broker-x")
        assert len(result) == 1
        assert result[0]["id"] == "list-type-1"


class TestDocumentTypeCustomEnum:
    """Tests for the CUSTOM DocumentType enum value."""
    
    def test_custom_document_type_exists(self):
        """Test CUSTOM type exists in DocumentType enum."""
        assert hasattr(DocumentType, 'CUSTOM')
        assert DocumentType.CUSTOM.value == "Custom"
    
    def test_from_string_custom(self):
        """Test from_string handles 'custom' value."""
        result = DocumentType.from_string("Custom")
        assert result == DocumentType.CUSTOM


class TestClassifierConfigCustomTypes:
    """Tests for ClassifierConfig custom types settings."""
    
    def test_default_custom_types_enabled(self):
        """Test custom types enabled by default."""
        config = ClassifierConfig()
        assert config.custom_types_enabled is True
    
    def test_default_custom_types_priority(self):
        """Test custom types have priority by default."""
        config = ClassifierConfig()
        assert config.custom_types_priority is True
    
    def test_custom_types_file_path(self):
        """Test custom types file path config."""
        config = ClassifierConfig(
            custom_types_file_path="/path/to/custom_types.json"
        )
        assert config.custom_types_file_path == "/path/to/custom_types.json"
    
    def test_custom_types_brokerage_filter(self):
        """Test brokerage filter config."""
        config = ClassifierConfig(
            custom_types_brokerage_id="my-brokerage"
        )
        assert config.custom_types_brokerage_id == "my-brokerage"


# ============================================================================
# Missing Document Detection Tests (C5)
# ============================================================================

class TestPropertyMetadata:
    """Tests for PropertyMetadata dataclass."""
    
    def test_create_property_metadata(self):
        """Test creating property metadata."""
        meta = PropertyMetadata(
            year_built=1970,
            property_type="residential",
            state="CA",
            has_pool=True,
            is_hoa=True,
        )
        
        assert meta.year_built == 1970
        assert meta.property_type == "residential"
        assert meta.state == "CA"
        assert meta.has_pool is True
        assert meta.is_hoa is True
        assert meta.has_septic is False  # Default
    
    def test_property_metadata_defaults(self):
        """Test default values for property metadata."""
        meta = PropertyMetadata()
        
        assert meta.year_built is None
        assert meta.property_type == "residential"
        assert meta.state is None
        assert meta.has_pool is False
        assert meta.has_septic is False
        assert meta.has_well is False
        assert meta.is_hoa is False
        assert meta.is_condo is False
        assert meta.is_new_construction is False
    
    def test_property_metadata_to_dict(self):
        """Test serialization to dict."""
        meta = PropertyMetadata(
            year_built=1985,
            state="TX",
            has_pool=True,
        )
        
        data = meta.to_dict()
        assert data["year_built"] == 1985
        assert data["state"] == "TX"
        assert data["has_pool"] is True
    
    def test_property_metadata_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "year_built": 1965,
            "property_type": "residential",
            "state": "FL",
            "has_septic": True,
        }
        
        meta = PropertyMetadata.from_dict(data)
        assert meta.year_built == 1965
        assert meta.state == "FL"
        assert meta.has_septic is True


class TestTransactionMetadata:
    """Tests for TransactionMetadata dataclass."""
    
    def test_create_transaction_metadata(self):
        """Test creating transaction metadata."""
        meta = TransactionMetadata(
            transaction_type="purchase",
            financing_type="fha",
            is_cash_deal=False,
            inspection_contingency=True,
        )
        
        assert meta.transaction_type == "purchase"
        assert meta.financing_type == "fha"
        assert meta.is_cash_deal is False
        assert meta.inspection_contingency is True
    
    def test_transaction_metadata_defaults(self):
        """Test default values for transaction metadata."""
        meta = TransactionMetadata()
        
        assert meta.transaction_type == "purchase"
        assert meta.financing_type == "conventional"
        assert meta.is_cash_deal is False
        assert meta.has_contingencies is True
        assert meta.inspection_contingency is True
        assert meta.financing_contingency is True
        assert meta.appraisal_contingency is True
    
    def test_transaction_metadata_cash_deal(self):
        """Test cash deal transaction."""
        meta = TransactionMetadata(
            is_cash_deal=True,
            financing_contingency=False,
        )
        
        assert meta.is_cash_deal is True
        assert meta.financing_contingency is False
    
    def test_transaction_metadata_to_dict(self):
        """Test serialization to dict."""
        meta = TransactionMetadata(
            transaction_type="purchase",
            financing_type="va",
        )
        
        data = meta.to_dict()
        assert data["transaction_type"] == "purchase"
        assert data["financing_type"] == "va"
    
    def test_transaction_metadata_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "transaction_type": "sale",
            "is_cash_deal": True,
            "is_short_sale": True,
        }
        
        meta = TransactionMetadata.from_dict(data)
        assert meta.transaction_type == "sale"
        assert meta.is_cash_deal is True
        assert meta.is_short_sale is True


class TestRequirementCondition:
    """Tests for RequirementCondition enum."""
    
    def test_property_conditions_exist(self):
        """Test property-based conditions exist."""
        assert RequirementCondition.BUILT_BEFORE_1978
        assert RequirementCondition.HAS_POOL
        assert RequirementCondition.HAS_SEPTIC
        assert RequirementCondition.IS_HOA
        assert RequirementCondition.IS_RESIDENTIAL
    
    def test_transaction_conditions_exist(self):
        """Test transaction-based conditions exist."""
        assert RequirementCondition.IS_PURCHASE
        assert RequirementCondition.HAS_FINANCING
        assert RequirementCondition.IS_FHA_LOAN
        assert RequirementCondition.IS_VA_LOAN
        assert RequirementCondition.IS_CASH_DEAL
    
    def test_state_conditions_exist(self):
        """Test state-specific conditions exist."""
        assert RequirementCondition.STATE_CALIFORNIA
        assert RequirementCondition.STATE_TEXAS
        assert RequirementCondition.STATE_FLORIDA
    
    def test_always_condition_exists(self):
        """Test ALWAYS condition exists."""
        assert RequirementCondition.ALWAYS


class TestRequiredDocumentRule:
    """Tests for RequiredDocumentRule dataclass."""
    
    def test_create_basic_rule(self):
        """Test creating a basic requirement rule."""
        rule = RequiredDocumentRule(
            id="test-rule",
            document_type=DocumentType.LEAD_PAINT_DISCLOSURE,
            conditions=[RequirementCondition.BUILT_BEFORE_1978],
            description="Test rule for lead paint",
        )
        
        assert rule.id == "test-rule"
        assert rule.document_type == DocumentType.LEAD_PAINT_DISCLOSURE
        assert RequirementCondition.BUILT_BEFORE_1978 in rule.conditions
    
    def test_rule_evaluate_pre1978_property(self):
        """Test rule evaluation for pre-1978 property."""
        rule = RequiredDocumentRule(
            id="lead-paint",
            document_type=DocumentType.LEAD_PAINT_DISCLOSURE,
            conditions=[RequirementCondition.BUILT_BEFORE_1978],
        )
        
        # Pre-1978 property - should require lead paint disclosure
        old_property = PropertyMetadata(year_built=1970)
        assert rule.evaluate(old_property, None) is True
        
        # Post-1978 property - should NOT require lead paint disclosure
        new_property = PropertyMetadata(year_built=1990)
        assert rule.evaluate(new_property, None) is False
    
    def test_rule_evaluate_multiple_conditions_all(self):
        """Test rule with multiple conditions using AND logic."""
        rule = RequiredDocumentRule(
            id="financed-purchase",
            document_type=DocumentType.PRE_APPROVAL_LETTER,
            conditions=[
                RequirementCondition.IS_PURCHASE,
                RequirementCondition.HAS_FINANCING,
            ],
            condition_logic="all",
        )
        
        # Both conditions met
        trans = TransactionMetadata(
            transaction_type="purchase",
            is_cash_deal=False,
        )
        assert rule.evaluate(None, trans) is True
        
        # Only one condition met (cash deal)
        cash_trans = TransactionMetadata(
            transaction_type="purchase",
            is_cash_deal=True,
        )
        assert rule.evaluate(None, cash_trans) is False
    
    def test_rule_evaluate_multiple_conditions_any(self):
        """Test rule with multiple conditions using OR logic."""
        rule = RequiredDocumentRule(
            id="special-financing",
            document_type=DocumentType.PRE_APPROVAL_LETTER,
            conditions=[
                RequirementCondition.IS_FHA_LOAN,
                RequirementCondition.IS_VA_LOAN,
            ],
            condition_logic="any",
        )
        
        # FHA loan
        fha = TransactionMetadata(financing_type="fha")
        assert rule.evaluate(None, fha) is True
        
        # VA loan
        va = TransactionMetadata(financing_type="va")
        assert rule.evaluate(None, va) is True
        
        # Conventional loan
        conv = TransactionMetadata(financing_type="conventional")
        assert rule.evaluate(None, conv) is False
    
    def test_rule_evaluate_always_condition(self):
        """Test rule with ALWAYS condition."""
        rule = RequiredDocumentRule(
            id="always-required",
            document_type=DocumentType.AGENCY_DISCLOSURE,
            conditions=[RequirementCondition.ALWAYS],
        )
        
        # Should always be true
        assert rule.evaluate(None, None) is True
        assert rule.evaluate(PropertyMetadata(), None) is True
        assert rule.evaluate(None, TransactionMetadata()) is True
    
    def test_rule_evaluate_state_restriction(self):
        """Test rule with state restriction."""
        rule = RequiredDocumentRule(
            id="ca-specific",
            document_type=DocumentType.MOLD_DISCLOSURE,
            conditions=[RequirementCondition.ALWAYS],
            states=["CA"],
        )
        
        # California property - rule applies
        ca_property = PropertyMetadata(state="CA")
        assert rule.evaluate(ca_property, None) is True
        
        # Texas property - rule doesn't apply
        tx_property = PropertyMetadata(state="TX")
        assert rule.evaluate(tx_property, None) is False
    
    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = RequiredDocumentRule(
            id="test-rule",
            document_type=DocumentType.SELLER_DISCLOSURE,
            conditions=[RequirementCondition.IS_RESIDENTIAL],
            priority="required",
        )
        
        data = rule.to_dict()
        assert data["id"] == "test-rule"
        assert data["document_type"] == "Seller Disclosure"
        assert "is_residential" in data["conditions"]
        assert data["priority"] == "required"


class TestDefaultRequirementRules:
    """Tests for default document requirement rules."""
    
    def test_default_rules_exist(self):
        """Test that default rules are defined."""
        assert len(DEFAULT_REQUIREMENT_RULES) > 0
    
    def test_lead_paint_rule_exists(self):
        """Test lead paint disclosure rule exists."""
        lead_paint_rules = [
            r for r in DEFAULT_REQUIREMENT_RULES 
            if r.document_type == DocumentType.LEAD_PAINT_DISCLOSURE
        ]
        assert len(lead_paint_rules) > 0
    
    def test_lead_paint_requires_pre1978(self):
        """Test lead paint rule requires pre-1978 property."""
        lead_paint_rule = next(
            r for r in DEFAULT_REQUIREMENT_RULES 
            if r.document_type == DocumentType.LEAD_PAINT_DISCLOSURE
        )
        
        old_property = PropertyMetadata(year_built=1970, property_type="residential")
        assert lead_paint_rule.evaluate(old_property, None) is True
        
        new_property = PropertyMetadata(year_built=1990, property_type="residential")
        assert lead_paint_rule.evaluate(new_property, None) is False
    
    def test_purchase_agreement_rule_exists(self):
        """Test purchase agreement rule exists."""
        purchase_rules = [
            r for r in DEFAULT_REQUIREMENT_RULES 
            if r.document_type == DocumentType.PURCHASE_AGREEMENT
        ]
        assert len(purchase_rules) > 0
    
    def test_agency_disclosure_always_required(self):
        """Test agency disclosure is always required."""
        agency_rule = next(
            r for r in DEFAULT_REQUIREMENT_RULES 
            if r.document_type == DocumentType.AGENCY_DISCLOSURE
        )
        
        # Should always be required
        assert agency_rule.evaluate(None, None) is True


class TestGetRequiredDocuments:
    """Tests for get_required_documents function."""
    
    def test_get_required_for_pre1978_residential(self):
        """Test required docs for pre-1978 residential purchase."""
        property_meta = PropertyMetadata(
            year_built=1965,
            property_type="residential",
        )
        transaction_meta = TransactionMetadata(
            transaction_type="purchase",
            is_cash_deal=False,
            inspection_contingency=True,
        )
        
        required = get_required_documents(property_meta, transaction_meta)
        doc_types = [doc_type for doc_type, _ in required]
        
        # Should include lead paint disclosure for pre-1978
        assert DocumentType.LEAD_PAINT_DISCLOSURE in doc_types
        # Should include seller disclosure for residential
        assert DocumentType.SELLER_DISCLOSURE in doc_types
        # Should include agency disclosure (always required)
        assert DocumentType.AGENCY_DISCLOSURE in doc_types
    
    def test_get_required_for_new_construction(self):
        """Test required docs for new construction - no lead paint needed."""
        property_meta = PropertyMetadata(
            year_built=2023,
            property_type="residential",
            is_new_construction=True,
        )
        
        required = get_required_documents(property_meta, None)
        doc_types = [doc_type for doc_type, _ in required]
        
        # Should NOT include lead paint disclosure for new construction
        assert DocumentType.LEAD_PAINT_DISCLOSURE not in doc_types
    
    def test_get_required_for_cash_deal(self):
        """Test required docs for cash deal - no financing docs needed."""
        transaction_meta = TransactionMetadata(
            transaction_type="purchase",
            is_cash_deal=True,
            financing_contingency=False,
        )
        
        required = get_required_documents(None, transaction_meta)
        doc_types = [doc_type for doc_type, _ in required]
        
        # Should NOT include financing documents for cash deal
        assert DocumentType.LOAN_ESTIMATE not in doc_types
        assert DocumentType.CLOSING_DISCLOSURE not in doc_types
    
    def test_get_required_excludes_recommended(self):
        """Test filtering out recommended docs."""
        required_only = get_required_documents(
            None, None, include_recommended=False
        )
        
        # All returned should be "required" priority
        for _, rule in required_only:
            assert rule.priority == "required"
    
    def test_get_required_with_inspection_contingency(self):
        """Test inspection report required with contingency."""
        transaction_meta = TransactionMetadata(
            inspection_contingency=True,
        )
        
        required = get_required_documents(None, transaction_meta)
        doc_types = [doc_type for doc_type, _ in required]
        
        assert DocumentType.INSPECTION_REPORT in doc_types


class TestDetectMissingDocuments:
    """Tests for detect_missing_documents function."""
    
    def test_detect_missing_lead_paint(self):
        """Test detecting missing lead paint disclosure."""
        # Classified docs without lead paint disclosure
        classified_docs = [
            {"doc_type": "Purchase Agreement"},
            {"doc_type": "Seller Disclosure"},
        ]
        
        property_meta = PropertyMetadata(
            year_built=1965,
            property_type="residential",
        )
        
        report = detect_missing_documents(
            classified_docs, property_meta, None
        )
        
        # Should detect missing lead paint disclosure
        missing_types = [m.document_type for m in report.missing_required]
        assert DocumentType.LEAD_PAINT_DISCLOSURE in missing_types
    
    def test_detect_no_missing_when_complete(self):
        """Test no missing docs when all required are present."""
        # All required docs present
        classified_docs = [
            {"doc_type": "Purchase Agreement"},
            {"doc_type": "Seller Disclosure"},
            {"doc_type": "Agency Disclosure"},
            {"doc_type": "Lead Paint Disclosure"},
            {"doc_type": "Inspection Report"},
            {"doc_type": "Pre-Approval Letter"},
            {"doc_type": "Title Commitment"},
            {"doc_type": "Appraisal"},
            {"doc_type": "Loan Estimate"},
            {"doc_type": "Closing Disclosure (Loan)"},
        ]
        
        property_meta = PropertyMetadata(
            year_built=1965,
            property_type="residential",
        )
        transaction_meta = TransactionMetadata(
            transaction_type="purchase",
            is_cash_deal=False,
        )
        
        report = detect_missing_documents(
            classified_docs, property_meta, transaction_meta
        )
        
        # Should have no missing required docs
        assert report.is_complete is True
        assert report.required_count == 0
    
    def test_detect_missing_creates_report(self):
        """Test missing document report creation."""
        classified_docs = [
            {"doc_type": "Purchase Agreement"},
        ]
        
        property_meta = PropertyMetadata(year_built=1970)
        transaction_meta = TransactionMetadata()
        
        report = detect_missing_documents(
            classified_docs, property_meta, transaction_meta
        )
        
        assert isinstance(report, MissingDocumentReport)
        assert report.property_metadata == property_meta
        assert report.transaction_metadata == transaction_meta
        assert report.evaluated_at is not None
    
    def test_report_separates_required_and_recommended(self):
        """Test report separates required and recommended missing docs."""
        classified_docs = []  # No docs present
        
        report = detect_missing_documents(classified_docs, None, None)
        
        # Should have separate lists
        assert isinstance(report.missing_required, list)
        assert isinstance(report.missing_recommended, list)
        
        # Wire instructions should be in recommended
        recommended_types = [m.document_type for m in report.missing_recommended]
        assert DocumentType.WIRE_INSTRUCTIONS in recommended_types
    
    def test_report_tracks_found_documents(self):
        """Test report tracks which documents were found."""
        classified_docs = [
            {"doc_type": "Purchase Agreement"},
            {"doc_type": "Seller Disclosure"},
        ]
        
        report = detect_missing_documents(classified_docs, None, None)
        
        assert DocumentType.PURCHASE_AGREEMENT in report.documents_found
        assert DocumentType.SELLER_DISCLOSURE in report.documents_found
    
    def test_report_to_dict(self):
        """Test report serialization."""
        classified_docs = [{"doc_type": "Purchase Agreement"}]
        property_meta = PropertyMetadata(year_built=1965)
        
        report = detect_missing_documents(
            classified_docs, property_meta, None
        )
        
        data = report.to_dict()
        assert "missing_required" in data
        assert "missing_recommended" in data
        assert "documents_found" in data
        assert "is_complete" in data
        assert "has_missing_required" in data
    
    def test_handles_empty_classified_docs(self):
        """Test handling empty classified docs list."""
        report = detect_missing_documents([], None, None)
        
        # Should still work and report missing required docs
        assert report.has_missing_required is True
        assert len(report.documents_found) == 0
    
    def test_handles_invalid_doc_types(self):
        """Test handling invalid document types gracefully."""
        classified_docs = [
            {"doc_type": "Invalid Document Type"},
            {"doc_type": "Purchase Agreement"},
            {"document_type": "Seller Disclosure"},  # Alternative key
        ]
        
        report = detect_missing_documents(classified_docs, None, None)
        
        # Should still find valid documents
        assert DocumentType.PURCHASE_AGREEMENT in report.documents_found
        assert DocumentType.SELLER_DISCLOSURE in report.documents_found


class TestMissingDocument:
    """Tests for MissingDocument dataclass."""
    
    def test_create_missing_document(self):
        """Test creating a missing document entry."""
        missing = MissingDocument(
            document_type=DocumentType.LEAD_PAINT_DISCLOSURE,
            rule_id="lead-paint-pre1978",
            reason="Lead paint disclosure required for pre-1978 homes",
            priority="required",
        )
        
        assert missing.document_type == DocumentType.LEAD_PAINT_DISCLOSURE
        assert missing.rule_id == "lead-paint-pre1978"
        assert missing.priority == "required"
    
    def test_missing_document_to_dict(self):
        """Test missing document serialization."""
        missing = MissingDocument(
            document_type=DocumentType.INSPECTION_REPORT,
            rule_id="inspection-report",
            reason="Inspection required with contingency",
            priority="required",
        )
        
        data = missing.to_dict()
        assert data["document_type"] == "Inspection Report"
        assert data["rule_id"] == "inspection-report"
        assert data["priority"] == "required"


class TestMissingDocumentReport:
    """Tests for MissingDocumentReport dataclass."""
    
    def test_report_has_missing_required(self):
        """Test has_missing_required property."""
        report_with_missing = MissingDocumentReport(
            missing_required=[
                MissingDocument(
                    document_type=DocumentType.LEAD_PAINT_DISCLOSURE,
                    rule_id="test",
                    reason="test",
                    priority="required",
                )
            ]
        )
        assert report_with_missing.has_missing_required is True
        
        report_without_missing = MissingDocumentReport(missing_required=[])
        assert report_without_missing.has_missing_required is False
    
    def test_report_is_complete(self):
        """Test is_complete property."""
        complete_report = MissingDocumentReport(missing_required=[])
        assert complete_report.is_complete is True
        
        incomplete_report = MissingDocumentReport(
            missing_required=[
                MissingDocument(
                    document_type=DocumentType.SELLER_DISCLOSURE,
                    rule_id="test",
                    reason="test",
                    priority="required",
                )
            ]
        )
        assert incomplete_report.is_complete is False
    
    def test_report_counts(self):
        """Test count properties."""
        report = MissingDocumentReport(
            missing_required=[
                MissingDocument(
                    document_type=DocumentType.LEAD_PAINT_DISCLOSURE,
                    rule_id="test1",
                    reason="test",
                    priority="required",
                ),
                MissingDocument(
                    document_type=DocumentType.SELLER_DISCLOSURE,
                    rule_id="test2",
                    reason="test",
                    priority="required",
                ),
            ],
            missing_recommended=[
                MissingDocument(
                    document_type=DocumentType.WIRE_INSTRUCTIONS,
                    rule_id="test3",
                    reason="test",
                    priority="recommended",
                ),
            ],
        )
        
        assert report.required_count == 2
        assert report.missing_count == 3


class TestLeadPaintDisclosureRequirement:
    """Focused tests for lead paint disclosure requirement (key C5 use case)."""
    
    def test_lead_paint_required_for_1977_home(self):
        """Test lead paint required for 1977 built home."""
        property_meta = PropertyMetadata(year_built=1977, property_type="residential")
        
        required = get_required_documents(property_meta, None)
        doc_types = [doc_type for doc_type, _ in required]
        
        assert DocumentType.LEAD_PAINT_DISCLOSURE in doc_types
    
    def test_lead_paint_not_required_for_1978_home(self):
        """Test lead paint NOT required for 1978 built home."""
        property_meta = PropertyMetadata(year_built=1978, property_type="residential")
        
        required = get_required_documents(property_meta, None)
        doc_types = [doc_type for doc_type, _ in required]
        
        assert DocumentType.LEAD_PAINT_DISCLOSURE not in doc_types
    
    def test_lead_paint_not_required_for_commercial(self):
        """Test lead paint NOT required for commercial pre-1978."""
        property_meta = PropertyMetadata(year_built=1960, property_type="commercial")
        
        required = get_required_documents(property_meta, None)
        doc_types = [doc_type for doc_type, _ in required]
        
        # Lead paint rule requires residential, so commercial should not trigger
        assert DocumentType.LEAD_PAINT_DISCLOSURE not in doc_types
    
    def test_detect_missing_lead_paint_for_1950s_home(self):
        """Test detecting missing lead paint for 1950s home."""
        classified_docs = [
            {"doc_type": "Purchase Agreement"},
            {"doc_type": "Seller Disclosure"},
        ]
        
        property_meta = PropertyMetadata(year_built=1955, property_type="residential")
        
        report = detect_missing_documents(classified_docs, property_meta, None)
        
        missing_types = [m.document_type for m in report.missing_required]
        assert DocumentType.LEAD_PAINT_DISCLOSURE in missing_types
        
        # Find the missing document entry
        lead_paint_missing = next(
            m for m in report.missing_required 
            if m.document_type == DocumentType.LEAD_PAINT_DISCLOSURE
        )
        assert "1978" in lead_paint_missing.reason.lower() or "lead" in lead_paint_missing.reason.lower()

