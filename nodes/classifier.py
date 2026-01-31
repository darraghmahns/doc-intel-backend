"""
Document Classifier Node - Automatic Document Type Classification

User Story C1:
As a user, I want documents automatically classified by type (Buy-Sell, 
Disclosure, Addendum) so the right extraction logic runs.

Acceptance Criteria:
- LLM/ML classifier with >90% accuracy on common document types
- Fallback keyword-based classification when LLM unavailable
- Confidence scores for classification decisions
- Support for common real estate document types

User Story C2:
As a user, I want scanned/image PDFs converted to searchable text via OCR
so handwritten contracts can be processed.

Acceptance Criteria:
- OCR runs on image-based pages, text extracted with >95% accuracy
- Classifier is OCR-aware and handles low-quality OCR text
- Re-OCR capability for documents below accuracy target
- OCR quality metrics tracked and reported

User Story C3:
As a user, I want low-confidence classifications flagged for human review
so misclassified documents don't cause errors.

Acceptance Criteria:
- Confidence threshold (e.g., <80%) triggers review queue
- Clear review reasons provided for flagged documents
- Review queue tracked in state and metrics

User Story C4:
As an admin, I want to add custom document types for my brokerage
so proprietary forms are recognized.

Acceptance Criteria:
- Admin API to add document types with example text/patterns
- Custom types take priority over built-in types
- Custom types can map to standard Dotloop folders
- CRUD operations for managing custom types

User Story C5:
As a user, I want the classifier to detect missing required documents
(e.g., no Lead Paint Disclosure for pre-1978 homes).

Acceptance Criteria:
- Missing document detection based on property/transaction metadata
- Rules engine for conditional document requirements
- Clear reporting of missing required documents
- Support for state/jurisdiction-specific requirements

Uses LLM (OpenAI/Anthropic) for intelligent classification with 
keyword-based fallback for reliability. Integrates with splitter's
OCR infrastructure for handling scanned documents.
"""

import os
import re
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from state import DealState

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# OCR Quality Constants
# ============================================================================

# OCR confidence threshold for acceptable quality (95% per C2 acceptance criteria)
OCR_ACCURACY_TARGET = 0.95

# Threshold below which OCR text may need preprocessing or re-OCR
OCR_LOW_CONFIDENCE_THRESHOLD = 0.70

# Threshold below which OCR quality is considered poor
OCR_POOR_QUALITY_THRESHOLD = 0.50


# ============================================================================
# Review Queue Constants (C3)
# ============================================================================

# Default confidence threshold below which documents are flagged for review
REVIEW_CONFIDENCE_THRESHOLD = 0.80

# Reasons for flagging documents for review
class ReviewReason(Enum):
    """Reasons why a document was flagged for human review."""
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN_TYPE = "unknown_type"
    AMBIGUOUS_CLASSIFICATION = "ambiguous_classification"
    OCR_QUALITY_ISSUE = "ocr_quality_issue"
    MULTIPLE_CLOSE_ALTERNATIVES = "multiple_close_alternatives"
    TITLE_ONLY_CLASSIFICATION = "title_only_classification"


# ============================================================================
# Document Types
# ============================================================================

class DocumentType(Enum):
    """
    Standard real estate document types.
    
    These map to Dotloop folder categories and extraction logic.
    """
    # Core transaction documents
    BUY_SELL_AGREEMENT = "Buy-Sell Agreement"
    PURCHASE_AGREEMENT = "Purchase Agreement"
    LISTING_AGREEMENT = "Listing Agreement"
    
    # Offer-related
    COUNTER_OFFER = "Counter Offer"
    AMENDMENT = "Amendment"
    ADDENDUM = "Addendum"
    
    # Disclosures
    SELLER_DISCLOSURE = "Seller Disclosure"
    PROPERTY_DISCLOSURE = "Property Disclosure"
    LEAD_PAINT_DISCLOSURE = "Lead Paint Disclosure"
    MOLD_DISCLOSURE = "Mold Disclosure"
    
    # Inspection & Due Diligence
    INSPECTION_REPORT = "Inspection Report"
    INSPECTION_RESPONSE = "Inspection Response"
    APPRAISAL = "Appraisal"
    
    # Closing documents
    CLOSING_STATEMENT = "Closing Statement"
    TITLE_COMMITMENT = "Title Commitment"
    DEED = "Deed"
    
    # Financing
    LOAN_ESTIMATE = "Loan Estimate"
    CLOSING_DISCLOSURE = "Closing Disclosure (Loan)"
    PRE_APPROVAL_LETTER = "Pre-Approval Letter"
    
    # Agency
    AGENCY_DISCLOSURE = "Agency Disclosure"
    BUYER_AGENCY_AGREEMENT = "Buyer Agency Agreement"
    
    # Other
    WIRE_INSTRUCTIONS = "Wire Instructions"
    EARNEST_MONEY_RECEIPT = "Earnest Money Receipt"
    UNKNOWN = "Unknown"
    
    # Custom (C4) - for brokerage-specific document types
    CUSTOM = "Custom"
    
    @classmethod
    def from_string(cls, value: str) -> "DocumentType":
        """Convert string to DocumentType, case-insensitive."""
        value_lower = value.lower().strip()
        
        # Direct match
        for doc_type in cls:
            if doc_type.value.lower() == value_lower:
                return doc_type
        
        # Fuzzy matching for common variations
        if "buy" in value_lower and "sell" in value_lower:
            return cls.BUY_SELL_AGREEMENT
        if "purchase" in value_lower and "agreement" in value_lower:
            return cls.PURCHASE_AGREEMENT
        if "counter" in value_lower and "offer" in value_lower:
            return cls.COUNTER_OFFER
        if "addendum" in value_lower:
            return cls.ADDENDUM
        if "amendment" in value_lower:
            return cls.AMENDMENT
        if "seller" in value_lower and "disclosure" in value_lower:
            return cls.SELLER_DISCLOSURE
        if "lead" in value_lower and "paint" in value_lower:
            return cls.LEAD_PAINT_DISCLOSURE
        if "mold" in value_lower:
            return cls.MOLD_DISCLOSURE
        if "inspection" in value_lower:
            if "response" in value_lower:
                return cls.INSPECTION_RESPONSE
            return cls.INSPECTION_REPORT
        if "closing" in value_lower:
            if "disclosure" in value_lower:
                return cls.CLOSING_DISCLOSURE
            if "statement" in value_lower:
                return cls.CLOSING_STATEMENT
        if "title" in value_lower:
            return cls.TITLE_COMMITMENT
        if "agency" in value_lower:
            if "buyer" in value_lower:
                return cls.BUYER_AGENCY_AGREEMENT
            return cls.AGENCY_DISCLOSURE
        
        return cls.UNKNOWN


# Mapping to Dotloop folders
DOCUMENT_TYPE_TO_FOLDER: Dict[DocumentType, str] = {
    DocumentType.BUY_SELL_AGREEMENT: "Contracts",
    DocumentType.PURCHASE_AGREEMENT: "Contracts",
    DocumentType.LISTING_AGREEMENT: "Contracts",
    DocumentType.COUNTER_OFFER: "Contracts",
    DocumentType.AMENDMENT: "Contracts",
    DocumentType.ADDENDUM: "Addenda",
    DocumentType.SELLER_DISCLOSURE: "Disclosures",
    DocumentType.PROPERTY_DISCLOSURE: "Disclosures",
    DocumentType.LEAD_PAINT_DISCLOSURE: "Disclosures",
    DocumentType.MOLD_DISCLOSURE: "Disclosures",
    DocumentType.INSPECTION_REPORT: "Inspections",
    DocumentType.INSPECTION_RESPONSE: "Inspections",
    DocumentType.APPRAISAL: "Inspections",
    DocumentType.CLOSING_STATEMENT: "Closing",
    DocumentType.TITLE_COMMITMENT: "Title",
    DocumentType.DEED: "Closing",
    DocumentType.LOAN_ESTIMATE: "Financing",
    DocumentType.CLOSING_DISCLOSURE: "Financing",
    DocumentType.PRE_APPROVAL_LETTER: "Financing",
    DocumentType.AGENCY_DISCLOSURE: "Disclosures",
    DocumentType.BUYER_AGENCY_AGREEMENT: "Contracts",
    DocumentType.WIRE_INSTRUCTIONS: "Closing",
    DocumentType.EARNEST_MONEY_RECEIPT: "Contracts",
    DocumentType.UNKNOWN: "Other",
}


# ============================================================================
# Custom Document Types (C4)
# ============================================================================

@dataclass
class CustomDocumentType:
    """
    Custom document type defined by admin for brokerage-specific forms (C4).
    
    Allows brokerages to add proprietary document types with custom
    patterns for recognition.
    """
    # Unique identifier
    id: str
    
    # Display name for the document type
    name: str
    
    # Brokerage/organization this type belongs to
    brokerage_id: str
    
    # Regex patterns for matching (list of (pattern, weight) tuples)
    patterns: List[Tuple[str, float]] = field(default_factory=list)
    
    # Example text snippets that characterize this document type
    example_texts: List[str] = field(default_factory=list)
    
    # Target folder in Dotloop/DocuSign
    folder: str = "Other"
    
    # Description for admin reference
    description: str = ""
    
    # Whether this type is active
    is_active: bool = True
    
    # Priority boost (higher = checked first, default 1.0)
    priority: float = 1.0
    
    # Minimum confidence to accept match
    min_confidence: float = 0.70
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: Optional[str] = None
    
    def matches(self, text: str) -> Tuple[bool, float]:
        """
        Check if text matches this custom document type.
        
        Args:
            text: Document text to check
            
        Returns:
            Tuple of (matches, confidence)
        """
        if not self.is_active or not text:
            return False, 0.0
        
        text_lower = text.lower()
        max_score = 0.0
        
        # Check patterns
        for pattern, weight in self.patterns:
            try:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    max_score = max(max_score, weight)
            except re.error:
                # Invalid regex pattern, skip
                logger.warning(f"Invalid pattern in custom type {self.id}: {pattern}")
                continue
        
        # Check example texts (simple substring matching)
        for example in self.example_texts:
            if example.lower() in text_lower:
                # Example text match gets 0.85 confidence
                max_score = max(max_score, 0.85)
        
        # Apply priority boost
        adjusted_score = min(max_score * self.priority, 1.0)
        
        return adjusted_score >= self.min_confidence, adjusted_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "brokerage_id": self.brokerage_id,
            "patterns": [{"pattern": p, "weight": w} for p, w in self.patterns],
            "example_texts": self.example_texts,
            "folder": self.folder,
            "description": self.description,
            "is_active": self.is_active,
            "priority": self.priority,
            "min_confidence": self.min_confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomDocumentType":
        """Create from dictionary."""
        patterns = [
            (p["pattern"], p["weight"]) 
            for p in data.get("patterns", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            brokerage_id=data["brokerage_id"],
            patterns=patterns,
            example_texts=data.get("example_texts", []),
            folder=data.get("folder", "Other"),
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
            priority=data.get("priority", 1.0),
            min_confidence=data.get("min_confidence", 0.70),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            created_by=data.get("created_by"),
        )


class CustomDocumentTypeRegistry:
    """
    Registry for managing custom document types (C4).
    
    Provides CRUD operations for custom types and integrates with
    the classifier. Supports persistence to JSON file.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the registry.
        
        Args:
            storage_path: Optional path to JSON file for persistence
        """
        self._types: Dict[str, CustomDocumentType] = {}
        self._storage_path = storage_path
        
        # Load from storage if available
        if storage_path and os.path.exists(storage_path):
            self.load()
    
    def add(self, custom_type: CustomDocumentType) -> CustomDocumentType:
        """
        Add a new custom document type.
        
        Args:
            custom_type: The custom type to add
            
        Returns:
            The added custom type
            
        Raises:
            ValueError: If a type with the same ID already exists
        """
        if custom_type.id in self._types:
            raise ValueError(f"Custom type with ID '{custom_type.id}' already exists")
        
        self._types[custom_type.id] = custom_type
        self._save_if_configured()
        
        logger.info(f"Added custom document type: {custom_type.name} ({custom_type.id})")
        return custom_type
    
    def update(self, type_id: str, updates: Dict[str, Any]) -> CustomDocumentType:
        """
        Update an existing custom document type.
        
        Args:
            type_id: ID of the type to update
            updates: Dictionary of fields to update
            
        Returns:
            The updated custom type
            
        Raises:
            KeyError: If the type doesn't exist
        """
        if type_id not in self._types:
            raise KeyError(f"Custom type with ID '{type_id}' not found")
        
        existing = self._types[type_id]
        
        # Update allowed fields
        allowed_fields = {
            'name', 'patterns', 'example_texts', 'folder', 'description',
            'is_active', 'priority', 'min_confidence'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                if field == 'patterns':
                    # Convert pattern dicts to tuples
                    value = [(p["pattern"], p["weight"]) for p in value]
                setattr(existing, field, value)
        
        existing.updated_at = datetime.now().isoformat()
        self._save_if_configured()
        
        logger.info(f"Updated custom document type: {existing.name} ({type_id})")
        return existing
    
    def delete(self, type_id: str) -> bool:
        """
        Delete a custom document type.
        
        Args:
            type_id: ID of the type to delete
            
        Returns:
            True if deleted, False if not found
        """
        if type_id not in self._types:
            return False
        
        deleted = self._types.pop(type_id)
        self._save_if_configured()
        
        logger.info(f"Deleted custom document type: {deleted.name} ({type_id})")
        return True
    
    def get(self, type_id: str) -> Optional[CustomDocumentType]:
        """
        Get a custom document type by ID.
        
        Args:
            type_id: ID of the type to get
            
        Returns:
            The custom type or None if not found
        """
        return self._types.get(type_id)
    
    def list(
        self, 
        brokerage_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[CustomDocumentType]:
        """
        List custom document types.
        
        Args:
            brokerage_id: Optional filter by brokerage
            active_only: If True, only return active types
            
        Returns:
            List of matching custom types
        """
        types = list(self._types.values())
        
        if brokerage_id:
            types = [t for t in types if t.brokerage_id == brokerage_id]
        
        if active_only:
            types = [t for t in types if t.is_active]
        
        # Sort by priority (highest first)
        types.sort(key=lambda t: t.priority, reverse=True)
        
        return types
    
    def find_match(
        self, 
        text: str, 
        brokerage_id: Optional[str] = None,
    ) -> Optional[Tuple[CustomDocumentType, float]]:
        """
        Find the best matching custom document type for text.
        
        Args:
            text: Document text to match
            brokerage_id: Optional filter by brokerage
            
        Returns:
            Tuple of (best_match, confidence) or None if no match
        """
        best_match: Optional[CustomDocumentType] = None
        best_confidence = 0.0
        
        for custom_type in self.list(brokerage_id=brokerage_id, active_only=True):
            matches, confidence = custom_type.matches(text)
            if matches and confidence > best_confidence:
                best_match = custom_type
                best_confidence = confidence
        
        if best_match:
            return best_match, best_confidence
        return None
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save registry to JSON file.
        
        Args:
            path: Optional path override
        """
        save_path = path or self._storage_path
        if not save_path:
            raise ValueError("No storage path configured")
        
        data = {
            "version": "1.0",
            "custom_types": [t.to_dict() for t in self._types.values()],
            "saved_at": datetime.now().isoformat(),
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self._types)} custom types to {save_path}")
    
    def load(self, path: Optional[str] = None) -> int:
        """
        Load registry from JSON file.
        
        Args:
            path: Optional path override
            
        Returns:
            Number of types loaded
        """
        load_path = path or self._storage_path
        if not load_path:
            raise ValueError("No storage path configured")
        
        if not os.path.exists(load_path):
            logger.warning(f"Storage file not found: {load_path}")
            return 0
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self._types.clear()
        for type_data in data.get("custom_types", []):
            custom_type = CustomDocumentType.from_dict(type_data)
            self._types[custom_type.id] = custom_type
        
        logger.info(f"Loaded {len(self._types)} custom types from {load_path}")
        return len(self._types)
    
    def _save_if_configured(self) -> None:
        """Save to storage if a path is configured."""
        if self._storage_path:
            self.save()
    
    def clear(self) -> int:
        """
        Clear all custom types.
        
        Returns:
            Number of types cleared
        """
        count = len(self._types)
        self._types.clear()
        self._save_if_configured()
        return count
    
    def __len__(self) -> int:
        return len(self._types)
    
    def __contains__(self, type_id: str) -> bool:
        return type_id in self._types


# Global registry instance (can be replaced with custom instance)
_custom_type_registry: Optional[CustomDocumentTypeRegistry] = None


def get_custom_type_registry() -> CustomDocumentTypeRegistry:
    """
    Get the global custom type registry.
    
    Creates a new registry if one doesn't exist.
    
    Returns:
        The global CustomDocumentTypeRegistry
    """
    global _custom_type_registry
    if _custom_type_registry is None:
        # Check for storage path from environment
        storage_path = os.getenv("CUSTOM_TYPES_STORAGE_PATH")
        _custom_type_registry = CustomDocumentTypeRegistry(storage_path)
    return _custom_type_registry


def set_custom_type_registry(registry: CustomDocumentTypeRegistry) -> None:
    """
    Set the global custom type registry.
    
    Args:
        registry: The registry to use globally
    """
    global _custom_type_registry
    _custom_type_registry = registry


# ============================================================================
# Admin API Functions (C4)
# ============================================================================

def create_custom_document_type(
    type_id: str,
    name: str,
    brokerage_id: str,
    patterns: Optional[List[Dict[str, Any]]] = None,
    example_texts: Optional[List[str]] = None,
    folder: str = "Other",
    description: str = "",
    priority: float = 1.0,
    min_confidence: float = 0.70,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new custom document type (Admin API).
    
    Args:
        type_id: Unique identifier for the type
        name: Display name
        brokerage_id: Brokerage/organization ID
        patterns: List of {"pattern": str, "weight": float} dicts
        example_texts: List of example text snippets
        folder: Target folder in Dotloop
        description: Admin description
        priority: Priority boost (default 1.0)
        min_confidence: Minimum confidence threshold
        created_by: Admin user ID
        
    Returns:
        Dict representation of created type
        
    Raises:
        ValueError: If type with ID already exists
    """
    registry = get_custom_type_registry()
    
    pattern_tuples = [
        (p["pattern"], p.get("weight", 0.90)) 
        for p in (patterns or [])
    ]
    
    custom_type = CustomDocumentType(
        id=type_id,
        name=name,
        brokerage_id=brokerage_id,
        patterns=pattern_tuples,
        example_texts=example_texts or [],
        folder=folder,
        description=description,
        priority=priority,
        min_confidence=min_confidence,
        created_by=created_by,
    )
    
    registry.add(custom_type)
    return custom_type.to_dict()


def update_custom_document_type(
    type_id: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update an existing custom document type (Admin API).
    
    Args:
        type_id: ID of the type to update
        updates: Dictionary of fields to update
        
    Returns:
        Dict representation of updated type
        
    Raises:
        KeyError: If type doesn't exist
    """
    registry = get_custom_type_registry()
    updated = registry.update(type_id, updates)
    return updated.to_dict()


def delete_custom_document_type(type_id: str) -> bool:
    """
    Delete a custom document type (Admin API).
    
    Args:
        type_id: ID of the type to delete
        
    Returns:
        True if deleted, False if not found
    """
    registry = get_custom_type_registry()
    return registry.delete(type_id)


def get_custom_document_type(type_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a custom document type by ID (Admin API).
    
    Args:
        type_id: ID of the type to get
        
    Returns:
        Dict representation or None if not found
    """
    registry = get_custom_type_registry()
    custom_type = registry.get(type_id)
    return custom_type.to_dict() if custom_type else None


def list_custom_document_types(
    brokerage_id: Optional[str] = None,
    active_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    List custom document types (Admin API).
    
    Args:
        brokerage_id: Optional filter by brokerage
        active_only: If True, only return active types
        
    Returns:
        List of dict representations
    """
    registry = get_custom_type_registry()
    types = registry.list(brokerage_id=brokerage_id, active_only=active_only)
    return [t.to_dict() for t in types]


# ============================================================================
# Missing Document Detection (C5)
# ============================================================================

@dataclass
class PropertyMetadata:
    """
    Property information used for document requirement evaluation.
    
    Contains details about the property that affect which documents
    are required (e.g., year built affects Lead Paint Disclosure).
    """
    year_built: Optional[int] = None
    property_type: str = "residential"  # residential, commercial, land, multi-family
    state: Optional[str] = None  # State abbreviation (e.g., "CA", "TX")
    county: Optional[str] = None
    has_pool: bool = False
    has_septic: bool = False
    has_well: bool = False
    is_hoa: bool = False
    is_condo: bool = False
    is_new_construction: bool = False
    square_footage: Optional[int] = None
    lot_size_acres: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "year_built": self.year_built,
            "property_type": self.property_type,
            "state": self.state,
            "county": self.county,
            "has_pool": self.has_pool,
            "has_septic": self.has_septic,
            "has_well": self.has_well,
            "is_hoa": self.is_hoa,
            "is_condo": self.is_condo,
            "is_new_construction": self.is_new_construction,
            "square_footage": self.square_footage,
            "lot_size_acres": self.lot_size_acres,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropertyMetadata":
        """Create from dictionary."""
        return cls(
            year_built=data.get("year_built"),
            property_type=data.get("property_type", "residential"),
            state=data.get("state"),
            county=data.get("county"),
            has_pool=data.get("has_pool", False),
            has_septic=data.get("has_septic", False),
            has_well=data.get("has_well", False),
            is_hoa=data.get("is_hoa", False),
            is_condo=data.get("is_condo", False),
            is_new_construction=data.get("is_new_construction", False),
            square_footage=data.get("square_footage"),
            lot_size_acres=data.get("lot_size_acres"),
        )


@dataclass
class TransactionMetadata:
    """
    Transaction information used for document requirement evaluation.
    
    Contains details about the transaction that affect which documents
    are required (e.g., financing type affects loan documents).
    """
    transaction_type: str = "purchase"  # purchase, sale, lease, refinance
    financing_type: str = "conventional"  # conventional, fha, va, cash, usda
    is_cash_deal: bool = False
    has_contingencies: bool = True
    inspection_contingency: bool = True
    financing_contingency: bool = True
    appraisal_contingency: bool = True
    sale_contingency: bool = False
    buyer_is_investor: bool = False
    is_short_sale: bool = False
    is_foreclosure: bool = False
    is_estate_sale: bool = False
    listing_agent_represents_buyer: bool = False  # Dual agency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transaction_type": self.transaction_type,
            "financing_type": self.financing_type,
            "is_cash_deal": self.is_cash_deal,
            "has_contingencies": self.has_contingencies,
            "inspection_contingency": self.inspection_contingency,
            "financing_contingency": self.financing_contingency,
            "appraisal_contingency": self.appraisal_contingency,
            "sale_contingency": self.sale_contingency,
            "buyer_is_investor": self.buyer_is_investor,
            "is_short_sale": self.is_short_sale,
            "is_foreclosure": self.is_foreclosure,
            "is_estate_sale": self.is_estate_sale,
            "listing_agent_represents_buyer": self.listing_agent_represents_buyer,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransactionMetadata":
        """Create from dictionary."""
        return cls(
            transaction_type=data.get("transaction_type", "purchase"),
            financing_type=data.get("financing_type", "conventional"),
            is_cash_deal=data.get("is_cash_deal", False),
            has_contingencies=data.get("has_contingencies", True),
            inspection_contingency=data.get("inspection_contingency", True),
            financing_contingency=data.get("financing_contingency", True),
            appraisal_contingency=data.get("appraisal_contingency", True),
            sale_contingency=data.get("sale_contingency", False),
            buyer_is_investor=data.get("buyer_is_investor", False),
            is_short_sale=data.get("is_short_sale", False),
            is_foreclosure=data.get("is_foreclosure", False),
            is_estate_sale=data.get("is_estate_sale", False),
            listing_agent_represents_buyer=data.get("listing_agent_represents_buyer", False),
        )


class RequirementCondition(Enum):
    """Conditions that determine if a document is required."""
    
    # Property-based conditions
    BUILT_BEFORE_1978 = "built_before_1978"
    HAS_POOL = "has_pool"
    HAS_SEPTIC = "has_septic"
    HAS_WELL = "has_well"
    IS_HOA = "is_hoa"
    IS_CONDO = "is_condo"
    IS_NEW_CONSTRUCTION = "is_new_construction"
    IS_RESIDENTIAL = "is_residential"
    IS_COMMERCIAL = "is_commercial"
    
    # Transaction-based conditions
    IS_PURCHASE = "is_purchase"
    IS_SALE = "is_sale"
    HAS_FINANCING = "has_financing"
    IS_FHA_LOAN = "is_fha_loan"
    IS_VA_LOAN = "is_va_loan"
    IS_CASH_DEAL = "is_cash_deal"
    HAS_INSPECTION_CONTINGENCY = "has_inspection_contingency"
    HAS_FINANCING_CONTINGENCY = "has_financing_contingency"
    HAS_APPRAISAL_CONTINGENCY = "has_appraisal_contingency"
    IS_SHORT_SALE = "is_short_sale"
    IS_FORECLOSURE = "is_foreclosure"
    IS_DUAL_AGENCY = "is_dual_agency"
    
    # State-specific
    STATE_CALIFORNIA = "state_california"
    STATE_TEXAS = "state_texas"
    STATE_FLORIDA = "state_florida"
    STATE_NEW_YORK = "state_new_york"
    
    # Always required
    ALWAYS = "always"


@dataclass
class RequiredDocumentRule:
    """
    Rule defining when a document type is required.
    
    A rule specifies which document type is required and under what
    conditions (based on property and transaction metadata).
    """
    id: str
    document_type: DocumentType
    conditions: List[RequirementCondition]
    condition_logic: str = "all"  # "all" (AND) or "any" (OR)
    description: str = ""
    priority: str = "required"  # "required", "recommended", "optional"
    states: Optional[List[str]] = None  # Restrict to specific states
    
    def evaluate(
        self, 
        property_meta: Optional[PropertyMetadata],
        transaction_meta: Optional[TransactionMetadata],
    ) -> bool:
        """
        Evaluate if this document is required given the metadata.
        
        Args:
            property_meta: Property information
            transaction_meta: Transaction information
            
        Returns:
            True if the document is required, False otherwise
        """
        if not self.conditions:
            return False
        
        # Check state restriction first
        if self.states and property_meta and property_meta.state:
            if property_meta.state.upper() not in [s.upper() for s in self.states]:
                return False
        
        # Evaluate each condition
        results = []
        for condition in self.conditions:
            result = self._evaluate_condition(condition, property_meta, transaction_meta)
            results.append(result)
        
        # Apply logic
        if self.condition_logic == "all":
            return all(results)
        else:  # "any"
            return any(results)
    
    def _evaluate_condition(
        self,
        condition: RequirementCondition,
        property_meta: Optional[PropertyMetadata],
        transaction_meta: Optional[TransactionMetadata],
    ) -> bool:
        """Evaluate a single condition."""
        
        # Always required
        if condition == RequirementCondition.ALWAYS:
            return True
        
        # Property-based conditions
        if property_meta:
            if condition == RequirementCondition.BUILT_BEFORE_1978:
                return property_meta.year_built is not None and property_meta.year_built < 1978
            if condition == RequirementCondition.HAS_POOL:
                return property_meta.has_pool
            if condition == RequirementCondition.HAS_SEPTIC:
                return property_meta.has_septic
            if condition == RequirementCondition.HAS_WELL:
                return property_meta.has_well
            if condition == RequirementCondition.IS_HOA:
                return property_meta.is_hoa
            if condition == RequirementCondition.IS_CONDO:
                return property_meta.is_condo
            if condition == RequirementCondition.IS_NEW_CONSTRUCTION:
                return property_meta.is_new_construction
            if condition == RequirementCondition.IS_RESIDENTIAL:
                return property_meta.property_type == "residential"
            if condition == RequirementCondition.IS_COMMERCIAL:
                return property_meta.property_type == "commercial"
            
            # State conditions
            if condition == RequirementCondition.STATE_CALIFORNIA:
                return bool(property_meta.state and property_meta.state.upper() == "CA")
            if condition == RequirementCondition.STATE_TEXAS:
                return bool(property_meta.state and property_meta.state.upper() == "TX")
            if condition == RequirementCondition.STATE_FLORIDA:
                return bool(property_meta.state and property_meta.state.upper() == "FL")
            if condition == RequirementCondition.STATE_NEW_YORK:
                return bool(property_meta.state and property_meta.state.upper() == "NY")
        
        # Transaction-based conditions
        if transaction_meta:
            if condition == RequirementCondition.IS_PURCHASE:
                return transaction_meta.transaction_type == "purchase"
            if condition == RequirementCondition.IS_SALE:
                return transaction_meta.transaction_type == "sale"
            if condition == RequirementCondition.HAS_FINANCING:
                return not transaction_meta.is_cash_deal
            if condition == RequirementCondition.IS_FHA_LOAN:
                return transaction_meta.financing_type == "fha"
            if condition == RequirementCondition.IS_VA_LOAN:
                return transaction_meta.financing_type == "va"
            if condition == RequirementCondition.IS_CASH_DEAL:
                return transaction_meta.is_cash_deal
            if condition == RequirementCondition.HAS_INSPECTION_CONTINGENCY:
                return transaction_meta.inspection_contingency
            if condition == RequirementCondition.HAS_FINANCING_CONTINGENCY:
                return transaction_meta.financing_contingency
            if condition == RequirementCondition.HAS_APPRAISAL_CONTINGENCY:
                return transaction_meta.appraisal_contingency
            if condition == RequirementCondition.IS_SHORT_SALE:
                return transaction_meta.is_short_sale
            if condition == RequirementCondition.IS_FORECLOSURE:
                return transaction_meta.is_foreclosure
            if condition == RequirementCondition.IS_DUAL_AGENCY:
                return transaction_meta.listing_agent_represents_buyer
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "document_type": self.document_type.value,
            "conditions": [c.value for c in self.conditions],
            "condition_logic": self.condition_logic,
            "description": self.description,
            "priority": self.priority,
            "states": self.states,
        }


# Default document requirement rules
DEFAULT_REQUIREMENT_RULES: List[RequiredDocumentRule] = [
    # Lead Paint Disclosure - required for pre-1978 residential properties
    RequiredDocumentRule(
        id="lead-paint-pre1978",
        document_type=DocumentType.LEAD_PAINT_DISCLOSURE,
        conditions=[RequirementCondition.BUILT_BEFORE_1978, RequirementCondition.IS_RESIDENTIAL],
        condition_logic="all",
        description="Lead-based paint disclosure required for homes built before 1978",
        priority="required",
    ),
    
    # Purchase Agreement - always required for purchases
    RequiredDocumentRule(
        id="purchase-agreement",
        document_type=DocumentType.PURCHASE_AGREEMENT,
        conditions=[RequirementCondition.IS_PURCHASE],
        condition_logic="all",
        description="Purchase agreement required for all purchase transactions",
        priority="required",
    ),
    
    # Seller Disclosure - required for residential sales
    RequiredDocumentRule(
        id="seller-disclosure",
        document_type=DocumentType.SELLER_DISCLOSURE,
        conditions=[RequirementCondition.IS_RESIDENTIAL],
        condition_logic="all",
        description="Seller property disclosure required for residential properties",
        priority="required",
    ),
    
    # Agency Disclosure - always required
    RequiredDocumentRule(
        id="agency-disclosure",
        document_type=DocumentType.AGENCY_DISCLOSURE,
        conditions=[RequirementCondition.ALWAYS],
        condition_logic="all",
        description="Agency disclosure required for all transactions",
        priority="required",
    ),
    
    # Inspection Report - required when inspection contingency exists
    RequiredDocumentRule(
        id="inspection-report",
        document_type=DocumentType.INSPECTION_REPORT,
        conditions=[RequirementCondition.HAS_INSPECTION_CONTINGENCY],
        condition_logic="all",
        description="Inspection report required when inspection contingency is in place",
        priority="required",
    ),
    
    # Pre-Approval Letter - required for financed purchases
    RequiredDocumentRule(
        id="pre-approval-letter",
        document_type=DocumentType.PRE_APPROVAL_LETTER,
        conditions=[RequirementCondition.IS_PURCHASE, RequirementCondition.HAS_FINANCING],
        condition_logic="all",
        description="Pre-approval letter required for financed purchases",
        priority="required",
    ),
    
    # Title Commitment - required for purchases
    RequiredDocumentRule(
        id="title-commitment",
        document_type=DocumentType.TITLE_COMMITMENT,
        conditions=[RequirementCondition.IS_PURCHASE],
        condition_logic="all",
        description="Title commitment required for purchase transactions",
        priority="required",
    ),
    
    # Appraisal - required when appraisal contingency exists
    RequiredDocumentRule(
        id="appraisal",
        document_type=DocumentType.APPRAISAL,
        conditions=[RequirementCondition.HAS_APPRAISAL_CONTINGENCY],
        condition_logic="all",
        description="Appraisal required when appraisal contingency is in place",
        priority="required",
    ),
    
    # Loan Estimate - required for financed transactions
    RequiredDocumentRule(
        id="loan-estimate",
        document_type=DocumentType.LOAN_ESTIMATE,
        conditions=[RequirementCondition.HAS_FINANCING],
        condition_logic="all",
        description="Loan estimate required for financed transactions",
        priority="required",
    ),
    
    # Closing Disclosure - required for financed transactions
    RequiredDocumentRule(
        id="closing-disclosure",
        document_type=DocumentType.CLOSING_DISCLOSURE,
        conditions=[RequirementCondition.HAS_FINANCING],
        condition_logic="all",
        description="Closing disclosure required for financed transactions",
        priority="required",
    ),
    
    # Wire Instructions - recommended for all transactions
    RequiredDocumentRule(
        id="wire-instructions",
        document_type=DocumentType.WIRE_INSTRUCTIONS,
        conditions=[RequirementCondition.ALWAYS],
        condition_logic="all",
        description="Wire instructions recommended for closing funds transfer",
        priority="recommended",
    ),
    
    # HOA Documents - required for HOA properties
    RequiredDocumentRule(
        id="hoa-docs",
        document_type=DocumentType.PROPERTY_DISCLOSURE,  # Using property disclosure for HOA docs
        conditions=[RequirementCondition.IS_HOA],
        condition_logic="all",
        description="HOA documents required for properties in HOA",
        priority="required",
    ),
]


@dataclass
class MissingDocument:
    """
    Represents a document that is required but missing.
    """
    document_type: DocumentType
    rule_id: str
    reason: str
    priority: str  # "required", "recommended", "optional"
    states: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_type": self.document_type.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "priority": self.priority,
            "states": self.states,
        }


@dataclass
class MissingDocumentReport:
    """
    Report of all missing required documents for a transaction.
    """
    missing_required: List[MissingDocument] = field(default_factory=list)
    missing_recommended: List[MissingDocument] = field(default_factory=list)
    documents_found: List[DocumentType] = field(default_factory=list)
    property_metadata: Optional[PropertyMetadata] = None
    transaction_metadata: Optional[TransactionMetadata] = None
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def has_missing_required(self) -> bool:
        """Check if there are any missing required documents."""
        return len(self.missing_required) > 0
    
    @property
    def missing_count(self) -> int:
        """Total count of missing documents (required + recommended)."""
        return len(self.missing_required) + len(self.missing_recommended)
    
    @property
    def required_count(self) -> int:
        """Count of missing required documents."""
        return len(self.missing_required)
    
    @property
    def is_complete(self) -> bool:
        """Check if all required documents are present."""
        return not self.has_missing_required
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "missing_required": [m.to_dict() for m in self.missing_required],
            "missing_recommended": [m.to_dict() for m in self.missing_recommended],
            "documents_found": [d.value for d in self.documents_found],
            "property_metadata": self.property_metadata.to_dict() if self.property_metadata else None,
            "transaction_metadata": self.transaction_metadata.to_dict() if self.transaction_metadata else None,
            "has_missing_required": self.has_missing_required,
            "missing_count": self.missing_count,
            "required_count": self.required_count,
            "is_complete": self.is_complete,
            "evaluated_at": self.evaluated_at,
        }


def get_required_documents(
    property_meta: Optional[PropertyMetadata] = None,
    transaction_meta: Optional[TransactionMetadata] = None,
    rules: Optional[List[RequiredDocumentRule]] = None,
    include_recommended: bool = True,
) -> List[Tuple[DocumentType, RequiredDocumentRule]]:
    """
    Get list of required documents based on property and transaction metadata.
    
    Args:
        property_meta: Property information
        transaction_meta: Transaction information
        rules: Optional custom rules (uses DEFAULT_REQUIREMENT_RULES if not provided)
        include_recommended: Include recommended documents, not just required
        
    Returns:
        List of (DocumentType, Rule) tuples for required documents
    """
    rules = rules or DEFAULT_REQUIREMENT_RULES
    required: List[Tuple[DocumentType, RequiredDocumentRule]] = []
    
    for rule in rules:
        # Filter by priority if not including recommended
        if not include_recommended and rule.priority != "required":
            continue
        
        if rule.evaluate(property_meta, transaction_meta):
            required.append((rule.document_type, rule))
    
    return required


def detect_missing_documents(
    classified_docs: List[Dict[str, Any]],
    property_meta: Optional[PropertyMetadata] = None,
    transaction_meta: Optional[TransactionMetadata] = None,
    rules: Optional[List[RequiredDocumentRule]] = None,
) -> MissingDocumentReport:
    """
    Detect missing required documents by comparing classified docs against requirements.
    
    Args:
        classified_docs: List of classified document dicts with 'doc_type' field
        property_meta: Property information for requirement evaluation
        transaction_meta: Transaction information for requirement evaluation
        rules: Optional custom rules (uses DEFAULT_REQUIREMENT_RULES if not provided)
        
    Returns:
        MissingDocumentReport with details on missing documents
    """
    # Extract document types that were found
    found_types: set[DocumentType] = set()
    for doc in classified_docs:
        doc_type_str = doc.get("doc_type", doc.get("document_type", ""))
        if doc_type_str:
            try:
                found_types.add(DocumentType.from_string(doc_type_str))
            except (ValueError, KeyError):
                pass
    
    # Get required documents
    required_docs = get_required_documents(
        property_meta, transaction_meta, rules, include_recommended=True
    )
    
    # Find missing documents
    missing_required: List[MissingDocument] = []
    missing_recommended: List[MissingDocument] = []
    
    for doc_type, rule in required_docs:
        if doc_type not in found_types:
            missing_doc = MissingDocument(
                document_type=doc_type,
                rule_id=rule.id,
                reason=rule.description,
                priority=rule.priority,
                states=rule.states,
            )
            
            if rule.priority == "required":
                missing_required.append(missing_doc)
            else:
                missing_recommended.append(missing_doc)
    
    return MissingDocumentReport(
        missing_required=missing_required,
        missing_recommended=missing_recommended,
        documents_found=list(found_types),
        property_metadata=property_meta,
        transaction_metadata=transaction_meta,
    )


# ============================================================================
# OCR Metadata and Quality Checking
# ============================================================================

@dataclass
class OcrDocumentInfo:
    """
    OCR metadata for a document, populated from splitter output.
    
    Tracks whether a document was processed via OCR and its quality.
    """
    has_scanned_pages: bool = False
    ocr_confidence: float = 1.0  # Default 1.0 for non-OCR documents
    meets_accuracy_target: bool = True  # >= 95% confidence
    ocr_engine_used: Optional[str] = None
    scanned_page_count: int = 0
    total_page_count: int = 0
    
    @property
    def is_low_confidence(self) -> bool:
        """Check if OCR confidence is below acceptable threshold."""
        return self.has_scanned_pages and self.ocr_confidence < OCR_LOW_CONFIDENCE_THRESHOLD
    
    @property
    def is_poor_quality(self) -> bool:
        """Check if OCR quality is poor (may need re-OCR)."""
        return self.has_scanned_pages and self.ocr_confidence < OCR_POOR_QUALITY_THRESHOLD
    
    @property
    def needs_review(self) -> bool:
        """Check if document needs manual review due to OCR quality."""
        return self.has_scanned_pages and not self.meets_accuracy_target
    
    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "OcrDocumentInfo":
        """
        Create OcrDocumentInfo from a split document dict.
        
        Args:
            doc: Document dict from splitter with OCR metadata
            
        Returns:
            OcrDocumentInfo populated from document
        """
        page_range = doc.get("page_range", [])
        total_pages = page_range[1] - page_range[0] + 1 if len(page_range) == 2 else 0
        
        return cls(
            has_scanned_pages=doc.get("has_scanned_pages", False),
            ocr_confidence=doc.get("ocr_confidence", 1.0),
            meets_accuracy_target=doc.get("meets_accuracy_target", True),
            ocr_engine_used=doc.get("ocr_engine_used"),
            scanned_page_count=doc.get("scanned_page_count", 0),
            total_page_count=total_pages,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "has_scanned_pages": self.has_scanned_pages,
            "ocr_confidence": round(self.ocr_confidence, 4),
            "meets_accuracy_target": self.meets_accuracy_target,
            "ocr_engine_used": self.ocr_engine_used,
            "scanned_page_count": self.scanned_page_count,
            "total_page_count": self.total_page_count,
            "is_low_confidence": self.is_low_confidence,
            "is_poor_quality": self.is_poor_quality,
            "needs_review": self.needs_review,
        }


# ============================================================================
# Review Flag (C3)
# ============================================================================

@dataclass
class ReviewFlag:
    """
    Flag indicating a document needs human review (C3).
    
    Created when classification confidence is below threshold or
    when other quality issues are detected.
    """
    document_id: int
    reasons: List[ReviewReason] = field(default_factory=list)
    confidence: float = 0.0
    classified_type: str = "Unknown"
    alternative_types: List[Dict[str, Any]] = field(default_factory=list)
    suggested_action: str = "manual_review"
    priority: str = "normal"  # "low", "normal", "high", "critical"
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Set priority based on reasons and confidence."""
        if not self.priority or self.priority == "normal":
            self.priority = self._calculate_priority()
    
    def _calculate_priority(self) -> str:
        """Calculate review priority based on reasons and confidence."""
        # Critical: completely unknown type
        if ReviewReason.UNKNOWN_TYPE in self.reasons and self.confidence == 0.0:
            return "critical"
        
        # High: very low confidence or multiple serious issues
        if self.confidence < 0.50 or len(self.reasons) >= 3:
            return "high"
        
        # Low: minor issues like title-only classification with decent confidence
        if (len(self.reasons) == 1 and 
            ReviewReason.TITLE_ONLY_CLASSIFICATION in self.reasons and
            self.confidence >= 0.70):
            return "low"
        
        return "normal"
    
    @property
    def reason_descriptions(self) -> List[str]:
        """Get human-readable descriptions of review reasons."""
        descriptions = {
            ReviewReason.LOW_CONFIDENCE: f"Classification confidence ({self.confidence:.0%}) below threshold",
            ReviewReason.UNKNOWN_TYPE: "Could not determine document type",
            ReviewReason.AMBIGUOUS_CLASSIFICATION: "Multiple document types matched with similar confidence",
            ReviewReason.OCR_QUALITY_ISSUE: "OCR quality may have affected classification accuracy",
            ReviewReason.MULTIPLE_CLOSE_ALTERNATIVES: "Multiple alternative classifications with close confidence scores",
            ReviewReason.TITLE_ONLY_CLASSIFICATION: "Classification based on document title only, not content",
        }
        return [descriptions.get(r, str(r)) for r in self.reasons]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "reasons": [r.value for r in self.reasons],
            "reason_descriptions": self.reason_descriptions,
            "confidence": round(self.confidence, 4),
            "classified_type": self.classified_type,
            "alternative_types": self.alternative_types,
            "suggested_action": self.suggested_action,
            "priority": self.priority,
            "notes": self.notes,
        }


def check_needs_review(
    result: "ClassificationResult",
    doc_id: int,
    config: "ClassifierConfig",
) -> Optional[ReviewFlag]:
    """
    Check if a classification result needs human review (C3).
    
    Args:
        result: Classification result to check
        doc_id: Document ID
        config: Classifier configuration with review threshold
        
    Returns:
        ReviewFlag if review is needed, None otherwise
    """
    reasons: List[ReviewReason] = []
    notes_parts: List[str] = []
    
    # Check confidence threshold (primary C3 criterion)
    if result.confidence < config.review_confidence_threshold:
        reasons.append(ReviewReason.LOW_CONFIDENCE)
        notes_parts.append(
            f"Confidence {result.confidence:.0%} < {config.review_confidence_threshold:.0%} threshold"
        )
    
    # Check for unknown type
    if result.document_type == DocumentType.UNKNOWN:
        reasons.append(ReviewReason.UNKNOWN_TYPE)
        notes_parts.append("Document type could not be determined")
    
    # Check for title-only classification (less reliable)
    if result.method == "title":
        reasons.append(ReviewReason.TITLE_ONLY_CLASSIFICATION)
        notes_parts.append("Classified using title only, content not analyzed")
    
    # Check for OCR quality issues affecting classification
    if result.ocr_impacted and result.ocr_quality_factor < 0.90:
        reasons.append(ReviewReason.OCR_QUALITY_ISSUE)
        notes_parts.append(
            f"OCR quality factor {result.ocr_quality_factor:.0%} may affect accuracy"
        )
    
    # Check for ambiguous classification (close alternatives)
    if result.alternative_types:
        # If best alternative is within 10% of main classification
        best_alt_confidence = result.alternative_types[0][1] if result.alternative_types else 0
        if best_alt_confidence > 0 and (result.confidence - best_alt_confidence) < 0.10:
            reasons.append(ReviewReason.AMBIGUOUS_CLASSIFICATION)
            alt_type = result.alternative_types[0][0].value
            notes_parts.append(
                f"Alternative type '{alt_type}' has similar confidence ({best_alt_confidence:.0%})"
            )
        
        # Multiple close alternatives
        close_alts = [
            (t, c) for t, c in result.alternative_types 
            if (result.confidence - c) < 0.15
        ]
        if len(close_alts) >= 2:
            reasons.append(ReviewReason.MULTIPLE_CLOSE_ALTERNATIVES)
            notes_parts.append(f"{len(close_alts)} alternative types with close confidence")
    
    # No review needed
    if not reasons:
        return None
    
    # Create review flag
    return ReviewFlag(
        document_id=doc_id,
        reasons=reasons,
        confidence=result.confidence,
        classified_type=result.document_type.value,
        alternative_types=[
            {"type": t.value, "confidence": round(c, 4)}
            for t, c in result.alternative_types[:3]  # Top 3 alternatives
        ],
        suggested_action=_get_suggested_action(reasons),
        notes="; ".join(notes_parts) if notes_parts else None,
    )


def _get_suggested_action(reasons: List[ReviewReason]) -> str:
    """Get suggested action based on review reasons."""
    if ReviewReason.UNKNOWN_TYPE in reasons:
        return "manual_classification"
    if ReviewReason.AMBIGUOUS_CLASSIFICATION in reasons:
        return "confirm_type_selection"
    if ReviewReason.OCR_QUALITY_ISSUE in reasons:
        return "verify_ocr_and_reclassify"
    if ReviewReason.TITLE_ONLY_CLASSIFICATION in reasons:
        return "verify_content_matches_title"
    return "manual_review"


def preprocess_ocr_text(text: str, ocr_confidence: float = 1.0) -> str:
    """
    Preprocess OCR'd text to improve classification accuracy.
    
    Cleans up common OCR artifacts that may interfere with classification.
    More aggressive cleaning for lower-confidence OCR.
    
    Args:
        text: Raw OCR text
        ocr_confidence: OCR confidence score (0.0-1.0)
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    cleaned = text
    
    # Basic cleanup for all text
    # Remove excessive whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # For lower confidence OCR, apply more aggressive cleaning
    if ocr_confidence < OCR_LOW_CONFIDENCE_THRESHOLD:
        # Common OCR errors: 0/O, 1/l/I, 5/S
        # Don't correct these automatically as it could corrupt data
        # But we can normalize some patterns
        
        # Remove stray single characters that are likely noise
        cleaned = re.sub(r'\s[^\w\s]\s', ' ', cleaned)
        
        # Remove lines that are mostly special characters (likely noise)
        lines = cleaned.split('\n')
        cleaned_lines = []
        for line in lines:
            # Count alphanumeric vs special chars
            if line.strip():
                alnum_count = sum(1 for c in line if c.isalnum() or c.isspace())
                total_count = len(line)
                if total_count == 0 or alnum_count / total_count > 0.3:
                    cleaned_lines.append(line)
        cleaned = '\n'.join(cleaned_lines)
    
    return cleaned.strip()


def assess_text_quality_for_classification(
    text: str,
    ocr_info: OcrDocumentInfo,
) -> Tuple[str, float]:
    """
    Assess and potentially preprocess text for classification.
    
    Returns cleaned text and a quality adjustment factor.
    
    Args:
        text: Document text
        ocr_info: OCR metadata
        
    Returns:
        Tuple of (cleaned_text, quality_factor)
    """
    if not ocr_info.has_scanned_pages:
        # Non-OCR document, no preprocessing needed
        return text, 1.0
    
    # Apply OCR-specific preprocessing
    cleaned_text = preprocess_ocr_text(text, ocr_info.ocr_confidence)
    
    # Calculate quality factor that will adjust classification confidence
    # Higher OCR confidence = higher quality factor
    # This penalizes classification confidence for low-quality OCR
    if ocr_info.meets_accuracy_target:
        quality_factor = 1.0
    elif ocr_info.ocr_confidence >= OCR_LOW_CONFIDENCE_THRESHOLD:
        # Moderate quality - small penalty
        quality_factor = 0.95
    elif ocr_info.ocr_confidence >= OCR_POOR_QUALITY_THRESHOLD:
        # Low quality - moderate penalty
        quality_factor = 0.85
    else:
        # Poor quality - significant penalty
        quality_factor = 0.70
    
    return cleaned_text, quality_factor


# ============================================================================
# Classification Result
# ============================================================================

@dataclass
class ClassificationResult:
    """Result of document classification."""
    
    document_type: DocumentType
    confidence: float  # 0.0 to 1.0
    method: str  # "llm", "keyword", "keyword_custom", "title", "mock"
    
    # Additional context
    suggested_folder: str = ""
    alternative_types: List[Tuple[DocumentType, float]] = field(default_factory=list)
    reasoning: Optional[str] = None
    processing_time_ms: float = 0.0
    
    # OCR quality tracking (C2)
    ocr_info: Optional[OcrDocumentInfo] = None
    ocr_quality_factor: float = 1.0  # Factor applied to confidence due to OCR quality
    original_confidence: Optional[float] = None  # Pre-OCR-adjustment confidence
    
    # Custom document type tracking (C4)
    custom_type_id: Optional[str] = None  # ID of matched custom type
    custom_type_name: Optional[str] = None  # Display name of matched custom type
    
    def __post_init__(self):
        if not self.suggested_folder:
            # For custom types, get folder from registry
            if self.document_type == DocumentType.CUSTOM and self.custom_type_id:
                registry = get_custom_type_registry()
                custom_type = registry.get(self.custom_type_id)
                if custom_type:
                    self.suggested_folder = custom_type.folder
                else:
                    self.suggested_folder = "Custom"
            else:
                self.suggested_folder = DOCUMENT_TYPE_TO_FOLDER.get(
                    self.document_type, "Other"
                )
    
    @property
    def meets_accuracy_target(self) -> bool:
        """Check if classification meets >90% accuracy target."""
        return self.confidence >= 0.90
    
    @property 
    def ocr_impacted(self) -> bool:
        """Check if classification was impacted by OCR quality."""
        return self.ocr_quality_factor < 1.0
    
    @property
    def is_custom_type(self) -> bool:
        """Check if this is a custom document type."""
        return self.document_type == DocumentType.CUSTOM and self.custom_type_id is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "document_type": self.document_type.value,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "suggested_folder": self.suggested_folder,
            "alternative_types": [
                {"type": t.value, "confidence": round(c, 4)}
                for t, c in self.alternative_types
            ],
            "reasoning": self.reasoning,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "meets_accuracy_target": self.meets_accuracy_target,
        }
        
        # Add OCR information if present
        if self.ocr_info:
            result["ocr_info"] = self.ocr_info.to_dict()
            result["ocr_quality_factor"] = round(self.ocr_quality_factor, 4)
            result["ocr_impacted"] = self.ocr_impacted
            if self.original_confidence is not None:
                result["original_confidence"] = round(self.original_confidence, 4)
        
        # Add custom type information if present (C4)
        if self.custom_type_id:
            result["custom_type_id"] = self.custom_type_id
            result["custom_type_name"] = self.custom_type_name
            result["is_custom_type"] = True
        
        return result


# ============================================================================
# Classifier Configuration
# ============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for document classification."""
    
    # LLM settings
    use_llm: bool = True
    llm_provider: str = "openai"  # "openai", "anthropic"
    llm_model: str = "gpt-4o-mini"  # or "claude-3-haiku-20240307"
    llm_temperature: float = 0.0  # Deterministic for classification
    llm_max_tokens: int = 500
    
    # Text sampling
    max_chars_for_classification: int = 2000  # First N chars to analyze
    include_detected_title: bool = True
    
    # Fallback behavior
    use_keyword_fallback: bool = True
    keyword_min_confidence: float = 0.70
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.50  # Below this, mark as Unknown
    high_confidence_threshold: float = 0.90  # Above this, consider definitive
    
    # Review queue settings (C3)
    review_confidence_threshold: float = 0.80  # Below this, flag for review
    enable_review_queue: bool = True  # Whether to create review flags
    flag_title_only_classifications: bool = True  # Flag title-based classifications
    flag_ocr_quality_issues: bool = True  # Flag OCR-impacted classifications
    
    # Custom document types settings (C4)
    custom_types_enabled: bool = True  # Enable custom type matching
    custom_types_file_path: Optional[str] = None  # Path to custom types JSON file
    custom_types_priority: bool = True  # Check custom types before standard types
    custom_types_brokerage_id: Optional[str] = None  # Filter by brokerage
    
    # Mock mode for testing
    use_mock: bool = False


# ============================================================================
# Keyword-Based Classification
# ============================================================================

# Keyword patterns for each document type with weights
KEYWORD_PATTERNS: Dict[DocumentType, List[Tuple[str, float]]] = {
    DocumentType.BUY_SELL_AGREEMENT: [
        (r"buy[\s-]*sell\s+agreement", 1.0),
        (r"residential\s+real\s+estate\s+purchase", 0.95),
        (r"purchase\s+and\s+sale\s+agreement", 0.95),
        (r"real\s+estate\s+purchase\s+agreement", 0.90),
        (r"purchase\s+price.*\$[\d,]+", 0.70),
        (r"earnest\s+money", 0.60),
        (r"buyer\s+agrees\s+to\s+purchase", 0.85),
    ],
    DocumentType.PURCHASE_AGREEMENT: [
        (r"purchase\s+agreement", 0.90),
        (r"agreement\s+to\s+purchase", 0.85),
        (r"offer\s+to\s+purchase", 0.80),
    ],
    DocumentType.COUNTER_OFFER: [
        (r"counter[\s-]*offer", 1.0),
        (r"counter\s+proposal", 0.90),
        (r"rejection\s+and\s+counter", 0.85),
    ],
    DocumentType.AMENDMENT: [
        (r"amendment\s+to", 0.95),
        (r"contract\s+amendment", 0.90),
        (r"amend.*agreement", 0.80),
    ],
    DocumentType.ADDENDUM: [
        (r"addendum", 0.95),
        (r"additional\s+terms", 0.70),
        (r"exhibit\s+[a-z]", 0.60),
    ],
    DocumentType.SELLER_DISCLOSURE: [
        (r"seller'?s?\s+property\s+disclosure", 1.0),
        (r"seller'?s?\s+disclosure\s+statement", 0.95),
        (r"property\s+disclosure\s+statement", 0.90),
        (r"the\s+seller\s+makes\s+the\s+following\s+disclosures", 0.85),
    ],
    DocumentType.LEAD_PAINT_DISCLOSURE: [
        (r"lead[\s-]*based\s+paint", 1.0),
        (r"lead\s+paint\s+disclosure", 0.95),
        (r"disclosure\s+of\s+information.*lead", 0.90),
        (r"pre[\s-]*1978", 0.70),
    ],
    DocumentType.MOLD_DISCLOSURE: [
        (r"mold\s+disclosure", 1.0),
        (r"mold\s+and\s+mildew", 0.90),
        (r"presence\s+of\s+mold", 0.85),
    ],
    DocumentType.INSPECTION_REPORT: [
        (r"inspection\s+report", 0.95),
        (r"home\s+inspection", 0.90),
        (r"property\s+inspection", 0.85),
        (r"inspector", 0.60),
    ],
    DocumentType.INSPECTION_RESPONSE: [
        (r"inspection\s+response", 0.95),
        (r"response\s+to\s+inspection", 0.90),
        (r"buyer'?s?\s+inspection\s+notice", 0.85),
        (r"inspection\s+objection", 0.80),
    ],
    DocumentType.CLOSING_STATEMENT: [
        (r"closing\s+statement", 0.95),
        (r"settlement\s+statement", 0.90),
        (r"hud[\s-]*1", 0.90),
    ],
    DocumentType.TITLE_COMMITMENT: [
        (r"title\s+commitment", 0.95),
        (r"commitment\s+for\s+title\s+insurance", 0.90),
        (r"preliminary\s+title\s+report", 0.85),
    ],
    DocumentType.CLOSING_DISCLOSURE: [
        (r"closing\s+disclosure", 0.95),
        (r"loan\s+disclosure", 0.80),
    ],
    DocumentType.LOAN_ESTIMATE: [
        (r"loan\s+estimate", 0.95),
        (r"good\s+faith\s+estimate", 0.85),
    ],
    DocumentType.PRE_APPROVAL_LETTER: [
        (r"pre[\s-]*approval\s+letter", 0.95),
        (r"mortgage\s+pre[\s-]*approval", 0.90),
        (r"pre[\s-]*qualified", 0.75),
    ],
    DocumentType.AGENCY_DISCLOSURE: [
        (r"agency\s+disclosure", 0.95),
        (r"disclosure\s+regarding\s+real\s+estate\s+agency", 0.90),
        (r"agency\s+relationship", 0.80),
    ],
    DocumentType.BUYER_AGENCY_AGREEMENT: [
        (r"buyer\s+agency\s+agreement", 0.95),
        (r"buyer\s+representation\s+agreement", 0.90),
        (r"exclusive\s+buyer\s+agency", 0.85),
    ],
    DocumentType.WIRE_INSTRUCTIONS: [
        (r"wire\s+transfer\s+instructions", 0.95),
        (r"wiring\s+instructions", 0.90),
        (r"wire\s+instructions", 0.90),
        (r"aba\s+routing", 0.70),
        (r"beneficiary\s+account", 0.65),
    ],
    DocumentType.EARNEST_MONEY_RECEIPT: [
        (r"earnest\s+money\s+receipt", 0.95),
        (r"receipt\s+for\s+earnest\s+money", 0.90),
        (r"deposit\s+receipt", 0.75),
    ],
    DocumentType.LISTING_AGREEMENT: [
        (r"listing\s+agreement", 0.95),
        (r"exclusive\s+right\s+to\s+sell", 0.90),
        (r"exclusive\s+listing", 0.85),
    ],
}


@dataclass
class CustomTypeMatch:
    """Result of matching against custom document types."""
    
    custom_type: CustomDocumentType
    confidence: float
    matched_patterns: List[str]  # Patterns that matched
    reasoning: str


def match_custom_types(
    text: str,
    config: Optional[ClassifierConfig] = None,
) -> Optional[CustomTypeMatch]:
    """
    Match document text against custom document types.
    
    Custom types are checked in priority order (highest first).
    Returns the best match if confidence exceeds threshold.
    
    Args:
        text: Document text content
        config: Classifier configuration
        
    Returns:
        CustomTypeMatch if a custom type matched, None otherwise
    """
    config = config or ClassifierConfig()
    
    if not config.custom_types_enabled:
        return None
    
    # Get registry and load from file if needed
    registry = get_custom_type_registry()
    
    if config.custom_types_file_path and len(registry) == 0:
        try:
            registry.load(config.custom_types_file_path)
        except (ValueError, FileNotFoundError):
            pass  # No file or invalid - continue with empty registry
    
    # Get custom types (optionally filtered by brokerage)
    custom_types = registry.list(
        brokerage_id=config.custom_types_brokerage_id,
        active_only=True,
    )
    
    if not custom_types:
        return None
    
    text_lower = text.lower()
    best_match: Optional[CustomTypeMatch] = None
    
    for custom_type in custom_types:
        matched_patterns: List[str] = []
        max_score = 0.0
        
        # Check each pattern
        for pattern, weight in custom_type.patterns:
            try:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matched_patterns.append(pattern)
                    max_score = max(max_score, weight)
            except re.error:
                # Skip invalid regex patterns
                continue
        
        if max_score > 0:
            # Apply min_confidence threshold from the custom type
            if max_score >= custom_type.min_confidence:
                if best_match is None or max_score > best_match.confidence:
                    best_match = CustomTypeMatch(
                        custom_type=custom_type,
                        confidence=max_score,
                        matched_patterns=matched_patterns,
                        reasoning=f"Matched {len(matched_patterns)} pattern(s) for custom type '{custom_type.name}'",
                    )
    
    return best_match


def classify_by_keywords(
    text: str,
    detected_title: Optional[str] = None,
    config: Optional[ClassifierConfig] = None,
) -> ClassificationResult:
    """
    Classify document using keyword pattern matching.
    
    Checks custom document types first if enabled and custom_types_priority is True.
    Then falls back to standard document type patterns.
    
    Args:
        text: Document text content
        detected_title: Title detected during splitting (if available)
        config: Classifier configuration
        
    Returns:
        ClassificationResult with keyword-based classification
    """
    config = config or ClassifierConfig()
    start_time = time.time()
    
    # Combine title and text for analysis
    full_text = ""
    if detected_title and config.include_detected_title:
        full_text = detected_title + "\n\n"
    full_text += text[:config.max_chars_for_classification]
    full_text_lower = full_text.lower()
    
    # Check custom types first if enabled and prioritized
    if config.custom_types_enabled and config.custom_types_priority:
        custom_match = match_custom_types(full_text, config)
        if custom_match and custom_match.confidence >= config.keyword_min_confidence:
            processing_time = (time.time() - start_time) * 1000
            return ClassificationResult(
                document_type=DocumentType.CUSTOM,
                confidence=custom_match.confidence,
                method="keyword_custom",
                custom_type_id=custom_match.custom_type.id,
                custom_type_name=custom_match.custom_type.name,
                reasoning=custom_match.reasoning,
                processing_time_ms=processing_time,
            )
    
    # Score each document type
    scores: Dict[DocumentType, float] = {}
    
    for doc_type, patterns in KEYWORD_PATTERNS.items():
        max_score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, full_text_lower, re.IGNORECASE):
                max_score = max(max_score, weight)
        if max_score > 0:
            scores[doc_type] = max_score
    
    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    processing_time = (time.time() - start_time) * 1000
    
    # If no standard match, check custom types as fallback
    if not sorted_scores and config.custom_types_enabled and not config.custom_types_priority:
        custom_match = match_custom_types(full_text, config)
        if custom_match:
            return ClassificationResult(
                document_type=DocumentType.CUSTOM,
                confidence=custom_match.confidence,
                method="keyword_custom",
                custom_type_id=custom_match.custom_type.id,
                custom_type_name=custom_match.custom_type.name,
                reasoning=custom_match.reasoning,
                processing_time_ms=processing_time,
            )
    
    if not sorted_scores:
        return ClassificationResult(
            document_type=DocumentType.UNKNOWN,
            confidence=0.0,
            method="keyword",
            reasoning="No keyword patterns matched",
            processing_time_ms=processing_time,
        )
    
    best_type, best_score = sorted_scores[0]
    
    # Get alternatives (top 3 excluding best)
    alternatives = sorted_scores[1:4]
    
    return ClassificationResult(
        document_type=best_type,
        confidence=best_score,
        method="keyword",
        alternative_types=alternatives,
        reasoning=f"Matched keyword patterns for {best_type.value}",
        processing_time_ms=processing_time,
    )


# ============================================================================
# LLM-Based Classification
# ============================================================================

# Check for LLM availability
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    # Type stubs for when imports fail - these are never actually called
    # because LLM_AVAILABLE will be False
    ChatOpenAI = None  # type: ignore
    ChatAnthropic = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore


CLASSIFICATION_SYSTEM_PROMPT = """You are a real estate document classifier. Your job is to identify the type of real estate document based on its content.

Analyze the document text and classify it into ONE of these categories:

CONTRACTS:
- Buy-Sell Agreement: Primary purchase agreement between buyer and seller
- Purchase Agreement: Alternative name for buy-sell agreement
- Listing Agreement: Agreement between seller and listing agent
- Counter Offer: Response to an offer with modified terms
- Amendment: Modification to an existing contract
- Addendum: Additional terms added to a contract

DISCLOSURES:
- Seller Disclosure: Seller's property condition disclosure
- Property Disclosure: General property condition disclosure  
- Lead Paint Disclosure: Lead-based paint hazard disclosure (pre-1978 homes)
- Mold Disclosure: Mold-related disclosures
- Agency Disclosure: Real estate agency relationship disclosure

INSPECTIONS:
- Inspection Report: Home/property inspection findings
- Inspection Response: Buyer's response to inspection findings
- Appraisal: Property value appraisal report

CLOSING:
- Closing Statement: Settlement/HUD-1 statement
- Title Commitment: Title insurance commitment
- Deed: Property deed/transfer document

FINANCING:
- Loan Estimate: Lender's loan terms estimate
- Closing Disclosure (Loan): Final loan terms disclosure
- Pre-Approval Letter: Mortgage pre-approval

OTHER:
- Wire Instructions: Wire transfer payment instructions
- Earnest Money Receipt: Receipt for earnest money deposit
- Buyer Agency Agreement: Buyer representation agreement
- Unknown: Cannot determine document type

Respond in JSON format:
{
    "document_type": "<type name exactly as listed above>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}"""


def classify_with_llm(
    text: str,
    detected_title: Optional[str] = None,
    config: Optional[ClassifierConfig] = None,
) -> Optional[ClassificationResult]:
    """
    Classify document using LLM.
    
    Args:
        text: Document text content
        detected_title: Title detected during splitting
        config: Classifier configuration
        
    Returns:
        ClassificationResult or None if LLM unavailable/fails
    """
    config = config or ClassifierConfig()
    
    if not LLM_AVAILABLE:
        logger.warning("LLM packages not available for classification")
        return None
    
    if not config.use_llm:
        return None
    
    start_time = time.time()
    
    # Build the prompt
    content_preview = text[:config.max_chars_for_classification]
    
    user_prompt = "Classify this real estate document:\n\n"
    if detected_title:
        user_prompt += f"DETECTED TITLE: {detected_title}\n\n"
    user_prompt += f"DOCUMENT CONTENT:\n{content_preview}"
    
    try:
        # Initialize LLM based on provider
        if config.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set")
                return None
            
            if ChatOpenAI is None:
                logger.warning("ChatOpenAI not available")
                return None
            
            llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_completion_tokens=config.llm_max_tokens,
            )
        elif config.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set")
                return None
            
            if ChatAnthropic is None:
                logger.warning("ChatAnthropic not available")
                return None
            
            llm = ChatAnthropic(  # type: ignore[call-arg]
                model_name=config.llm_model,
                temperature=config.llm_temperature,
            )
        else:
            logger.warning(f"Unknown LLM provider: {config.llm_provider}")
            return None
        
        if SystemMessage is None or HumanMessage is None:
            logger.warning("LangChain messages not available")
            return None
        
        # Call LLM
        messages = [
            SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        
        response = llm.invoke(messages)
        response_text: str = str(response.content)
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result_json = json.loads(response_text.strip())
        
        processing_time = (time.time() - start_time) * 1000
        
        # Parse document type
        doc_type = DocumentType.from_string(result_json.get("document_type", "Unknown"))
        confidence = float(result_json.get("confidence", 0.5))
        reasoning = result_json.get("reasoning", "")
        
        return ClassificationResult(
            document_type=doc_type,
            confidence=confidence,
            method="llm",
            reasoning=reasoning,
            processing_time_ms=processing_time,
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        return None


# ============================================================================
# Main Classification Function
# ============================================================================

def classify_document(
    text: str,
    detected_title: Optional[str] = None,
    config: Optional[ClassifierConfig] = None,
    ocr_info: Optional[OcrDocumentInfo] = None,
) -> ClassificationResult:
    """
    Classify a document using the best available method.
    
    Tries LLM first, falls back to keyword-based classification.
    Applies OCR quality adjustments when OCR metadata is available.
    
    Args:
        text: Document text content
        detected_title: Title detected during splitting
        config: Classifier configuration
        ocr_info: OCR metadata from splitter (optional)
        
    Returns:
        ClassificationResult with classification details
    """
    config = config or ClassifierConfig()
    
    # Process OCR metadata and potentially preprocess text
    quality_factor = 1.0
    processed_text = text
    
    if ocr_info and ocr_info.has_scanned_pages:
        processed_text, quality_factor = assess_text_quality_for_classification(
            text, ocr_info
        )
        
        # Log OCR quality issues
        if ocr_info.is_poor_quality:
            logger.warning(
                f"Document has poor OCR quality ({ocr_info.ocr_confidence:.1%}), "
                "classification confidence will be reduced"
            )
        elif ocr_info.is_low_confidence:
            logger.info(
                f"Document has low OCR confidence ({ocr_info.ocr_confidence:.1%})"
            )
    
    # Mock mode for testing
    if config.use_mock:
        result = _get_mock_classification(processed_text, detected_title)
        result.ocr_info = ocr_info
        result.ocr_quality_factor = quality_factor
        if quality_factor < 1.0:
            result.original_confidence = result.confidence
            result.confidence = result.confidence * quality_factor
        return result
    
    # Try LLM classification first
    if config.use_llm:
        llm_result = classify_with_llm(processed_text, detected_title, config)
        if llm_result and llm_result.confidence >= config.min_confidence_threshold:
            # Apply OCR quality factor
            llm_result.ocr_info = ocr_info
            llm_result.ocr_quality_factor = quality_factor
            if quality_factor < 1.0:
                llm_result.original_confidence = llm_result.confidence
                llm_result.confidence = llm_result.confidence * quality_factor
            
            logger.info(f"LLM classified as {llm_result.document_type.value} "
                       f"({llm_result.confidence:.1%} confidence)")
            return llm_result
    
    # Fall back to keyword classification
    if config.use_keyword_fallback:
        keyword_result = classify_by_keywords(processed_text, detected_title, config)
        
        # Apply OCR quality factor
        keyword_result.ocr_info = ocr_info
        keyword_result.ocr_quality_factor = quality_factor
        if quality_factor < 1.0:
            keyword_result.original_confidence = keyword_result.confidence
            keyword_result.confidence = keyword_result.confidence * quality_factor
        
        # If keyword confidence is acceptable, use it
        if keyword_result.confidence >= config.keyword_min_confidence:
            logger.info(f"Keyword classified as {keyword_result.document_type.value} "
                       f"({keyword_result.confidence:.1%} confidence)")
            return keyword_result
    
    # Last resort: try to use detected title
    if detected_title:
        title_type = DocumentType.from_string(detected_title)
        if title_type != DocumentType.UNKNOWN:
            adjusted_confidence = 0.75 * quality_factor
            return ClassificationResult(
                document_type=title_type,
                confidence=adjusted_confidence,
                method="title",
                reasoning=f"Inferred from document title: {detected_title}",
                ocr_info=ocr_info,
                ocr_quality_factor=quality_factor,
                original_confidence=0.75 if quality_factor < 1.0 else None,
            )
    
    # Cannot classify
    return ClassificationResult(
        document_type=DocumentType.UNKNOWN,
        confidence=0.0,
        method="none",
        reasoning="Could not determine document type",
        ocr_info=ocr_info,
        ocr_quality_factor=quality_factor,
    )


def _get_mock_classification(
    text: str,
    detected_title: Optional[str] = None,
) -> ClassificationResult:
    """
    Return mock classification for testing.
    
    Uses simple keyword matching similar to original implementation.
    """
    text_lower = (text or "").lower()
    title_lower = (detected_title or "").lower()
    combined = f"{title_lower} {text_lower}"
    
    if "buy" in combined and "sell" in combined:
        return ClassificationResult(
            document_type=DocumentType.BUY_SELL_AGREEMENT,
            confidence=0.95,
            method="mock",
            reasoning="Mock: Detected 'buy-sell' keywords",
        )
    elif "purchase" in combined and "agreement" in combined:
        return ClassificationResult(
            document_type=DocumentType.PURCHASE_AGREEMENT,
            confidence=0.92,
            method="mock",
            reasoning="Mock: Detected 'purchase agreement' keywords",
        )
    elif "disclosure" in combined:
        if "lead" in combined or "paint" in combined:
            return ClassificationResult(
                document_type=DocumentType.LEAD_PAINT_DISCLOSURE,
                confidence=0.93,
                method="mock",
                reasoning="Mock: Detected lead paint disclosure keywords",
            )
        elif "mold" in combined:
            return ClassificationResult(
                document_type=DocumentType.MOLD_DISCLOSURE,
                confidence=0.92,
                method="mock",
                reasoning="Mock: Detected mold disclosure keywords",
            )
        elif "seller" in combined:
            return ClassificationResult(
                document_type=DocumentType.SELLER_DISCLOSURE,
                confidence=0.93,
                method="mock",
                reasoning="Mock: Detected seller disclosure keywords",
            )
        else:
            return ClassificationResult(
                document_type=DocumentType.PROPERTY_DISCLOSURE,
                confidence=0.85,
                method="mock",
                reasoning="Mock: Detected generic disclosure keywords",
            )
    elif "counter" in combined and "offer" in combined:
        return ClassificationResult(
            document_type=DocumentType.COUNTER_OFFER,
            confidence=0.94,
            method="mock",
            reasoning="Mock: Detected counter offer keywords",
        )
    elif "addendum" in combined:
        return ClassificationResult(
            document_type=DocumentType.ADDENDUM,
            confidence=0.91,
            method="mock",
            reasoning="Mock: Detected addendum keyword",
        )
    elif "amendment" in combined:
        return ClassificationResult(
            document_type=DocumentType.AMENDMENT,
            confidence=0.91,
            method="mock",
            reasoning="Mock: Detected amendment keyword",
        )
    elif "inspection" in combined:
        return ClassificationResult(
            document_type=DocumentType.INSPECTION_REPORT,
            confidence=0.88,
            method="mock",
            reasoning="Mock: Detected inspection keywords",
        )
    else:
        return ClassificationResult(
            document_type=DocumentType.UNKNOWN,
            confidence=0.0,
            method="mock",
            reasoning="Mock: No matching keywords found",
        )


# ============================================================================
# Classifier Metrics
# ============================================================================

@dataclass
class ClassifierMetrics:
    """Performance metrics for classification operations."""
    
    total_documents: int = 0
    documents_classified: int = 0
    documents_unknown: int = 0
    
    llm_classifications: int = 0
    keyword_classifications: int = 0
    title_classifications: int = 0
    mock_classifications: int = 0
    
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    
    high_confidence_count: int = 0  # >= 90%
    low_confidence_count: int = 0   # < 70%
    
    total_processing_time_ms: float = 0.0
    
    # OCR metrics (C2)
    ocr_documents_count: int = 0  # Documents with scanned pages
    ocr_accuracy_met_count: int = 0  # Documents meeting 95% OCR accuracy
    ocr_low_confidence_count: int = 0  # Documents with low OCR confidence
    ocr_poor_quality_count: int = 0  # Documents with poor OCR quality
    ocr_impacted_classifications: int = 0  # Classifications penalized by OCR quality
    avg_ocr_confidence: float = 0.0  # Average OCR confidence for scanned docs
    ocr_needs_review_count: int = 0  # Documents flagged for OCR quality review
    
    # Review queue metrics (C3)
    review_flagged_count: int = 0  # Total documents flagged for review
    review_by_reason: Dict[str, int] = field(default_factory=dict)  # Count by reason
    review_by_priority: Dict[str, int] = field(default_factory=dict)  # Count by priority
    avg_flagged_confidence: float = 0.0  # Average confidence of flagged docs
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "documents_classified": self.documents_classified,
            "documents_unknown": self.documents_unknown,
            "llm_classifications": self.llm_classifications,
            "keyword_classifications": self.keyword_classifications,
            "title_classifications": self.title_classifications,
            "mock_classifications": self.mock_classifications,
            "avg_confidence": round(self.avg_confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "max_confidence": round(self.max_confidence, 4),
            "high_confidence_count": self.high_confidence_count,
            "low_confidence_count": self.low_confidence_count,
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            # OCR metrics
            "ocr_documents_count": self.ocr_documents_count,
            "ocr_accuracy_met_count": self.ocr_accuracy_met_count,
            "ocr_low_confidence_count": self.ocr_low_confidence_count,
            "ocr_poor_quality_count": self.ocr_poor_quality_count,
            "ocr_impacted_classifications": self.ocr_impacted_classifications,
            "avg_ocr_confidence": round(self.avg_ocr_confidence, 4),
            "ocr_needs_review_count": self.ocr_needs_review_count,
            # Review queue metrics (C3)
            "review_flagged_count": self.review_flagged_count,
            "review_by_reason": self.review_by_reason,
            "review_by_priority": self.review_by_priority,
            "avg_flagged_confidence": round(self.avg_flagged_confidence, 4),
            "timestamp": self.timestamp,
        }


# ============================================================================
# Main Node Function
# ============================================================================

def doc_type_classifier_node(state: DealState) -> dict:
    """
    Node C: Document Classifier
    
    Identifies the type of each sub-document using LLM/ML classification.
    
    User Story C1: Documents automatically classified by type
    with >90% accuracy target.
    
    User Story C2: Scanned/image PDFs converted to searchable text via OCR
    with >95% accuracy target. OCR quality impacts classification confidence.
    
    User Story C3: Low-confidence classifications flagged for human review
    with configurable confidence threshold (default <80%).
    
    Process:
    1. For each split document
    2. Extract OCR metadata from splitter output
    3. Preprocess OCR'd text if needed
    4. Attempt LLM classification (if available)
    5. Fall back to keyword-based classification
    6. Apply OCR quality adjustments to confidence
    7. Check if review is needed (C3)
    8. Assign document type and suggested folder
    9. Track classification, OCR, and review metrics
    
    Returns:
        dict with classified split_docs, review_queue, and metrics
    """
    print("--- NODE: Classifier ---")
    
    start_time = time.time()
    
    docs = state.get("split_docs", [])
    
    # Build configuration
    config = ClassifierConfig(
        use_llm=os.getenv("CLASSIFIER_USE_LLM", "true").lower() == "true",
        llm_provider=os.getenv("CLASSIFIER_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("CLASSIFIER_LLM_MODEL", "gpt-4o-mini"),
        use_mock=os.getenv("USE_MOCK_CLASSIFIER", "true").lower() == "true",
        review_confidence_threshold=float(os.getenv("REVIEW_CONFIDENCE_THRESHOLD", "0.80")),
        enable_review_queue=os.getenv("ENABLE_REVIEW_QUEUE", "true").lower() == "true",
    )
    
    # Initialize metrics
    metrics = ClassifierMetrics(total_documents=len(docs))
    confidence_sum = 0.0
    ocr_confidence_sum = 0.0
    flagged_confidence_sum = 0.0
    
    classified_docs = []
    review_queue: List[Dict[str, Any]] = []
    
    for doc in docs:
        content = doc.get("raw_text", "")
        detected_title = doc.get("detected_title")
        doc_id = doc.get("id", 0)
        
        # Extract OCR metadata from splitter output (C2)
        ocr_info = OcrDocumentInfo.from_document(doc)
        
        # Classify the document with OCR awareness
        result = classify_document(content, detected_title, config, ocr_info)
        
        # Check if review is needed (C3)
        review_flag = None
        if config.enable_review_queue:
            review_flag = check_needs_review(result, doc_id, config)
        
        # Update document with classification
        doc["type"] = result.document_type.value
        doc["classification_confidence"] = result.confidence
        doc["classification_method"] = result.method
        doc["suggested_folder"] = result.suggested_folder
        doc["classification_reasoning"] = result.reasoning
        doc["alternative_types"] = [
            {"type": t.value, "confidence": c}
            for t, c in result.alternative_types
        ]
        
        # Add OCR-related classification info
        if result.ocr_impacted:
            doc["ocr_quality_factor"] = result.ocr_quality_factor
            doc["original_confidence"] = result.original_confidence
        if ocr_info.has_scanned_pages:
            doc["ocr_needs_review"] = ocr_info.needs_review
        
        # Add review flag info to document (C3)
        if review_flag:
            doc["needs_review"] = True
            doc["review_reasons"] = [r.value for r in review_flag.reasons]
            doc["review_priority"] = review_flag.priority
            review_queue.append(review_flag.to_dict())
            
            # Update review metrics
            metrics.review_flagged_count += 1
            flagged_confidence_sum += result.confidence
            
            # Track by reason
            for reason in review_flag.reasons:
                reason_key = reason.value
                metrics.review_by_reason[reason_key] = \
                    metrics.review_by_reason.get(reason_key, 0) + 1
            
            # Track by priority
            metrics.review_by_priority[review_flag.priority] = \
                metrics.review_by_priority.get(review_flag.priority, 0) + 1
        else:
            doc["needs_review"] = False
        
        classified_docs.append(doc)
        
        # Update classification metrics
        if result.document_type != DocumentType.UNKNOWN:
            metrics.documents_classified += 1
        else:
            metrics.documents_unknown += 1
        
        # Track method counts
        if result.method == "llm":
            metrics.llm_classifications += 1
        elif result.method == "keyword":
            metrics.keyword_classifications += 1
        elif result.method == "title":
            metrics.title_classifications += 1
        elif result.method == "mock":
            metrics.mock_classifications += 1
        
        # Track confidence stats
        confidence_sum += result.confidence
        metrics.min_confidence = min(metrics.min_confidence, result.confidence)
        metrics.max_confidence = max(metrics.max_confidence, result.confidence)
        
        if result.confidence >= 0.90:
            metrics.high_confidence_count += 1
        elif result.confidence < 0.70:
            metrics.low_confidence_count += 1
        
        # Track OCR metrics (C2)
        if ocr_info.has_scanned_pages:
            metrics.ocr_documents_count += 1
            ocr_confidence_sum += ocr_info.ocr_confidence
            
            if ocr_info.meets_accuracy_target:
                metrics.ocr_accuracy_met_count += 1
            if ocr_info.is_low_confidence:
                metrics.ocr_low_confidence_count += 1
            if ocr_info.is_poor_quality:
                metrics.ocr_poor_quality_count += 1
            if ocr_info.needs_review:
                metrics.ocr_needs_review_count += 1
        
        if result.ocr_impacted:
            metrics.ocr_impacted_classifications += 1
        
        # Log classification
        confidence_pct = f"{result.confidence:.0%}"
        folder_info = f" → {result.suggested_folder}" if result.suggested_folder else ""
        ocr_note = ""
        if ocr_info.has_scanned_pages:
            if ocr_info.needs_review:
                ocr_note = f" [OCR: {ocr_info.ocr_confidence:.0%}⚠️]"
            else:
                ocr_note = f" [OCR: {ocr_info.ocr_confidence:.0%}✓]"
        review_note = " 🔍" if review_flag else ""
        
        print(f"   Doc {doc.get('id', '?')}: {result.document_type.value} "
              f"({confidence_pct}, {result.method}){folder_info}{ocr_note}{review_note}")
    
    # Finalize metrics
    if docs:
        metrics.avg_confidence = confidence_sum / len(docs)
    if metrics.ocr_documents_count > 0:
        metrics.avg_ocr_confidence = ocr_confidence_sum / metrics.ocr_documents_count
    if metrics.review_flagged_count > 0:
        metrics.avg_flagged_confidence = flagged_confidence_sum / metrics.review_flagged_count
    metrics.total_processing_time_ms = (time.time() - start_time) * 1000
    
    # Log summary
    print(f"   Classified {metrics.documents_classified}/{metrics.total_documents} documents")
    if metrics.documents_unknown > 0:
        print(f"   ⚠️  {metrics.documents_unknown} document(s) could not be classified")
    if metrics.high_confidence_count > 0:
        print(f"   ✓ {metrics.high_confidence_count} high-confidence (≥90%) classifications")
    
    # Log OCR summary (C2)
    if metrics.ocr_documents_count > 0:
        print(f"   📄 {metrics.ocr_documents_count} document(s) contained scanned pages")
        if metrics.ocr_accuracy_met_count > 0:
            print(f"   ✓ {metrics.ocr_accuracy_met_count} met OCR accuracy target (≥95%)")
        if metrics.ocr_needs_review_count > 0:
            print(f"   ⚠️  {metrics.ocr_needs_review_count} flagged for OCR quality review")
    
    # Log review queue summary (C3)
    if metrics.review_flagged_count > 0:
        print(f"   🔍 {metrics.review_flagged_count} document(s) flagged for human review")
        if metrics.review_by_priority:
            priority_summary = ", ".join(
                f"{count} {priority}" 
                for priority, count in sorted(metrics.review_by_priority.items())
            )
            print(f"      Priority: {priority_summary}")
    
    return {
        "split_docs": classified_docs,
        "review_queue": review_queue,
        "classifier_metrics": metrics.to_dict(),
    }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    """Test the classifier standalone with OCR scenarios."""
    
    # Test with mock documents including OCR scenarios
    test_state: DealState = {
        "deal_id": "test-001",
        "status": "",
        "email_metadata": {},
        "raw_pdf_path": "",
        "split_docs": [
            {
                "id": 1,
                "page_range": [1, 4],
                "raw_text": """RESIDENTIAL REAL ESTATE BUY-SELL AGREEMENT
                
                Property Address: 123 Main St
                Purchase Price: $350,000
                Buyer agrees to purchase the property...""",
                "detected_title": "RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT",
                # No OCR - native PDF
                "has_scanned_pages": False,
            },
            {
                "id": 2,
                "page_range": [5, 6],
                "raw_text": """SELLER'S PROPERTY DISCLOSURE STATEMENT
                
                The Seller makes the following disclosures...
                MOLD DISCLOSURE: Seller is not aware of any mold issues.""",
                "detected_title": "SELLER'S PROPERTY DISCLOSURE STATEMENT",
                # High quality OCR
                "has_scanned_pages": True,
                "ocr_confidence": 0.98,
                "meets_accuracy_target": True,
            },
            {
                "id": 3,
                "page_range": [7, 8],
                "raw_text": """LEAD PAINT DISCLOSURE
                
                This property was built before 1978. The following
                information is provided regarding lead-based paint...""",
                "detected_title": "Lead Paint Disclosure",
                # Low quality OCR
                "has_scanned_pages": True,
                "ocr_confidence": 0.65,
                "meets_accuracy_target": False,
            },
            {
                "id": 4,
                "page_range": [9, 10],
                "raw_text": """CQUNTER QFFER  # Simulated OCR errors
                
                Buycr proposes the following changes to the offcr...""",
                "detected_title": "COUNTER OFFER",
                # Poor quality OCR with errors
                "has_scanned_pages": True,
                "ocr_confidence": 0.45,
                "meets_accuracy_target": False,
            },
        ],
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
    
    result = doc_type_classifier_node(test_state)
    
    print("\n--- Classification Result ---")
    for doc in result["split_docs"]:
        print(f"\n  Doc {doc['id']}: {doc['type']}")
        print(f"    Confidence: {doc['classification_confidence']:.0%}")
        print(f"    Folder: {doc['suggested_folder']}")
        print(f"    Method: {doc['classification_method']}")
        if doc.get("has_scanned_pages"):
            print(f"    OCR: Yes (confidence: {doc.get('ocr_confidence', 0):.0%})")
            if doc.get("ocr_needs_review"):
                print(f"    ⚠️  Flagged for OCR review")
            if doc.get("ocr_quality_factor"):
                print(f"    Quality factor: {doc['ocr_quality_factor']:.2f}")
    
    print("\n--- Metrics ---")
    metrics = result["classifier_metrics"]
    print(f"  Total documents: {metrics['total_documents']}")
    print(f"  Classified: {metrics['documents_classified']}")
    print(f"  OCR documents: {metrics['ocr_documents_count']}")
    print(f"  OCR accuracy met: {metrics['ocr_accuracy_met_count']}")
    print(f"  OCR needs review: {metrics['ocr_needs_review_count']}")
