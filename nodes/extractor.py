"""
Extractor Node - Document Data Extraction & Signature Detection

This module handles:
1. Extracting structured data from real estate documents
2. Detecting signature fields and their locations
3. Mapping signature fields to Dotloop participant roles
4. Building Dotloop-compatible payloads

E1: Enhanced participant extraction with >95% accuracy target
- Full names with confidence scoring
- Email validation and extraction
- Phone number normalization
- Role inference from context
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from state import (
    DealState, 
    SignatureField, 
    ParticipantInfo,
    PropertyAddress,
    FinancialDetails,
    ContractDates,
    DocumentWithSignatures
)


# ============================================================================
# E1: Participant Extraction - Constants & Patterns
# ============================================================================

# Confidence thresholds
EXTRACTION_CONFIDENCE_THRESHOLD = 0.80  # 80% minimum for auto-acceptance
HIGH_CONFIDENCE_THRESHOLD = 0.95  # 95% for no-review extraction


class ExtractionConfidence(Enum):
    """Confidence levels for extracted data."""
    HIGH = "high"  # >= 95% - auto-accept
    MEDIUM = "medium"  # 80-94% - accept with flag
    LOW = "low"  # < 80% - needs review


@dataclass
class ExtractedValue:
    """A value extracted from text with confidence metadata."""
    value: str
    confidence: float
    source_text: str
    extraction_method: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    
    def confidence_level(self) -> ExtractionConfidence:
        """Get the confidence level category."""
        if self.confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return ExtractionConfidence.HIGH
        elif self.confidence >= EXTRACTION_CONFIDENCE_THRESHOLD:
            return ExtractionConfidence.MEDIUM
        return ExtractionConfidence.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level().value,
            "source_text": self.source_text,
            "extraction_method": self.extraction_method,
        }


@dataclass
class ExtractedParticipant:
    """
    A participant extracted from document text with confidence scores.
    
    This is the internal extraction result before converting to ParticipantInfo.
    """
    full_name: ExtractedValue
    role: str
    email: Optional[ExtractedValue] = None
    phone: Optional[ExtractedValue] = None
    company_name: Optional[ExtractedValue] = None
    license_number: Optional[ExtractedValue] = None
    
    # Address fields
    street_address: Optional[ExtractedValue] = None
    city: Optional[ExtractedValue] = None
    state: Optional[ExtractedValue] = None
    zip_code: Optional[ExtractedValue] = None
    
    def overall_confidence(self) -> float:
        """Calculate overall extraction confidence."""
        scores = [self.full_name.confidence]
        if self.email:
            scores.append(self.email.confidence)
        if self.phone:
            scores.append(self.phone.confidence)
        return sum(scores) / len(scores) if scores else 0.0
    
    def to_participant_info(self) -> ParticipantInfo:
        """Convert to ParticipantInfo TypedDict."""
        info: ParticipantInfo = {
            "full_name": self.full_name.value,
            "role": self.role,
            "email": self.email.value if self.email else None,
            "phone": self.phone.value if self.phone else None,
        }
        if self.company_name:
            info["company_name"] = self.company_name.value
        if self.license_number:
            info["license_number"] = self.license_number.value
        if self.city:
            info["city"] = self.city.value
        if self.state:
            info["state"] = self.state.value
        if self.zip_code:
            info["zip_code"] = self.zip_code.value
        return info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with confidence metadata."""
        return {
            "full_name": self.full_name.to_dict(),
            "role": self.role,
            "email": self.email.to_dict() if self.email else None,
            "phone": self.phone.to_dict() if self.phone else None,
            "company_name": self.company_name.to_dict() if self.company_name else None,
            "overall_confidence": self.overall_confidence(),
        }


# Email validation pattern (RFC 5322 simplified)
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE
)

# Phone number patterns (US formats)
PHONE_PATTERNS = [
    # (123) 456-7890
    re.compile(r"\((\d{3})\)\s*(\d{3})[-.\s]?(\d{4})"),
    # 123-456-7890 or 123.456.7890
    re.compile(r"(\d{3})[-.](\d{3})[-.](\d{4})"),
    # 1234567890
    re.compile(r"(?<!\d)(\d{3})(\d{3})(\d{4})(?!\d)"),
    # +1 123 456 7890 or 1-123-456-7890
    re.compile(r"(?:\+?1[-.\s]?)?(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})"),
]

# Name patterns for real estate participants
# Matches: "John Smith", "Mary Jane Smith", "John Q. Public III"
# Uses [^\S\n] to match whitespace except newlines
NAME_PATTERN = re.compile(
    r"([A-Z][a-z]+(?:[^\S\n]+[A-Z]\.?)?(?:[^\S\n]+[A-Z][a-z]+)+(?:[^\S\n]+(?:Jr\.?|Sr\.?|II|III|IV))?)",
    re.UNICODE
)

# Common name prefixes and suffixes to validate names
NAME_PREFIXES = {"Mr", "Mrs", "Ms", "Dr", "Prof"}
NAME_SUFFIXES = {"Jr", "Sr", "II", "III", "IV", "Esq", "PhD", "MD"}

# Words that should NOT be considered names
NAME_EXCLUSIONS = {
    "purchase", "agreement", "contract", "property", "address", "street",
    "disclosure", "real", "estate", "transaction", "buyer", "seller",
    "agent", "broker", "listing", "closing", "date", "price", "earnest",
    "money", "deposit", "inspection", "title", "escrow", "loan", "mortgage",
    "county", "state", "city", "zip", "code", "phone", "email", "fax",
    "signature", "initial", "date", "print", "name", "hereby", "agrees",
    "represents", "warrants", "acknowledges", "hereinafter", "referred",
    "collectively", "individually", "property", "located", "situated",
    "legal", "description", "parcel", "assessor", "tax", "mls", "number",
    "the", "and", "for", "this", "that", "with", "from", "have", "been",
}

# Role detection patterns with context
PARTICIPANT_ROLE_PATTERNS = {
    "BUYER": [
        r"(?i)buyer[:\s]+",
        r"(?i)purchaser[:\s]+",
        r"(?i)buyer\s*(?:\d|one|two|1|2)?[:\s]+",
        r"(?i)purchasing\s+party[:\s]+",
        r"(?i)grantee[:\s]+",
    ],
    "SELLER": [
        r"(?i)seller[:\s]+",
        r"(?i)vendor[:\s]+",
        r"(?i)seller\s*(?:\d|one|two|1|2)?[:\s]+",
        r"(?i)selling\s+party[:\s]+",
        r"(?i)grantor[:\s]+",
        r"(?i)owner[:\s]+",
    ],
    "LISTING_AGENT": [
        r"(?i)listing\s+agent[:\s]+",
        r"(?i)seller'?s?\s+agent[:\s]+",
        r"(?i)listing\s+representative[:\s]+",
    ],
    "BUYING_AGENT": [
        r"(?i)buyer'?s?\s+agent[:\s]+",
        r"(?i)selling\s+agent[:\s]+",
        r"(?i)cooperating\s+agent[:\s]+",
        r"(?i)buyer\s+representative[:\s]+",
    ],
    "LISTING_BROKER": [
        r"(?i)listing\s+broker[:\s]+",
        r"(?i)seller'?s?\s+broker[:\s]+",
        r"(?i)listing\s+brokerage[:\s]+",
    ],
    "BUYING_BROKER": [
        r"(?i)buyer'?s?\s+broker[:\s]+",
        r"(?i)selling\s+broker[:\s]+",
        r"(?i)cooperating\s+broker[:\s]+",
    ],
    "ESCROW_TITLE_REP": [
        r"(?i)escrow\s+(?:officer|agent)[:\s]+",
        r"(?i)title\s+(?:officer|agent|rep)[:\s]+",
        r"(?i)closing\s+agent[:\s]+",
    ],
    "LOAN_OFFICER": [
        r"(?i)loan\s+officer[:\s]+",
        r"(?i)mortgage\s+(?:officer|broker)[:\s]+",
        r"(?i)lender\s+representative[:\s]+",
    ],
}

# Labeled field patterns (e.g., "Name: John Smith", "Email: john@example.com")
LABELED_FIELD_PATTERNS = {
    "name": [
        r"(?i)(?:full\s+)?name[:\s]+([^\n,]+)",
        r"(?i)print(?:ed)?\s+name[:\s]+([^\n,]+)",
        r"(?i)(?:buyer|seller|agent|broker)\s+name[:\s]+([^\n,]+)",
    ],
    "email": [
        r"(?i)e-?mail(?:\s+address)?[:\s]+([^\s,]+@[^\s,]+)",
        r"(?i)email[:\s]+(\S+@\S+)",
    ],
    "phone": [
        r"(?i)(?:phone|tel(?:ephone)?|cell|mobile)[:\s#]+([^\n,]+)",
        r"(?i)(?:ph|tel)[:\s#]+([^\n,]+)",
    ],
    "company": [
        r"(?i)(?:company|firm|brokerage|agency)[:\s]+([^\n,]+)",
        r"(?i)(?:representing|affiliated\s+with)[:\s]+([^\n,]+)",
    ],
    "license": [
        r"(?i)(?:license|lic)\.?\s*(?:#|no\.?|number)?[:\s]+(\S+)",
        r"(?i)(?:bre|dre|re)\s*(?:#|no\.?)?[:\s]+(\S+)",
    ],
}


# ============================================================================
# E1: Participant Extraction Functions
# ============================================================================

def validate_email(email: str) -> Tuple[bool, float]:
    """
    Validate an email address and return confidence score.
    
    Returns:
        Tuple of (is_valid, confidence_score)
    """
    if not email:
        return False, 0.0
    
    email = email.strip().lower()
    
    # Basic pattern match
    if not EMAIL_PATTERN.fullmatch(email):
        return False, 0.0
    
    confidence = 0.90  # Base confidence for pattern match
    
    # Check for common real estate domains (higher confidence)
    trusted_domains = [
        "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "aol.com", "icloud.com", "me.com",
        "kw.com", "remax.com", "century21.com", "coldwellbanker.com",
        "berkshirehathaway.com", "compass.com", "sothebysrealty.com",
    ]
    domain = email.split("@")[1] if "@" in email else ""
    if any(domain.endswith(td) for td in trusted_domains):
        confidence = 0.98
    
    # Check for suspicious patterns
    if email.count("@") != 1:
        return False, 0.0
    if ".." in email:
        return False, 0.0
    if len(email) < 5:
        return False, 0.0
    if len(email) > 254:
        return False, 0.0
    
    # Check local part length
    local_part = email.split("@")[0]
    if len(local_part) > 64:
        return False, 0.0
    
    return True, confidence


def normalize_phone(phone: str) -> Tuple[Optional[str], float]:
    """
    Normalize a phone number to standard format and return confidence.
    
    Returns:
        Tuple of (normalized_phone, confidence_score)
        Returns (None, 0.0) if not a valid phone number.
    """
    if not phone:
        return None, 0.0
    
    # Remove common non-digit characters except for validation
    cleaned = re.sub(r"[^\d]", "", phone)
    
    # Handle country code
    if len(cleaned) == 11 and cleaned.startswith("1"):
        cleaned = cleaned[1:]
    
    # Must be 10 digits for US phone
    if len(cleaned) != 10:
        return None, 0.0
    
    # Check for invalid area codes (can't start with 0 or 1)
    if cleaned[0] in "01":
        return None, 0.0
    
    # Check for invalid exchange codes (can't start with 0 or 1)
    if cleaned[3] in "01":
        return None, 0.0
    
    # Format as (XXX) XXX-XXXX
    normalized = f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
    
    # Calculate confidence based on original format
    confidence = 0.85  # Base confidence
    
    # Higher confidence for properly formatted numbers
    for pattern in PHONE_PATTERNS:
        if pattern.search(phone):
            confidence = 0.95
            break
    
    return normalized, confidence


def validate_name(name: str) -> Tuple[bool, float]:
    """
    Validate a person's name and return confidence score.
    
    Returns:
        Tuple of (is_valid, confidence_score)
    """
    if not name:
        return False, 0.0
    
    name = name.strip()
    
    # Must have at least first and last name
    parts = name.split()
    if len(parts) < 2:
        return False, 0.0
    
    # Check for excluded words (contract terms, etc.)
    name_lower = name.lower()
    for exclusion in NAME_EXCLUSIONS:
        if exclusion in name_lower.split():
            return False, 0.0
    
    # Check for suspicious patterns
    if re.search(r"\d{3,}", name):  # Too many consecutive digits
        return False, 0.0
    if re.search(r"[@#$%^&*()+=\[\]{}|\\/<>]", name):  # Special chars
        return False, 0.0
    if len(name) > 100:  # Too long
        return False, 0.0
    if len(name) < 3:  # Too short
        return False, 0.0
    
    confidence = 0.80  # Base confidence
    
    # Higher confidence for standard name patterns
    if NAME_PATTERN.fullmatch(name):
        confidence = 0.95
    
    # Check if first letter of each part is uppercase
    if all(part[0].isupper() for part in parts if part):
        confidence += 0.02
    
    # Penalize all caps or all lowercase
    if name.isupper() or name.islower():
        confidence -= 0.10
    
    # Validate each part looks like a name
    for part in parts:
        # Allow initials (single letter with optional period)
        if len(part) <= 2 and part.rstrip(".").isalpha():
            continue
        # Allow suffixes
        if part.rstrip(".") in NAME_SUFFIXES:
            continue
        # Allow prefixes
        if part.rstrip(".") in NAME_PREFIXES:
            continue
        # Must start with letter and be mostly alpha
        if not part[0].isalpha():
            confidence -= 0.10
        if not re.match(r"^[A-Za-z'-]+$", part):
            confidence -= 0.05
    
    return confidence >= 0.50, max(0.0, min(1.0, confidence))


def extract_email_from_text(text: str) -> List[ExtractedValue]:
    """
    Extract all email addresses from text with confidence scores.
    """
    emails: List[ExtractedValue] = []
    seen: Set[str] = set()
    
    # First, try labeled patterns (higher confidence)
    for pattern in LABELED_FIELD_PATTERNS["email"]:
        for match in re.finditer(pattern, text):
            email = match.group(1).strip()
            email_lower = email.lower()
            if email_lower in seen:
                continue
            is_valid, confidence = validate_email(email)
            if is_valid:
                seen.add(email_lower)
                emails.append(ExtractedValue(
                    value=email_lower,
                    confidence=min(confidence + 0.03, 1.0),  # Bonus for labeled
                    source_text=match.group(0),
                    extraction_method="labeled_pattern",
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
    
    # Then try general pattern matching
    for match in EMAIL_PATTERN.finditer(text):
        email = match.group(0).strip()
        email_lower = email.lower()
        if email_lower in seen:
            continue
        is_valid, confidence = validate_email(email)
        if is_valid:
            seen.add(email_lower)
            emails.append(ExtractedValue(
                value=email_lower,
                confidence=confidence,
                source_text=match.group(0),
                extraction_method="pattern_match",
                start_pos=match.start(),
                end_pos=match.end(),
            ))
    
    return emails


def extract_phone_from_text(text: str) -> List[ExtractedValue]:
    """
    Extract all phone numbers from text with confidence scores.
    """
    phones: List[ExtractedValue] = []
    seen: Set[str] = set()
    
    # First, try labeled patterns (higher confidence)
    for pattern in LABELED_FIELD_PATTERNS["phone"]:
        for match in re.finditer(pattern, text):
            phone_text = match.group(1).strip()
            normalized, confidence = normalize_phone(phone_text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                phones.append(ExtractedValue(
                    value=normalized,
                    confidence=min(confidence + 0.03, 1.0),  # Bonus for labeled
                    source_text=match.group(0),
                    extraction_method="labeled_pattern",
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
    
    # Then try direct pattern matching
    for phone_pattern in PHONE_PATTERNS:
        for match in phone_pattern.finditer(text):
            phone_text = match.group(0)
            normalized, confidence = normalize_phone(phone_text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                phones.append(ExtractedValue(
                    value=normalized,
                    confidence=confidence,
                    source_text=match.group(0),
                    extraction_method="pattern_match",
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
    
    return phones


def extract_names_from_text(text: str) -> List[ExtractedValue]:
    """
    Extract person names from text with confidence scores.
    """
    names: List[ExtractedValue] = []
    seen: Set[str] = set()
    
    # First, try labeled patterns (higher confidence)
    for pattern in LABELED_FIELD_PATTERNS["name"]:
        for match in re.finditer(pattern, text):
            name = match.group(1).strip()
            # Clean up the name
            name = re.sub(r"\s+", " ", name)
            name = name.strip(" ,:;")
            
            name_lower = name.lower()
            if name_lower in seen:
                continue
            
            is_valid, confidence = validate_name(name)
            if is_valid:
                seen.add(name_lower)
                names.append(ExtractedValue(
                    value=name,
                    confidence=min(confidence + 0.03, 1.0),  # Bonus for labeled
                    source_text=match.group(0),
                    extraction_method="labeled_pattern",
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
    
    # Then try general name pattern matching
    for match in NAME_PATTERN.finditer(text):
        name = match.group(0).strip()
        name_lower = name.lower()
        if name_lower in seen:
            continue
        
        is_valid, confidence = validate_name(name)
        if is_valid:
            seen.add(name_lower)
            names.append(ExtractedValue(
                value=name,
                confidence=confidence,
                source_text=match.group(0),
                extraction_method="pattern_match",
                start_pos=match.start(),
                end_pos=match.end(),
            ))
    
    return names


def extract_participant_by_role(
    text: str,
    role: str,
    patterns: List[str],
) -> List[ExtractedParticipant]:
    """
    Extract participants for a specific role using role-specific patterns.
    
    This looks for patterns like "Buyer: John Smith" and extracts
    the participant info following the role label.
    """
    participants: List[ExtractedParticipant] = []
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            # Get the text following the role label (next 500 chars or until double newline)
            start_pos = match.end()
            end_pos = min(start_pos + 500, len(text))
            following_text = text[start_pos:end_pos]
            
            # Stop at double newline or next section header
            section_end = re.search(r"\n\s*\n|\n[A-Z][A-Z\s]+:", following_text)
            if section_end:
                following_text = following_text[:section_end.start()]
            
            # Extract name from following text
            names = extract_names_from_text(following_text)
            if not names:
                continue
            
            # Take the first (highest confidence) name
            name = names[0]
            
            # Extract email and phone from the same context
            emails = extract_email_from_text(following_text)
            phones = extract_phone_from_text(following_text)
            
            participant = ExtractedParticipant(
                full_name=name,
                role=role,
                email=emails[0] if emails else None,
                phone=phones[0] if phones else None,
            )
            
            participants.append(participant)
    
    return participants


def extract_all_participants(text: str) -> List[ExtractedParticipant]:
    """
    Extract all participants from document text.
    
    Uses role-specific patterns to find buyers, sellers, agents, etc.
    Returns list of ExtractedParticipant with confidence scores.
    """
    participants: List[ExtractedParticipant] = []
    seen_names: Set[str] = set()
    
    # Extract participants by role
    for role, patterns in PARTICIPANT_ROLE_PATTERNS.items():
        role_participants = extract_participant_by_role(text, role, patterns)
        for p in role_participants:
            # Avoid duplicates
            name_key = p.full_name.value.lower()
            if name_key not in seen_names:
                seen_names.add(name_key)
                participants.append(p)
    
    return participants


def deduplicate_participants(
    participants: List[ExtractedParticipant]
) -> List[ExtractedParticipant]:
    """
    Deduplicate participants, merging information from multiple extractions.
    
    If the same name appears multiple times, keep the one with highest confidence
    and merge contact info.
    """
    by_name: Dict[str, ExtractedParticipant] = {}
    
    for p in participants:
        name_key = p.full_name.value.lower()
        
        if name_key not in by_name:
            by_name[name_key] = p
        else:
            existing = by_name[name_key]
            
            # Merge email if missing
            if not existing.email and p.email:
                existing.email = p.email
            elif existing.email and p.email:
                # Keep higher confidence email
                if p.email.confidence > existing.email.confidence:
                    existing.email = p.email
            
            # Merge phone if missing
            if not existing.phone and p.phone:
                existing.phone = p.phone
            elif existing.phone and p.phone:
                # Keep higher confidence phone
                if p.phone.confidence > existing.phone.confidence:
                    existing.phone = p.phone
            
            # Keep higher confidence name
            if p.full_name.confidence > existing.full_name.confidence:
                existing.full_name = p.full_name
    
    return list(by_name.values())


def filter_participants_by_confidence(
    participants: List[ExtractedParticipant],
    min_confidence: float = EXTRACTION_CONFIDENCE_THRESHOLD,
) -> Tuple[List[ExtractedParticipant], List[ExtractedParticipant]]:
    """
    Filter participants by confidence threshold.
    
    Returns:
        Tuple of (accepted_participants, needs_review_participants)
    """
    accepted: List[ExtractedParticipant] = []
    needs_review: List[ExtractedParticipant] = []
    
    for p in participants:
        if p.overall_confidence() >= min_confidence:
            accepted.append(p)
        else:
            needs_review.append(p)
    
    return accepted, needs_review


@dataclass
class ParticipantExtractionResult:
    """Result of participant extraction with metadata."""
    participants: List[ExtractedParticipant]
    needs_review: List[ExtractedParticipant]
    extraction_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_participant_info_list(self) -> List[ParticipantInfo]:
        """Convert all accepted participants to ParticipantInfo list."""
        return [p.to_participant_info() for p in self.participants]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "participants": [p.to_dict() for p in self.participants],
            "needs_review": [p.to_dict() for p in self.needs_review],
            "stats": self.extraction_stats,
        }


def extract_participants_with_confidence(
    text: str,
    min_confidence: float = EXTRACTION_CONFIDENCE_THRESHOLD,
) -> ParticipantExtractionResult:
    """
    Main entry point for participant extraction.
    
    Extracts all participants from text, deduplicates, and filters by confidence.
    
    Args:
        text: Document text to extract from
        min_confidence: Minimum confidence threshold for auto-acceptance
        
    Returns:
        ParticipantExtractionResult with accepted and needs-review participants
    """
    # Extract all participants
    raw_participants = extract_all_participants(text)
    
    # Deduplicate
    deduped = deduplicate_participants(raw_participants)
    
    # Filter by confidence
    accepted, needs_review = filter_participants_by_confidence(deduped, min_confidence)
    
    # Calculate stats
    stats = {
        "total_extracted": len(raw_participants),
        "after_dedup": len(deduped),
        "accepted": len(accepted),
        "needs_review": len(needs_review),
        "by_role": {},
    }
    
    for p in deduped:
        role = p.role
        if role not in stats["by_role"]:
            stats["by_role"][role] = 0
        stats["by_role"][role] += 1
    
    return ParticipantExtractionResult(
        participants=accepted,
        needs_review=needs_review,
        extraction_stats=stats,
    )


# ============================================================================
# E3: Financial Terms Extraction - Constants & Types
# ============================================================================

class FinancialFieldType(Enum):
    """Types of financial fields in real estate documents."""
    PURCHASE_PRICE = "purchase_price"
    EARNEST_MONEY = "earnest_money"
    DOWN_PAYMENT = "down_payment"
    LOAN_AMOUNT = "loan_amount"
    CLOSING_COSTS = "closing_costs"
    COMMISSION = "commission"
    COMMISSION_RATE = "commission_rate"
    DEPOSIT = "deposit"
    ESCROW = "escrow"
    PRORATION = "proration"
    CREDIT = "credit"
    OTHER = "other"


class FinancialConfidence(Enum):
    """Confidence levels for extracted financial values."""
    HIGH = "high"       # >= 0.90 - Clear label + valid format + reasonable range
    MEDIUM = "medium"   # >= 0.70 - Partial match or edge of range
    LOW = "low"         # < 0.70 - Uncertain extraction


# Financial validation ranges (in USD)
# These represent reasonable ranges for residential real estate
FINANCIAL_VALIDATION_RANGES: Dict[FinancialFieldType, Tuple[float, float]] = {
    FinancialFieldType.PURCHASE_PRICE: (10_000.0, 100_000_000.0),      # $10K - $100M
    FinancialFieldType.EARNEST_MONEY: (100.0, 1_000_000.0),            # $100 - $1M
    FinancialFieldType.DOWN_PAYMENT: (0.0, 50_000_000.0),              # $0 - $50M
    FinancialFieldType.LOAN_AMOUNT: (1_000.0, 50_000_000.0),           # $1K - $50M
    FinancialFieldType.CLOSING_COSTS: (0.0, 500_000.0),                # $0 - $500K
    FinancialFieldType.COMMISSION: (0.0, 5_000_000.0),                 # $0 - $5M
    FinancialFieldType.DEPOSIT: (0.0, 1_000_000.0),                    # $0 - $1M
    FinancialFieldType.ESCROW: (0.0, 1_000_000.0),                     # $0 - $1M
    FinancialFieldType.PRORATION: (0.0, 100_000.0),                    # $0 - $100K
    FinancialFieldType.CREDIT: (0.0, 100_000.0),                       # $0 - $100K
    FinancialFieldType.OTHER: (0.0, 100_000_000.0),                    # Wide range
}

# Percentage validation ranges
PERCENTAGE_VALIDATION_RANGES: Dict[FinancialFieldType, Tuple[float, float]] = {
    FinancialFieldType.COMMISSION_RATE: (0.0, 15.0),                   # 0% - 15%
    FinancialFieldType.DOWN_PAYMENT: (0.0, 100.0),                     # 0% - 100%
}

# Earnest money typically 1-5% of purchase price
EARNEST_MONEY_PERCENT_RANGE = (0.5, 10.0)

# Currency patterns for parsing
CURRENCY_PATTERN = re.compile(
    r'\$\s*([\d,]+(?:\.\d{1,2})?)\s*(?:(?:million|mil|m)\b)?|'  # $1,234.56 or $1.2M
    r'\$\s*([\d,]+(?:\.\d{1,2})?)\s*[Mm]|'                      # $1.5M
    r'([\d,]+(?:\.\d{1,2})?)\s*(?:dollars?|USD)\b|'            # 1,234.56 dollars
    r'(?:USD|US\$)\s*([\d,]+(?:\.\d{1,2})?)',                  # USD 1,234.56
    re.IGNORECASE
)

# Pattern for numbers with optional currency symbols
NUMBER_WITH_OPTIONAL_CURRENCY = re.compile(
    r'\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:(?:million|mil|m)\b)?',
    re.IGNORECASE
)

# Pattern for percentages
PERCENTAGE_PATTERN = re.compile(
    r'([\d.]+)\s*%|'                                            # 5.5%
    r'([\d.]+)\s*percent',                                      # 5.5 percent
    re.IGNORECASE
)


@dataclass
class ExtractedFinancialValue:
    """
    A financial value extracted from document text.
    
    Stores the parsed amount, original text, field type, and confidence.
    """
    amount: float
    field_type: FinancialFieldType
    confidence: float = 0.0
    source_text: str = ""
    extraction_method: str = "pattern_match"
    is_percentage: bool = False
    currency: str = "USD"
    
    @property
    def confidence_level(self) -> FinancialConfidence:
        """Get confidence level enum."""
        if self.confidence >= 0.90:
            return FinancialConfidence.HIGH
        elif self.confidence >= 0.70:
            return FinancialConfidence.MEDIUM
        return FinancialConfidence.LOW
    
    @property
    def formatted_amount(self) -> str:
        """Format amount as currency string."""
        if self.is_percentage:
            return f"{self.amount:.2f}%"
        return f"${self.amount:,.2f}"
    
    def is_in_valid_range(self) -> bool:
        """Check if amount is within valid range for field type."""
        if self.is_percentage:
            ranges = PERCENTAGE_VALIDATION_RANGES.get(self.field_type)
        else:
            ranges = FINANCIAL_VALIDATION_RANGES.get(self.field_type)
        
        if not ranges:
            return True  # No range defined, assume valid
        
        return ranges[0] <= self.amount <= ranges[1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "amount": self.amount,
            "formatted": self.formatted_amount,
            "field_type": self.field_type.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "source_text": self.source_text,
            "is_percentage": self.is_percentage,
            "is_valid_range": self.is_in_valid_range(),
            "currency": self.currency,
        }


@dataclass
class FinancialExtractionResult:
    """
    Result of financial terms extraction from a document.
    """
    purchase_price: Optional[ExtractedFinancialValue] = None
    earnest_money: Optional[ExtractedFinancialValue] = None
    down_payment: Optional[ExtractedFinancialValue] = None
    loan_amount: Optional[ExtractedFinancialValue] = None
    closing_costs: Optional[ExtractedFinancialValue] = None
    commission: Optional[ExtractedFinancialValue] = None
    commission_rate: Optional[ExtractedFinancialValue] = None
    additional_terms: List[ExtractedFinancialValue] = field(default_factory=list)
    extraction_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_financial_details(self) -> FinancialDetails:
        """Convert to FinancialDetails TypedDict for state."""
        details: FinancialDetails = {}
        
        if self.purchase_price:
            details["purchase_sale_price"] = self.purchase_price.amount
        if self.earnest_money:
            details["earnest_money_amount"] = self.earnest_money.amount
        if self.commission:
            details["sale_commission_total"] = self.commission.amount
        if self.commission_rate:
            details["sale_commission_rate"] = f"{self.commission_rate.amount}%"
        
        return details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "stats": self.extraction_stats,
        }
        
        if self.purchase_price:
            result["purchase_price"] = self.purchase_price.to_dict()
        if self.earnest_money:
            result["earnest_money"] = self.earnest_money.to_dict()
        if self.down_payment:
            result["down_payment"] = self.down_payment.to_dict()
        if self.loan_amount:
            result["loan_amount"] = self.loan_amount.to_dict()
        if self.closing_costs:
            result["closing_costs"] = self.closing_costs.to_dict()
        if self.commission:
            result["commission"] = self.commission.to_dict()
        if self.commission_rate:
            result["commission_rate"] = self.commission_rate.to_dict()
        if self.additional_terms:
            result["additional_terms"] = [t.to_dict() for t in self.additional_terms]
        
        return result


# ============================================================================
# E3: Financial Terms Extraction Patterns
# ============================================================================

# Patterns for different financial terms with their field types
FINANCIAL_EXTRACTION_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Purchase Price patterns
    r"(?i)purchase\s*(?:price|amount)\s*[:=]?\s*": {
        "field_type": FinancialFieldType.PURCHASE_PRICE,
        "priority": 1,
    },
    r"(?i)sale[s]?\s*price\s*[:=]?\s*": {
        "field_type": FinancialFieldType.PURCHASE_PRICE,
        "priority": 1,
    },
    r"(?i)total\s*(?:purchase\s*)?price\s*[:=]?\s*": {
        "field_type": FinancialFieldType.PURCHASE_PRICE,
        "priority": 1,
    },
    r"(?i)contract\s*price\s*[:=]?\s*": {
        "field_type": FinancialFieldType.PURCHASE_PRICE,
        "priority": 2,
    },
    r"(?i)agreed\s*(?:upon\s*)?price\s*[:=]?\s*": {
        "field_type": FinancialFieldType.PURCHASE_PRICE,
        "priority": 2,
    },
    r"(?i)offering\s*price\s*[:=]?\s*": {
        "field_type": FinancialFieldType.PURCHASE_PRICE,
        "priority": 2,
    },
    
    # Earnest Money patterns
    r"(?i)earnest\s*money\s*(?:deposit|amount)?\s*[:=]?\s*": {
        "field_type": FinancialFieldType.EARNEST_MONEY,
        "priority": 1,
    },
    r"(?i)emd\s*[:=]?\s*": {
        "field_type": FinancialFieldType.EARNEST_MONEY,
        "priority": 2,
    },
    r"(?i)good\s*faith\s*deposit\s*[:=]?\s*": {
        "field_type": FinancialFieldType.EARNEST_MONEY,
        "priority": 1,
    },
    r"(?i)initial\s*deposit\s*[:=]?\s*": {
        "field_type": FinancialFieldType.EARNEST_MONEY,
        "priority": 2,
    },
    
    # Down Payment patterns
    r"(?i)down\s*payment\s*[:=]?\s*": {
        "field_type": FinancialFieldType.DOWN_PAYMENT,
        "priority": 1,
    },
    r"(?i)buyer['']?s?\s*down\s*payment\s*[:=]?\s*": {
        "field_type": FinancialFieldType.DOWN_PAYMENT,
        "priority": 1,
    },
    
    # Loan Amount patterns
    r"(?i)loan\s*amount\s*[:=]?\s*": {
        "field_type": FinancialFieldType.LOAN_AMOUNT,
        "priority": 1,
    },
    r"(?i)mortgage\s*amount\s*[:=]?\s*": {
        "field_type": FinancialFieldType.LOAN_AMOUNT,
        "priority": 1,
    },
    r"(?i)financing\s*amount\s*[:=]?\s*": {
        "field_type": FinancialFieldType.LOAN_AMOUNT,
        "priority": 2,
    },
    r"(?i)first\s*(?:mortgage|loan)\s*[:=]?\s*": {
        "field_type": FinancialFieldType.LOAN_AMOUNT,
        "priority": 2,
    },
    
    # Closing Costs patterns
    r"(?i)closing\s*costs?\s*[:=]?\s*": {
        "field_type": FinancialFieldType.CLOSING_COSTS,
        "priority": 1,
    },
    r"(?i)settlement\s*(?:costs?|charges?)\s*[:=]?\s*": {
        "field_type": FinancialFieldType.CLOSING_COSTS,
        "priority": 1,
    },
    r"(?i)buyer['']?s?\s*closing\s*costs?\s*[:=]?\s*": {
        "field_type": FinancialFieldType.CLOSING_COSTS,
        "priority": 1,
    },
    r"(?i)seller['']?s?\s*(?:closing\s*)?concession\s*[:=]?\s*": {
        "field_type": FinancialFieldType.CLOSING_COSTS,
        "priority": 2,
    },
    
    # Commission patterns
    r"(?i)(?:total\s*)?commission\s*[:=]?\s*": {
        "field_type": FinancialFieldType.COMMISSION,
        "priority": 1,
    },
    r"(?i)brokerage\s*(?:fee|commission)\s*[:=]?\s*": {
        "field_type": FinancialFieldType.COMMISSION,
        "priority": 1,
    },
    r"(?i)agent['']?s?\s*commission\s*[:=]?\s*": {
        "field_type": FinancialFieldType.COMMISSION,
        "priority": 2,
    },
    
    # Commission Rate patterns
    r"(?i)commission\s*(?:rate|percent(?:age)?)\s*[:=]?\s*": {
        "field_type": FinancialFieldType.COMMISSION_RATE,
        "priority": 1,
        "is_percentage": True,
    },
    r"(?i)(?:at|of)\s*(\d+(?:\.\d+)?)\s*%\s*(?:commission)?": {
        "field_type": FinancialFieldType.COMMISSION_RATE,
        "priority": 2,
        "is_percentage": True,
    },
    
    # Deposit patterns
    r"(?i)(?:additional\s*)?deposit\s*[:=]?\s*": {
        "field_type": FinancialFieldType.DEPOSIT,
        "priority": 3,
    },
    
    # Escrow patterns  
    r"(?i)escrow\s*(?:amount|deposit)?\s*[:=]?\s*": {
        "field_type": FinancialFieldType.ESCROW,
        "priority": 2,
    },
    
    # Credit patterns
    r"(?i)(?:seller['']?s?\s*)?credit\s*(?:to\s*buyer)?\s*[:=]?\s*": {
        "field_type": FinancialFieldType.CREDIT,
        "priority": 2,
    },
    r"(?i)repair\s*credit\s*[:=]?\s*": {
        "field_type": FinancialFieldType.CREDIT,
        "priority": 2,
    },
}


# ============================================================================
# E3: Currency Parsing Functions
# ============================================================================

def parse_currency_amount(text: str) -> Optional[Tuple[float, str]]:
    """
    Parse a currency amount from text.
    
    Handles formats:
    - $1,234.56
    - $1.5M or $1.5 million
    - 1,234.56 dollars
    - USD 1,234.56
    
    Returns:
        Tuple of (amount, original_match_text) or None if no match
    """
    if not text:
        return None
    
    # Try to find currency pattern
    text = text.strip()
    
    # Check for million/M suffix
    million_pattern = re.search(
        r'\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(?:million|mil|m)\b',
        text, re.IGNORECASE
    )
    if million_pattern:
        num_str = million_pattern.group(1).replace(',', '')
        try:
            amount = float(num_str) * 1_000_000
            return (amount, million_pattern.group(0))
        except ValueError:
            pass
    
    # Check for standard currency format
    currency_match = re.search(
        r'\$\s*([\d,]+(?:\.\d{1,2})?)',
        text
    )
    if currency_match:
        num_str = currency_match.group(1).replace(',', '')
        try:
            amount = float(num_str)
            return (amount, currency_match.group(0))
        except ValueError:
            pass
    
    # Check for "X dollars" format
    dollars_match = re.search(
        r'([\d,]+(?:\.\d{1,2})?)\s*(?:dollars?|USD)',
        text, re.IGNORECASE
    )
    if dollars_match:
        num_str = dollars_match.group(1).replace(',', '')
        try:
            amount = float(num_str)
            return (amount, dollars_match.group(0))
        except ValueError:
            pass
    
    # Try plain number (last resort, lower confidence)
    plain_match = re.search(r'([\d,]+(?:\.\d{1,2})?)', text)
    if plain_match:
        num_str = plain_match.group(1).replace(',', '')
        try:
            amount = float(num_str)
            # Only accept if it looks like a reasonable currency amount
            if amount >= 100:  # At least $100
                return (amount, plain_match.group(0))
        except ValueError:
            pass
    
    return None


def parse_percentage(text: str) -> Optional[Tuple[float, str]]:
    """
    Parse a percentage from text.
    
    Handles formats:
    - 5.5%
    - 5.5 percent
    - 5.5 pct
    - 5 % (with space)
    
    Returns:
        Tuple of (percentage_value, original_match_text) or None
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Standard percentage pattern - % doesn't have word boundary, use alternation
    pct_match = re.search(r'([\d]+(?:\.[\d]+)?)\s*(%|percent\b|pct\b)', text, re.IGNORECASE)
    if pct_match:
        try:
            value = float(pct_match.group(1))
            if 0 <= value <= 100:  # Valid percentage range
                return (value, pct_match.group(0))
        except ValueError:
            pass
    
    return None


def validate_financial_amount(
    amount: float,
    field_type: FinancialFieldType,
    is_percentage: bool = False
) -> Tuple[bool, float]:
    """
    Validate a financial amount against reasonable ranges.
    
    Returns:
        Tuple of (is_valid, confidence_modifier)
        - confidence_modifier: 1.0 for valid, 0.7 for edge cases, 0.0 for invalid
    """
    if is_percentage:
        ranges = PERCENTAGE_VALIDATION_RANGES.get(field_type)
        if not ranges:
            # Default percentage range
            ranges = (0.0, 100.0)
    else:
        ranges = FINANCIAL_VALIDATION_RANGES.get(field_type)
        if not ranges:
            return (True, 1.0)  # No validation defined
    
    min_val, max_val = ranges
    
    if amount < min_val or amount > max_val:
        return (False, 0.0)
    
    # Check if near edges - use percentage of boundary, not of range
    # Near minimum: within 50% of min_val
    # Near maximum: within 10% of max_val
    near_min = amount < min_val * 1.5 if min_val > 0 else amount < 1000
    near_max = amount > max_val * 0.9
    
    if near_min or near_max:
        return (True, 0.85)  # Valid but edge case
    
    return (True, 1.0)


def validate_earnest_money_ratio(
    earnest_money: float,
    purchase_price: float
) -> Tuple[bool, float]:
    """
    Validate earnest money as a percentage of purchase price.
    
    Typically earnest money is 1-5% of purchase price.
    
    Returns:
        Tuple of (is_reasonable, confidence_modifier)
    """
    if purchase_price <= 0:
        return (True, 1.0)
    
    ratio = (earnest_money / purchase_price) * 100
    
    min_pct, max_pct = EARNEST_MONEY_PERCENT_RANGE
    
    if ratio < min_pct or ratio > max_pct:
        # Outside typical range but might still be valid
        if ratio < 0.1 or ratio > 20:
            return (False, 0.5)  # Very unusual
        return (True, 0.75)  # Unusual but possible
    
    return (True, 1.0)


# ============================================================================
# E3: Financial Terms Extraction Functions
# ============================================================================

def extract_financial_value_after_pattern(
    text: str,
    pattern: str,
    field_type: FinancialFieldType,
    is_percentage: bool = False,
    priority: int = 1,
) -> Optional[ExtractedFinancialValue]:
    """
    Extract a financial value that follows a pattern match.
    
    Args:
        text: Document text to search
        pattern: Regex pattern to find the label
        field_type: Type of financial field
        is_percentage: Whether to look for percentage
        priority: Pattern priority (1 = highest)
    
    Returns:
        ExtractedFinancialValue or None
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    
    # Get text after the match (next ~50 characters)
    end_pos = match.end()
    search_text = text[end_pos:end_pos + 50]
    
    # Base confidence from pattern priority
    base_confidence = 1.0 - (priority - 1) * 0.1
    
    if is_percentage:
        parsed = parse_percentage(search_text)
        if parsed:
            value, source = parsed
            is_valid, confidence_mod = validate_financial_amount(
                value, field_type, is_percentage=True
            )
            return ExtractedFinancialValue(
                amount=value,
                field_type=field_type,
                confidence=base_confidence * confidence_mod,
                source_text=match.group(0) + source,
                is_percentage=True,
            )
    else:
        parsed = parse_currency_amount(search_text)
        if parsed:
            amount, source = parsed
            is_valid, confidence_mod = validate_financial_amount(
                amount, field_type, is_percentage=False
            )
            return ExtractedFinancialValue(
                amount=amount,
                field_type=field_type,
                confidence=base_confidence * confidence_mod,
                source_text=match.group(0) + source,
                is_percentage=False,
            )
    
    return None


def extract_purchase_price(text: str) -> Optional[ExtractedFinancialValue]:
    """Extract purchase/sale price from document text."""
    patterns = [
        (r"(?i)purchase\s*(?:price|amount)\s*[:=]?\s*", 1),
        (r"(?i)sale[s]?\s*price\s*[:=]?\s*", 1),
        (r"(?i)total\s*(?:purchase\s*)?price\s*[:=]?\s*", 1),
        (r"(?i)contract\s*price\s*[:=]?\s*", 2),
        (r"(?i)agreed\s*(?:upon\s*)?price\s*[:=]?\s*", 2),
        (r"(?i)offering\s*price\s*[:=]?\s*", 2),
        (r"(?i)price\s*[:=]\s*\$", 3),  # Generic "price: $X"
    ]
    
    best_match: Optional[ExtractedFinancialValue] = None
    
    for pattern, priority in patterns:
        result = extract_financial_value_after_pattern(
            text, pattern, FinancialFieldType.PURCHASE_PRICE, 
            is_percentage=False, priority=priority
        )
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_earnest_money(text: str) -> Optional[ExtractedFinancialValue]:
    """Extract earnest money deposit amount from document text."""
    patterns = [
        (r"(?i)earnest\s*money\s*(?:deposit|amount)?\s*[:=]?\s*", 1),
        (r"(?i)emd\s*[:=]?\s*", 2),
        (r"(?i)good\s*faith\s*deposit\s*[:=]?\s*", 1),
        (r"(?i)initial\s*deposit\s*[:=]?\s*", 2),
        (r"(?i)deposit\s*[:=]\s*\$", 3),
    ]
    
    best_match: Optional[ExtractedFinancialValue] = None
    
    for pattern, priority in patterns:
        result = extract_financial_value_after_pattern(
            text, pattern, FinancialFieldType.EARNEST_MONEY,
            is_percentage=False, priority=priority
        )
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_down_payment(text: str) -> Optional[ExtractedFinancialValue]:
    """Extract down payment amount from document text."""
    patterns = [
        (r"(?i)down\s*payment\s*[:=]?\s*", 1),
        (r"(?i)buyer['']?s?\s*down\s*payment\s*[:=]?\s*", 1),
    ]
    
    best_match: Optional[ExtractedFinancialValue] = None
    
    for pattern, priority in patterns:
        result = extract_financial_value_after_pattern(
            text, pattern, FinancialFieldType.DOWN_PAYMENT,
            is_percentage=False, priority=priority
        )
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_loan_amount(text: str) -> Optional[ExtractedFinancialValue]:
    """Extract loan/mortgage amount from document text."""
    patterns = [
        (r"(?i)loan\s*amount\s*[:=]?\s*", 1),
        (r"(?i)mortgage\s*amount\s*[:=]?\s*", 1),
        (r"(?i)financing\s*amount\s*[:=]?\s*", 2),
        (r"(?i)first\s*(?:mortgage|loan)\s*[:=]?\s*", 2),
    ]
    
    best_match: Optional[ExtractedFinancialValue] = None
    
    for pattern, priority in patterns:
        result = extract_financial_value_after_pattern(
            text, pattern, FinancialFieldType.LOAN_AMOUNT,
            is_percentage=False, priority=priority
        )
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_closing_costs(text: str) -> Optional[ExtractedFinancialValue]:
    """Extract closing costs from document text."""
    patterns = [
        (r"(?i)closing\s*costs?\s*[:=]?\s*", 1),
        (r"(?i)settlement\s*(?:costs?|charges?)\s*[:=]?\s*", 1),
        (r"(?i)buyer['']?s?\s*closing\s*costs?\s*[:=]?\s*", 1),
        (r"(?i)seller['']?s?\s*(?:closing\s*)?concession\s*[:=]?\s*", 2),
    ]
    
    best_match: Optional[ExtractedFinancialValue] = None
    
    for pattern, priority in patterns:
        result = extract_financial_value_after_pattern(
            text, pattern, FinancialFieldType.CLOSING_COSTS,
            is_percentage=False, priority=priority
        )
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_commission(text: str) -> Optional[ExtractedFinancialValue]:
    """Extract commission amount from document text."""
    patterns = [
        (r"(?i)(?:total\s*)?commission\s*(?:amount)?\s*[:=]?\s*\$", 1),  # Require $ sign
        (r"(?i)brokerage\s*(?:fee|commission)\s*[:=]?\s*\$", 1),
        (r"(?i)agent['']?s?\s*commission\s*[:=]?\s*\$", 2),
    ]
    
    best_match: Optional[ExtractedFinancialValue] = None
    
    for pattern, priority in patterns:
        result = extract_financial_value_after_pattern(
            text, pattern, FinancialFieldType.COMMISSION,
            is_percentage=False, priority=priority
        )
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_commission_rate(text: str) -> Optional[ExtractedFinancialValue]:
    """Extract commission rate (percentage) from document text."""
    patterns = [
        (r"(?i)commission\s*(?:rate|percent(?:age)?)\s*[:=]?\s*", 1),
        (r"(?i)at\s*a?\s*rate\s*of\s*", 1),
        (r"(?i)(\d+(?:\.\d+)?)\s*%\s*(?:commission|of)", 2),
    ]
    
    best_match: Optional[ExtractedFinancialValue] = None
    
    for pattern, priority in patterns:
        result = extract_financial_value_after_pattern(
            text, pattern, FinancialFieldType.COMMISSION_RATE,
            is_percentage=True, priority=priority
        )
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_financial_terms(text: str) -> FinancialExtractionResult:
    """
    Extract all financial terms from document text.
    
    This is the main entry point for E3 financial extraction.
    
    Args:
        text: Document text to analyze
        
    Returns:
        FinancialExtractionResult with all extracted values
    """
    result = FinancialExtractionResult()
    
    # Extract each type of financial term
    result.purchase_price = extract_purchase_price(text)
    result.earnest_money = extract_earnest_money(text)
    result.down_payment = extract_down_payment(text)
    result.loan_amount = extract_loan_amount(text)
    result.closing_costs = extract_closing_costs(text)
    result.commission = extract_commission(text)
    result.commission_rate = extract_commission_rate(text)
    
    # Cross-validate earnest money against purchase price
    if result.purchase_price and result.earnest_money:
        is_reasonable, confidence_mod = validate_earnest_money_ratio(
            result.earnest_money.amount,
            result.purchase_price.amount
        )
        if not is_reasonable:
            result.earnest_money.confidence *= confidence_mod
    
    # Build stats
    extracted_count = sum(1 for v in [
        result.purchase_price, result.earnest_money, result.down_payment,
        result.loan_amount, result.closing_costs, result.commission,
        result.commission_rate
    ] if v is not None)
    
    high_confidence = sum(1 for v in [
        result.purchase_price, result.earnest_money, result.down_payment,
        result.loan_amount, result.closing_costs, result.commission,
        result.commission_rate
    ] if v is not None and v.confidence >= 0.90)
    
    result.extraction_stats = {
        "total_extracted": extracted_count,
        "high_confidence_count": high_confidence,
        "has_purchase_price": result.purchase_price is not None,
        "has_earnest_money": result.earnest_money is not None,
    }
    
    return result


# ============================================================================
# E4: Contract Dates Extraction - Constants & Types
# ============================================================================

from datetime import datetime, date, timedelta

class DateFieldType(Enum):
    """Types of date fields in real estate documents."""
    OFFER_DATE = "offer_date"
    OFFER_EXPIRATION = "offer_expiration"
    ACCEPTANCE_DATE = "acceptance_date"
    CONTRACT_DATE = "contract_date"
    CLOSING_DATE = "closing_date"
    INSPECTION_DATE = "inspection_date"
    INSPECTION_DEADLINE = "inspection_deadline"
    FINANCING_DEADLINE = "financing_deadline"
    APPRAISAL_DEADLINE = "appraisal_deadline"
    OCCUPANCY_DATE = "occupancy_date"
    POSSESSION_DATE = "possession_date"
    EARNEST_MONEY_DUE = "earnest_money_due"
    OTHER = "other"


class DateValidationIssue(Enum):
    """Types of date validation issues."""
    CLOSING_BEFORE_OFFER = "closing_before_offer"
    CLOSING_BEFORE_ACCEPTANCE = "closing_before_acceptance"
    INSPECTION_AFTER_CLOSING = "inspection_after_closing"
    OFFER_EXPIRED = "offer_expired"
    CLOSING_IN_PAST = "closing_in_past"
    DATES_TOO_FAR_APART = "dates_too_far_apart"
    IMPOSSIBLE_DATE = "impossible_date"
    WEEKEND_CLOSING = "weekend_closing"  # Warning, not error


# Common date formats in real estate documents
DATE_FORMATS = [
    "%m/%d/%Y",      # 12/31/2025
    "%m-%d-%Y",      # 12-31-2025
    "%m/%d/%y",      # 12/31/25
    "%m-%d-%y",      # 12-31-25
    "%B %d, %Y",     # December 31, 2025
    "%b %d, %Y",     # Dec 31, 2025
    "%B %d %Y",      # December 31 2025
    "%b %d %Y",      # Dec 31 2025
    "%d %B %Y",      # 31 December 2025
    "%d %b %Y",      # 31 Dec 2025
    "%Y-%m-%d",      # 2025-12-31 (ISO)
    "%d/%m/%Y",      # 31/12/2025 (international, try last)
]

# Maximum reasonable days between dates
MAX_OFFER_TO_CLOSING_DAYS = 365  # 1 year
MAX_OFFER_TO_EXPIRATION_DAYS = 30  # Offers typically expire within 30 days
MIN_OFFER_TO_CLOSING_DAYS = 7  # At least a week typically


@dataclass
class ExtractedDateValue:
    """
    A date value extracted from document text.
    
    Stores the parsed date, original text, field type, and confidence.
    """
    date_value: date
    field_type: DateFieldType
    confidence: float = 0.0
    source_text: str = ""
    original_format: str = ""
    extraction_method: str = "pattern_match"
    
    @property
    def formatted_date(self) -> str:
        """Format date as MM/DD/YYYY for Dotloop compatibility."""
        return self.date_value.strftime("%m/%d/%Y")
    
    @property
    def iso_date(self) -> str:
        """Format date as ISO 8601 (YYYY-MM-DD)."""
        return self.date_value.isoformat()
    
    def is_in_past(self) -> bool:
        """Check if date is in the past."""
        return self.date_value < date.today()
    
    def is_weekend(self) -> bool:
        """Check if date falls on a weekend."""
        return self.date_value.weekday() >= 5  # Saturday=5, Sunday=6
    
    def days_from_today(self) -> int:
        """Get number of days from today (negative if past)."""
        return (self.date_value - date.today()).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.formatted_date,
            "iso_date": self.iso_date,
            "field_type": self.field_type.value,
            "confidence": self.confidence,
            "source_text": self.source_text,
            "is_past": self.is_in_past(),
            "is_weekend": self.is_weekend(),
            "days_from_today": self.days_from_today(),
        }


@dataclass
class DateValidationResult:
    """Result of validating a set of dates."""
    is_valid: bool
    issues: List[Tuple[DateValidationIssue, str]]  # (issue_type, description)
    warnings: List[Tuple[DateValidationIssue, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [(i.value, msg) for i, msg in self.issues],
            "warnings": [(i.value, msg) for i, msg in self.warnings],
        }


@dataclass
class DateExtractionResult:
    """
    Result of contract dates extraction from a document.
    """
    offer_date: Optional[ExtractedDateValue] = None
    offer_expiration: Optional[ExtractedDateValue] = None
    acceptance_date: Optional[ExtractedDateValue] = None
    contract_date: Optional[ExtractedDateValue] = None
    closing_date: Optional[ExtractedDateValue] = None
    inspection_date: Optional[ExtractedDateValue] = None
    inspection_deadline: Optional[ExtractedDateValue] = None
    financing_deadline: Optional[ExtractedDateValue] = None
    occupancy_date: Optional[ExtractedDateValue] = None
    earnest_money_due: Optional[ExtractedDateValue] = None
    additional_dates: List[ExtractedDateValue] = field(default_factory=list)
    validation: Optional[DateValidationResult] = None
    extraction_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_contract_dates(self) -> ContractDates:
        """Convert to ContractDates TypedDict for state."""
        dates: ContractDates = {}
        
        if self.contract_date:
            dates["contract_agreement_date"] = self.contract_date.formatted_date
        elif self.acceptance_date:
            dates["contract_agreement_date"] = self.acceptance_date.formatted_date
        if self.closing_date:
            dates["closing_date"] = self.closing_date.formatted_date
        if self.offer_date:
            dates["offer_date"] = self.offer_date.formatted_date
        if self.offer_expiration:
            dates["offer_expiration_date"] = self.offer_expiration.formatted_date
        if self.inspection_date or self.inspection_deadline:
            insp = self.inspection_date or self.inspection_deadline
            if insp:
                dates["inspection_date"] = insp.formatted_date
        if self.occupancy_date:
            dates["occupancy_date"] = self.occupancy_date.formatted_date
        
        return dates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "stats": self.extraction_stats,
        }
        
        if self.offer_date:
            result["offer_date"] = self.offer_date.to_dict()
        if self.offer_expiration:
            result["offer_expiration"] = self.offer_expiration.to_dict()
        if self.acceptance_date:
            result["acceptance_date"] = self.acceptance_date.to_dict()
        if self.contract_date:
            result["contract_date"] = self.contract_date.to_dict()
        if self.closing_date:
            result["closing_date"] = self.closing_date.to_dict()
        if self.inspection_date:
            result["inspection_date"] = self.inspection_date.to_dict()
        if self.inspection_deadline:
            result["inspection_deadline"] = self.inspection_deadline.to_dict()
        if self.financing_deadline:
            result["financing_deadline"] = self.financing_deadline.to_dict()
        if self.occupancy_date:
            result["occupancy_date"] = self.occupancy_date.to_dict()
        if self.earnest_money_due:
            result["earnest_money_due"] = self.earnest_money_due.to_dict()
        if self.validation:
            result["validation"] = self.validation.to_dict()
        if self.additional_dates:
            result["additional_dates"] = [d.to_dict() for d in self.additional_dates]
        
        return result


# ============================================================================
# E4: Date Extraction Patterns
# ============================================================================

DATE_EXTRACTION_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Offer Date patterns
    r"(?i)offer\s*date\s*[:=]?\s*": {
        "field_type": DateFieldType.OFFER_DATE,
        "priority": 1,
    },
    r"(?i)date\s*of\s*offer\s*[:=]?\s*": {
        "field_type": DateFieldType.OFFER_DATE,
        "priority": 1,
    },
    r"(?i)this\s*offer\s*(?:is\s*)?made\s*(?:on|this)?\s*": {
        "field_type": DateFieldType.OFFER_DATE,
        "priority": 2,
    },
    
    # Offer Expiration patterns
    r"(?i)offer\s*expir(?:es|ation)\s*[:=]?\s*": {
        "field_type": DateFieldType.OFFER_EXPIRATION,
        "priority": 1,
    },
    r"(?i)expir(?:es|ation)\s*(?:date)?\s*[:=]?\s*": {
        "field_type": DateFieldType.OFFER_EXPIRATION,
        "priority": 2,
    },
    r"(?i)valid\s*(?:until|through)\s*[:=]?\s*": {
        "field_type": DateFieldType.OFFER_EXPIRATION,
        "priority": 2,
    },
    
    # Acceptance Date patterns
    r"(?i)acceptance\s*date\s*[:=]?\s*": {
        "field_type": DateFieldType.ACCEPTANCE_DATE,
        "priority": 1,
    },
    r"(?i)date\s*of\s*acceptance\s*[:=]?\s*": {
        "field_type": DateFieldType.ACCEPTANCE_DATE,
        "priority": 1,
    },
    r"(?i)accepted\s*(?:on|this)?\s*": {
        "field_type": DateFieldType.ACCEPTANCE_DATE,
        "priority": 2,
    },
    
    # Contract Date patterns
    r"(?i)contract\s*date\s*[:=]?\s*": {
        "field_type": DateFieldType.CONTRACT_DATE,
        "priority": 1,
    },
    r"(?i)agreement\s*date\s*[:=]?\s*": {
        "field_type": DateFieldType.CONTRACT_DATE,
        "priority": 1,
    },
    r"(?i)effective\s*date\s*[:=]?\s*": {
        "field_type": DateFieldType.CONTRACT_DATE,
        "priority": 1,
    },
    r"(?i)dated?\s*(?:this)?\s*": {
        "field_type": DateFieldType.CONTRACT_DATE,
        "priority": 3,
    },
    
    # Closing Date patterns
    r"(?i)closing\s*date\s*[:=]?\s*": {
        "field_type": DateFieldType.CLOSING_DATE,
        "priority": 1,
    },
    r"(?i)close\s*(?:of\s*)?escrow\s*(?:date)?\s*[:=]?\s*": {
        "field_type": DateFieldType.CLOSING_DATE,
        "priority": 1,
    },
    r"(?i)settlement\s*date\s*[:=]?\s*": {
        "field_type": DateFieldType.CLOSING_DATE,
        "priority": 1,
    },
    r"(?i)closing\s*(?:on\s*or\s*before|by)\s*[:=]?\s*": {
        "field_type": DateFieldType.CLOSING_DATE,
        "priority": 1,
    },
    
    # Inspection Date patterns
    r"(?i)inspection\s*(?:date|period)?\s*[:=]?\s*": {
        "field_type": DateFieldType.INSPECTION_DATE,
        "priority": 1,
    },
    r"(?i)inspection\s*(?:deadline|due)\s*[:=]?\s*": {
        "field_type": DateFieldType.INSPECTION_DEADLINE,
        "priority": 1,
    },
    r"(?i)due\s*diligence\s*(?:period|deadline)?\s*[:=]?\s*": {
        "field_type": DateFieldType.INSPECTION_DEADLINE,
        "priority": 1,
    },
    
    # Financing Deadline patterns
    r"(?i)financing\s*(?:deadline|contingency)\s*[:=]?\s*": {
        "field_type": DateFieldType.FINANCING_DEADLINE,
        "priority": 1,
    },
    r"(?i)loan\s*(?:approval\s*)?deadline\s*[:=]?\s*": {
        "field_type": DateFieldType.FINANCING_DEADLINE,
        "priority": 1,
    },
    r"(?i)mortgage\s*contingency\s*[:=]?\s*": {
        "field_type": DateFieldType.FINANCING_DEADLINE,
        "priority": 2,
    },
    
    # Occupancy/Possession Date patterns
    r"(?i)occupancy\s*(?:date)?\s*[:=]?\s*": {
        "field_type": DateFieldType.OCCUPANCY_DATE,
        "priority": 1,
    },
    r"(?i)possession\s*(?:date)?\s*[:=]?\s*": {
        "field_type": DateFieldType.POSSESSION_DATE,
        "priority": 1,
    },
    
    # Earnest Money Due patterns
    r"(?i)earnest\s*money\s*(?:due|deposit\s*by)\s*[:=]?\s*": {
        "field_type": DateFieldType.EARNEST_MONEY_DUE,
        "priority": 1,
    },
    r"(?i)deposit\s*due\s*(?:by|on)?\s*[:=]?\s*": {
        "field_type": DateFieldType.EARNEST_MONEY_DUE,
        "priority": 2,
    },
}


# ============================================================================
# E4: Date Parsing Functions
# ============================================================================

def parse_date(text: str) -> Optional[Tuple[date, str, str]]:
    """
    Parse a date from text trying multiple formats.
    
    Args:
        text: Text that may contain a date
        
    Returns:
        Tuple of (parsed_date, matched_text, format_used) or None
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Try to find date-like patterns first
    # Match various date formats
    date_patterns = [
        # MM/DD/YYYY or MM-DD-YYYY
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        # Month DD, YYYY or Month DD YYYY
        r'((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{2,4})',
        # DD Month YYYY
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{2,4})',
        # YYYY-MM-DD (ISO)
        r'(\d{4}-\d{2}-\d{2})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            
            # Try each format
            for fmt in DATE_FORMATS:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    # Validate year is reasonable (1990-2100)
                    if 1990 <= parsed.year <= 2100:
                        return (parsed.date(), date_str, fmt)
                except ValueError:
                    continue
            
            # Handle 2-digit year edge case
            if len(date_str) <= 10:  # Short format like 12/31/25
                for fmt in ["%m/%d/%y", "%m-%d-%y"]:
                    try:
                        parsed = datetime.strptime(date_str, fmt)
                        return (parsed.date(), date_str, fmt)
                    except ValueError:
                        continue
    
    return None


def parse_date_with_confidence(
    text: str,
    expected_type: DateFieldType,
) -> Optional[ExtractedDateValue]:
    """
    Parse a date with confidence scoring.
    
    Args:
        text: Text to parse
        expected_type: Expected date field type
        
    Returns:
        ExtractedDateValue or None
    """
    result = parse_date(text)
    if not result:
        return None
    
    parsed_date, matched_text, fmt = result
    
    # Calculate confidence based on format clarity
    if fmt in ["%m/%d/%Y", "%B %d, %Y", "%Y-%m-%d"]:
        confidence = 0.95  # Clear unambiguous format
    elif fmt in ["%m/%d/%y", "%b %d, %Y"]:
        confidence = 0.85  # Slightly ambiguous (2-digit year or abbrev month)
    else:
        confidence = 0.75
    
    return ExtractedDateValue(
        date_value=parsed_date,
        field_type=expected_type,
        confidence=confidence,
        source_text=matched_text,
        original_format=fmt,
    )


# ============================================================================
# E4: Date Validation Functions
# ============================================================================

def validate_date_sequence(
    offer_date: Optional[date] = None,
    acceptance_date: Optional[date] = None,
    closing_date: Optional[date] = None,
    inspection_deadline: Optional[date] = None,
    offer_expiration: Optional[date] = None,
) -> DateValidationResult:
    """
    Validate that dates are in logical sequence.
    
    Logical order:
    1. Offer Date
    2. Acceptance Date (>= Offer Date)
    3. Inspection Deadline (between Acceptance and Closing)
    4. Closing Date (after Acceptance, reasonable timeframe)
    
    Returns:
        DateValidationResult with issues and warnings
    """
    issues: List[Tuple[DateValidationIssue, str]] = []
    warnings: List[Tuple[DateValidationIssue, str]] = []
    today = date.today()
    
    # Check closing date vs offer date
    if closing_date and offer_date:
        if closing_date < offer_date:
            issues.append((
                DateValidationIssue.CLOSING_BEFORE_OFFER,
                f"Closing date ({closing_date}) is before offer date ({offer_date})"
            ))
        
        days_diff = (closing_date - offer_date).days
        if days_diff > MAX_OFFER_TO_CLOSING_DAYS:
            warnings.append((
                DateValidationIssue.DATES_TOO_FAR_APART,
                f"Closing is {days_diff} days after offer (typical max is {MAX_OFFER_TO_CLOSING_DAYS})"
            ))
        elif days_diff < MIN_OFFER_TO_CLOSING_DAYS:
            warnings.append((
                DateValidationIssue.DATES_TOO_FAR_APART,
                f"Only {days_diff} days between offer and closing (typically at least {MIN_OFFER_TO_CLOSING_DAYS})"
            ))
    
    # Check closing date vs acceptance date
    if closing_date and acceptance_date:
        if closing_date < acceptance_date:
            issues.append((
                DateValidationIssue.CLOSING_BEFORE_ACCEPTANCE,
                f"Closing date ({closing_date}) is before acceptance date ({acceptance_date})"
            ))
    
    # Check inspection deadline vs closing
    if inspection_deadline and closing_date:
        if inspection_deadline > closing_date:
            issues.append((
                DateValidationIssue.INSPECTION_AFTER_CLOSING,
                f"Inspection deadline ({inspection_deadline}) is after closing date ({closing_date})"
            ))
    
    # Check if offer has expired
    if offer_expiration and offer_expiration < today:
        warnings.append((
            DateValidationIssue.OFFER_EXPIRED,
            f"Offer expiration ({offer_expiration}) is in the past"
        ))
    
    # Check offer expiration is reasonable
    if offer_expiration and offer_date:
        exp_days = (offer_expiration - offer_date).days
        if exp_days > MAX_OFFER_TO_EXPIRATION_DAYS:
            warnings.append((
                DateValidationIssue.DATES_TOO_FAR_APART,
                f"Offer expires {exp_days} days after offer date (typical max is {MAX_OFFER_TO_EXPIRATION_DAYS})"
            ))
    
    # Check if closing is in the past (warning, not error)
    if closing_date and closing_date < today:
        warnings.append((
            DateValidationIssue.CLOSING_IN_PAST,
            f"Closing date ({closing_date}) is in the past"
        ))
    
    # Check for weekend closing (warning)
    if closing_date and closing_date.weekday() >= 5:
        day_name = "Saturday" if closing_date.weekday() == 5 else "Sunday"
        warnings.append((
            DateValidationIssue.WEEKEND_CLOSING,
            f"Closing date ({closing_date}) falls on a {day_name}"
        ))
    
    is_valid = len(issues) == 0
    return DateValidationResult(is_valid=is_valid, issues=issues, warnings=warnings)


# ============================================================================
# E4: Date Extraction Functions
# ============================================================================

def extract_date_after_pattern(
    text: str,
    pattern: str,
    field_type: DateFieldType,
    priority: int = 1,
) -> Optional[ExtractedDateValue]:
    """
    Extract a date that follows a pattern match.
    
    Args:
        text: Document text to search
        pattern: Regex pattern to find the label
        field_type: Type of date field
        priority: Pattern priority (1 = highest)
    
    Returns:
        ExtractedDateValue or None
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    
    # Get text after the match (next ~50 characters)
    end_pos = match.end()
    search_text = text[end_pos:end_pos + 50]
    
    result = parse_date_with_confidence(search_text, field_type)
    if result:
        # Adjust confidence based on priority
        priority_adjustment = 1.0 - (priority - 1) * 0.05
        result.confidence *= priority_adjustment
        result.source_text = match.group(0) + result.source_text
    
    return result


def extract_offer_date(text: str) -> Optional[ExtractedDateValue]:
    """Extract offer date from document text."""
    patterns = [
        (r"(?i)offer\s*date\s*[:=]?\s*", 1),
        (r"(?i)date\s*of\s*offer\s*[:=]?\s*", 1),
        (r"(?i)this\s*offer\s*(?:is\s*)?made\s*(?:on|this)?\s*", 2),
    ]
    
    best_match: Optional[ExtractedDateValue] = None
    
    for pattern, priority in patterns:
        result = extract_date_after_pattern(text, pattern, DateFieldType.OFFER_DATE, priority)
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_closing_date(text: str) -> Optional[ExtractedDateValue]:
    """Extract closing date from document text."""
    patterns = [
        (r"(?i)closing\s*date\s*[:=]?\s*", 1),
        (r"(?i)close\s*(?:of\s*)?escrow\s*(?:date)?\s*[:=]?\s*", 1),
        (r"(?i)settlement\s*date\s*[:=]?\s*", 1),
        (r"(?i)closing\s*(?:on\s*or\s*before|by)\s*[:=]?\s*", 1),
        (r"(?i)close\s*on\s*", 2),
    ]
    
    best_match: Optional[ExtractedDateValue] = None
    
    for pattern, priority in patterns:
        result = extract_date_after_pattern(text, pattern, DateFieldType.CLOSING_DATE, priority)
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_acceptance_date(text: str) -> Optional[ExtractedDateValue]:
    """Extract acceptance date from document text."""
    patterns = [
        (r"(?i)acceptance\s*date\s*[:=]?\s*", 1),
        (r"(?i)date\s*of\s*acceptance\s*[:=]?\s*", 1),
        (r"(?i)accepted\s*(?:on|this)?\s*", 2),
    ]
    
    best_match: Optional[ExtractedDateValue] = None
    
    for pattern, priority in patterns:
        result = extract_date_after_pattern(text, pattern, DateFieldType.ACCEPTANCE_DATE, priority)
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_offer_expiration(text: str) -> Optional[ExtractedDateValue]:
    """Extract offer expiration date from document text."""
    patterns = [
        (r"(?i)offer\s*expir(?:es|ation)\s*[:=]?\s*", 1),
        (r"(?i)expir(?:es|ation)\s*(?:date)?\s*[:=]?\s*", 2),
        (r"(?i)valid\s*(?:until|through)\s*[:=]?\s*", 2),
    ]
    
    best_match: Optional[ExtractedDateValue] = None
    
    for pattern, priority in patterns:
        result = extract_date_after_pattern(text, pattern, DateFieldType.OFFER_EXPIRATION, priority)
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_contract_date(text: str) -> Optional[ExtractedDateValue]:
    """Extract contract/agreement date from document text."""
    patterns = [
        (r"(?i)contract\s*date\s*[:=]?\s*", 1),
        (r"(?i)agreement\s*date\s*[:=]?\s*", 1),
        (r"(?i)effective\s*date\s*[:=]?\s*", 1),
    ]
    
    best_match: Optional[ExtractedDateValue] = None
    
    for pattern, priority in patterns:
        result = extract_date_after_pattern(text, pattern, DateFieldType.CONTRACT_DATE, priority)
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_inspection_date(text: str) -> Optional[ExtractedDateValue]:
    """Extract inspection date/deadline from document text."""
    patterns = [
        (r"(?i)inspection\s*deadline\s*[:=]?\s*", 1),
        (r"(?i)inspection\s*(?:date|period)?\s*[:=]?\s*", 1),
        (r"(?i)due\s*diligence\s*(?:period|deadline)?\s*[:=]?\s*", 1),
    ]
    
    best_match: Optional[ExtractedDateValue] = None
    
    for pattern, priority in patterns:
        result = extract_date_after_pattern(text, pattern, DateFieldType.INSPECTION_DEADLINE, priority)
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_occupancy_date(text: str) -> Optional[ExtractedDateValue]:
    """Extract occupancy/possession date from document text."""
    patterns = [
        (r"(?i)occupancy\s*(?:date)?\s*[:=]?\s*", 1),
        (r"(?i)possession\s*(?:date)?\s*[:=]?\s*", 1),
    ]
    
    best_match: Optional[ExtractedDateValue] = None
    
    for pattern, priority in patterns:
        result = extract_date_after_pattern(text, pattern, DateFieldType.OCCUPANCY_DATE, priority)
        if result and (not best_match or result.confidence > best_match.confidence):
            best_match = result
    
    return best_match


def extract_all_contract_dates(text: str, validate: bool = True) -> DateExtractionResult:
    """
    Extract all contract dates from document text with validation.
    
    This is the main entry point for E4 date extraction with full
    validation and structured output.
    
    Args:
        text: Document text to analyze
        validate: Whether to validate date sequence
        
    Returns:
        DateExtractionResult with all extracted dates and validation
    """
    result = DateExtractionResult()
    
    # Extract each type of date
    result.offer_date = extract_offer_date(text)
    result.offer_expiration = extract_offer_expiration(text)
    result.acceptance_date = extract_acceptance_date(text)
    result.contract_date = extract_contract_date(text)
    result.closing_date = extract_closing_date(text)
    result.inspection_deadline = extract_inspection_date(text)
    result.occupancy_date = extract_occupancy_date(text)
    
    # Validate dates if requested
    if validate:
        result.validation = validate_date_sequence(
            offer_date=result.offer_date.date_value if result.offer_date else None,
            acceptance_date=result.acceptance_date.date_value if result.acceptance_date else None,
            closing_date=result.closing_date.date_value if result.closing_date else None,
            inspection_deadline=result.inspection_deadline.date_value if result.inspection_deadline else None,
            offer_expiration=result.offer_expiration.date_value if result.offer_expiration else None,
        )
    
    # Build stats
    extracted_count = sum(1 for v in [
        result.offer_date, result.offer_expiration, result.acceptance_date,
        result.contract_date, result.closing_date, result.inspection_deadline,
        result.occupancy_date, result.earnest_money_due,
    ] if v is not None)
    
    result.extraction_stats = {
        "total_extracted": extracted_count,
        "has_offer_date": result.offer_date is not None,
        "has_closing_date": result.closing_date is not None,
        "validation_passed": result.validation.is_valid if result.validation else None,
        "validation_issues": len(result.validation.issues) if result.validation else 0,
        "validation_warnings": len(result.validation.warnings) if result.validation else 0,
    }
    
    return result


# ============================================================================
# E2: Signature Detection - Constants & Types
# ============================================================================

class SignatureFieldType(Enum):
    """Types of signature-related fields in documents."""
    SIGNATURE = "signature"
    INITIAL = "initial"
    DATE = "date"
    TEXT = "text"
    CHECKBOX = "checkbox"


class PageSize(Enum):
    """Standard page sizes with dimensions in points (72 points = 1 inch)."""
    LETTER = (612.0, 792.0)  # 8.5 x 11 inches
    LEGAL = (612.0, 1008.0)  # 8.5 x 14 inches
    A4 = (595.0, 842.0)  # 210 x 297 mm
    
    @property
    def width(self) -> float:
        return self.value[0]
    
    @property
    def height(self) -> float:
        return self.value[1]


# Default field dimensions in points (72 points = 1 inch)
DEFAULT_SIGNATURE_WIDTH = 200.0  # ~2.78 inches
DEFAULT_SIGNATURE_HEIGHT = 30.0  # ~0.42 inches
DEFAULT_INITIAL_WIDTH = 50.0  # ~0.69 inches
DEFAULT_INITIAL_HEIGHT = 30.0
DEFAULT_DATE_WIDTH = 100.0  # ~1.39 inches
DEFAULT_DATE_HEIGHT = 20.0
DEFAULT_TEXT_WIDTH = 150.0
DEFAULT_TEXT_HEIGHT = 20.0

# Estimated character width in points (for position calculation)
ESTIMATED_CHAR_WIDTH = 7.0  # Approximate for 12pt font
ESTIMATED_LINE_HEIGHT = 14.0  # Approximate for 12pt font with spacing


@dataclass
class SignatureCoordinates:
    """
    Precise PDF coordinates for a signature field.
    
    All values are in points (72 points = 1 inch).
    Origin (0, 0) is at bottom-left of page in PDF coordinate system.
    
    For DocuSign/Dotloop, coordinates may need to be converted
    to top-left origin (y_from_top = page_height - y - height).
    """
    x: float  # X position from left edge in points
    y: float  # Y position from bottom edge in points
    width: float  # Width in points
    height: float  # Height in points
    page_number: int  # 1-indexed page number
    page_width: float = PageSize.LETTER.width  # Page width in points
    page_height: float = PageSize.LETTER.height  # Page height in points
    
    @property
    def x_percentage(self) -> float:
        """X position as percentage of page width."""
        return (self.x / self.page_width) * 100 if self.page_width > 0 else 0.0
    
    @property
    def y_percentage(self) -> float:
        """Y position as percentage of page height."""
        return (self.y / self.page_height) * 100 if self.page_height > 0 else 0.0
    
    @property
    def y_from_top(self) -> float:
        """Y position from top of page (for top-left origin systems)."""
        return self.page_height - self.y - self.height
    
    @property
    def y_from_top_percentage(self) -> float:
        """Y position from top as percentage of page height."""
        return (self.y_from_top / self.page_height) * 100 if self.page_height > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all coordinate representations."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "page_number": self.page_number,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "x_percentage": self.x_percentage,
            "y_percentage": self.y_percentage,
            "y_from_top": self.y_from_top,
            "y_from_top_percentage": self.y_from_top_percentage,
        }
    
    def to_docusign_format(self) -> Dict[str, Any]:
        """
        Convert to DocuSign anchor/position format.
        DocuSign uses top-left origin with positions in pixels (at 72 DPI = points).
        """
        return {
            "pageNumber": self.page_number,
            "xPosition": int(self.x),
            "yPosition": int(self.y_from_top),
            "width": int(self.width),
            "height": int(self.height),
        }
    
    def to_dotloop_format(self) -> Dict[str, Any]:
        """
        Convert to Dotloop signature placement format.
        Dotloop uses percentage-based positioning.
        """
        return {
            "page": self.page_number,
            "x": round(self.x_percentage, 2),
            "y": round(self.y_from_top_percentage, 2),
            "width": round((self.width / self.page_width) * 100, 2),
            "height": round((self.height / self.page_height) * 100, 2),
        }


@dataclass
class DetectedSignatureField:
    """
    A signature field detected in a document with full metadata.
    
    This is the internal detection result before converting to SignatureField.
    """
    coordinates: SignatureCoordinates
    field_type: SignatureFieldType
    role: str  # BUYER, SELLER, LISTING_AGENT, etc.
    label: str
    required: bool = True
    confidence: float = 0.0
    detection_method: str = "pattern_match"
    context_text: str = ""
    pattern_matched: Optional[str] = None
    
    # Additional metadata for field identification
    field_id: Optional[str] = None
    group_id: Optional[str] = None  # For grouping related fields (e.g., sig + date)
    
    def to_signature_field(self) -> SignatureField:
        """Convert to SignatureField TypedDict for state."""
        return {
            "page_number": self.coordinates.page_number,
            "x_position": self.coordinates.x,
            "y_position": self.coordinates.y,
            "width": self.coordinates.width,
            "height": self.coordinates.height,
            "field_type": self.field_type.value,
            "label": self.label,
            "required": self.required,
            "assigned_role": self.role,
            "context_text": self.context_text[:100] if self.context_text else None,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full metadata."""
        return {
            "coordinates": self.coordinates.to_dict(),
            "field_type": self.field_type.value,
            "role": self.role,
            "label": self.label,
            "required": self.required,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "context_text": self.context_text,
            "pattern_matched": self.pattern_matched,
            "field_id": self.field_id,
            "group_id": self.group_id,
        }


@dataclass
class SignatureDetectionResult:
    """Result of signature detection with metadata."""
    fields: List[DetectedSignatureField]
    page_count: int
    detection_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_signature_field_list(self) -> List[SignatureField]:
        """Convert all detected fields to SignatureField list."""
        return [f.to_signature_field() for f in self.fields]
    
    def filter_by_role(self, role: str) -> List[DetectedSignatureField]:
        """Filter fields by assigned role."""
        return [f for f in self.fields if f.role == role]
    
    def filter_by_type(self, field_type: SignatureFieldType) -> List[DetectedSignatureField]:
        """Filter fields by field type."""
        return [f for f in self.fields if f.field_type == field_type]
    
    def group_by_role(self) -> Dict[str, List[DetectedSignatureField]]:
        """Group fields by assigned role."""
        grouped: Dict[str, List[DetectedSignatureField]] = {}
        for f in self.fields:
            if f.role not in grouped:
                grouped[f.role] = []
            grouped[f.role].append(f)
        return grouped
    
    def group_by_page(self) -> Dict[int, List[DetectedSignatureField]]:
        """Group fields by page number."""
        grouped: Dict[int, List[DetectedSignatureField]] = {}
        for f in self.fields:
            page = f.coordinates.page_number
            if page not in grouped:
                grouped[page] = []
            grouped[page].append(f)
        return grouped
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fields": [f.to_dict() for f in self.fields],
            "page_count": self.page_count,
            "stats": self.detection_stats,
        }


# ============================================================================
# E2: Signature Detection Patterns
# ============================================================================

# Enhanced signature patterns with field type and typical dimensions
SIGNATURE_DETECTION_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Buyer signatures
    r"(?i)buyer['']?s?\s*signature\s*:?\s*_{0,}": {
        "role": "BUYER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 1,
    },
    r"(?i)purchaser['']?s?\s*signature": {
        "role": "BUYER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 1,
    },
    r"(?i)buyer['']?s?\s*initials?\s*:?\s*_{0,}": {
        "role": "BUYER",
        "field_type": SignatureFieldType.INITIAL,
        "width": DEFAULT_INITIAL_WIDTH,
        "height": DEFAULT_INITIAL_HEIGHT,
        "priority": 2,
    },
    r"(?i)buyer\s*\d?\s*:?\s*_{3,}": {
        "role": "BUYER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 3,
    },
    
    # Seller signatures
    r"(?i)seller['']?s?\s*signature\s*:?\s*_{0,}": {
        "role": "SELLER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 1,
    },
    r"(?i)seller['']?s?\s*initials?\s*:?\s*_{0,}": {
        "role": "SELLER",
        "field_type": SignatureFieldType.INITIAL,
        "width": DEFAULT_INITIAL_WIDTH,
        "height": DEFAULT_INITIAL_HEIGHT,
        "priority": 2,
    },
    r"(?i)seller\s*\d?\s*:?\s*_{3,}": {
        "role": "SELLER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 3,
    },
    
    # Agent signatures
    r"(?i)listing\s*agent['']?s?\s*signature": {
        "role": "LISTING_AGENT",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 1,
    },
    r"(?i)buyer['']?s?\s*agent['']?s?\s*signature": {
        "role": "BUYING_AGENT",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 1,
    },
    r"(?i)selling\s*agent['']?s?\s*signature": {
        "role": "BUYING_AGENT",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 1,
    },
    r"(?i)agent\s*signature": {
        "role": "LISTING_AGENT",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 2,
    },
    
    # Broker signatures
    r"(?i)broker['']?s?\s*signature": {
        "role": "LISTING_BROKER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 1,
    },
    r"(?i)listing\s*broker\s*:?\s*_{0,}": {
        "role": "LISTING_BROKER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 2,
    },
    r"(?i)selling\s*broker\s*:?\s*_{0,}": {
        "role": "BUYING_BROKER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 2,
    },
    
    # Date fields (often paired with signatures)
    r"(?i)date\s*:?\s*_{3,}": {
        "role": "UNKNOWN",
        "field_type": SignatureFieldType.DATE,
        "width": DEFAULT_DATE_WIDTH,
        "height": DEFAULT_DATE_HEIGHT,
        "priority": 3,
    },
    r"(?i)dated?\s*this\s*_{2,}\s*day": {
        "role": "UNKNOWN",
        "field_type": SignatureFieldType.DATE,
        "width": DEFAULT_DATE_WIDTH,
        "height": DEFAULT_DATE_HEIGHT,
        "priority": 3,
    },
    
    # Witness/Notary
    r"(?i)witness['']?s?\s*signature": {
        "role": "OTHER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 2,
    },
    r"(?i)notary\s*(?:public)?\s*signature": {
        "role": "OTHER",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 2,
    },
    
    # Generic signature lines
    r"(?i)sign\s*here\s*:?\s*_{0,}": {
        "role": "UNKNOWN",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 4,
    },
    r"(?i)signature\s*:?\s*_{3,}": {
        "role": "UNKNOWN",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 4,
    },
    r"(?i)x\s*_{5,}": {
        "role": "UNKNOWN",
        "field_type": SignatureFieldType.SIGNATURE,
        "width": DEFAULT_SIGNATURE_WIDTH,
        "height": DEFAULT_SIGNATURE_HEIGHT,
        "priority": 5,
    },
    r"(?i)print\s*name\s*:?\s*_{0,}": {
        "role": "UNKNOWN",
        "field_type": SignatureFieldType.TEXT,
        "width": DEFAULT_TEXT_WIDTH,
        "height": DEFAULT_TEXT_HEIGHT,
        "priority": 3,
    },
}


# ============================================================================
# E2: Signature Detection Functions
# ============================================================================

def calculate_text_coordinates(
    line_index: int,
    char_start: int,
    char_end: int,
    total_lines: int,
    line_length: int,
    page_number: int = 1,
    page_size: PageSize = PageSize.LETTER,
    margin_top: float = 72.0,  # 1 inch top margin
    margin_left: float = 72.0,  # 1 inch left margin
    margin_bottom: float = 72.0,  # 1 inch bottom margin
) -> Tuple[float, float]:
    """
    Calculate approximate PDF coordinates from text position.
    
    Args:
        line_index: 0-indexed line number
        char_start: Starting character position in line
        char_end: Ending character position in line
        total_lines: Total number of lines on page
        line_length: Length of the current line
        page_number: Page number (1-indexed)
        page_size: Page dimensions
        margin_top: Top margin in points
        margin_left: Left margin in points
        margin_bottom: Bottom margin in points
        
    Returns:
        Tuple of (x_position, y_position) in points from bottom-left origin
    """
    # Calculate usable content area
    content_height = page_size.height - margin_top - margin_bottom
    content_width = page_size.width - (margin_left * 2)
    
    # Calculate Y position (from bottom of page)
    # Line 0 is at top, so we calculate from top and convert
    if total_lines > 0:
        line_height = content_height / total_lines
        y_from_top = margin_top + (line_index * line_height)
    else:
        y_from_top = margin_top
    
    y_position = page_size.height - y_from_top - ESTIMATED_LINE_HEIGHT
    
    # Calculate X position
    if line_length > 0:
        char_ratio = char_start / line_length
        x_position = margin_left + (char_ratio * content_width)
    else:
        x_position = margin_left
    
    return x_position, y_position


def detect_signature_fields_enhanced(
    text: str,
    page_number: int = 1,
    page_size: PageSize = PageSize.LETTER,
    doc_type: str = "Unknown",
) -> List[DetectedSignatureField]:
    """
    Detect signature fields in document text with precise coordinates.
    
    Uses pattern matching to identify signature locations and calculates
    approximate PDF coordinates based on text position.
    
    Args:
        text: Document text content
        page_number: Page number (1-indexed)
        page_size: Page dimensions for coordinate calculation
        doc_type: Document type for context-aware detection
        
    Returns:
        List of DetectedSignatureField with coordinates
    """
    detected_fields: List[DetectedSignatureField] = []
    seen_positions: Set[Tuple[int, int]] = set()  # (line_idx, char_pos) to avoid duplicates
    
    lines = text.split('\n')
    total_lines = len(lines)
    
    for line_idx, line in enumerate(lines):
        for pattern, field_info in SIGNATURE_DETECTION_PATTERNS.items():
            match = re.search(pattern, line)
            if match:
                # Check for duplicates at same position
                position_key = (line_idx, match.start())
                if position_key in seen_positions:
                    continue
                seen_positions.add(position_key)
                
                # Calculate coordinates
                x_pos, y_pos = calculate_text_coordinates(
                    line_index=line_idx,
                    char_start=match.start(),
                    char_end=match.end(),
                    total_lines=total_lines,
                    line_length=len(line),
                    page_number=page_number,
                    page_size=page_size,
                )
                
                # Determine role from context if UNKNOWN
                role = field_info["role"]
                if role == "UNKNOWN":
                    role = _infer_role_from_context(lines, line_idx)
                
                # Get field dimensions
                field_type = field_info["field_type"]
                width = field_info.get("width", DEFAULT_SIGNATURE_WIDTH)
                height = field_info.get("height", DEFAULT_SIGNATURE_HEIGHT)
                
                # Create coordinates
                coords = SignatureCoordinates(
                    x=x_pos,
                    y=y_pos,
                    width=width,
                    height=height,
                    page_number=page_number,
                    page_width=page_size.width,
                    page_height=page_size.height,
                )
                
                # Generate label
                label = _generate_signature_field_label(field_type, role)
                
                # Calculate confidence based on pattern priority and match quality
                priority = field_info.get("priority", 5)
                confidence = _calculate_detection_confidence(
                    priority=priority,
                    match_length=match.end() - match.start(),
                    role=role,
                    field_type=field_type,
                )
                
                # Create detected field
                detected_field = DetectedSignatureField(
                    coordinates=coords,
                    field_type=field_type,
                    role=role,
                    label=label,
                    required=True,
                    confidence=confidence,
                    detection_method="pattern_match",
                    context_text=line.strip(),
                    pattern_matched=pattern,
                    field_id=f"sig_{page_number}_{line_idx}_{match.start()}",
                )
                
                detected_fields.append(detected_field)
    
    # Sort by position (top to bottom, left to right)
    detected_fields.sort(
        key=lambda f: (-f.coordinates.y, f.coordinates.x)
    )
    
    return detected_fields


def _generate_signature_field_label(
    field_type: SignatureFieldType,
    role: str,
) -> str:
    """Generate a human-readable label for the signature field."""
    role_name = role.replace("_", " ").title()
    
    if field_type == SignatureFieldType.SIGNATURE:
        return f"{role_name} Signature"
    elif field_type == SignatureFieldType.INITIAL:
        return f"{role_name} Initials"
    elif field_type == SignatureFieldType.DATE:
        return f"{role_name} Date"
    elif field_type == SignatureFieldType.TEXT:
        return f"{role_name} Name"
    elif field_type == SignatureFieldType.CHECKBOX:
        return f"{role_name} Checkbox"
    else:
        return f"{role_name} Field"


def _calculate_detection_confidence(
    priority: int,
    match_length: int,
    role: str,
    field_type: SignatureFieldType,
) -> float:
    """
    Calculate confidence score for detected signature field.
    
    Factors:
    - Pattern priority (1 = highest, 5 = lowest)
    - Match length (longer matches are more specific)
    - Role specificity (known role vs UNKNOWN)
    - Field type specificity
    """
    # Base confidence from priority
    priority_scores = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.75, 5: 0.65}
    confidence = priority_scores.get(priority, 0.70)
    
    # Bonus for longer matches (more specific patterns)
    if match_length > 20:
        confidence += 0.03
    elif match_length > 10:
        confidence += 0.01
    
    # Penalty for unknown role
    if role == "UNKNOWN":
        confidence -= 0.10
    
    # Bonus for signature/initial types (vs generic)
    if field_type in (SignatureFieldType.SIGNATURE, SignatureFieldType.INITIAL):
        confidence += 0.02
    
    return max(0.0, min(1.0, confidence))


def detect_all_signature_fields(
    text: Union[str, List[str]],
    page_count: int = 1,
    page_size: PageSize = PageSize.LETTER,
    doc_type: str = "Unknown",
) -> SignatureDetectionResult:
    """
    Detect all signature fields across multiple pages.
    
    For multi-page documents, text can be:
    - A single string with page breaks (form feed or "--- Page X ---")
    - A list of strings, one per page
    
    Args:
        text: Full document text or list of page texts
        page_count: Total number of pages
        page_size: Page dimensions
        doc_type: Document type for context
        
    Returns:
        SignatureDetectionResult with all detected fields
    """
    all_fields: List[DetectedSignatureField] = []
    
    # Handle list of pages
    if isinstance(text, list):
        for page_idx, page_text in enumerate(text):
            if page_text.strip():
                page_fields = detect_signature_fields_enhanced(
                    text=page_text,
                    page_number=page_idx + 1,
                    page_size=page_size,
                    doc_type=doc_type,
                )
                all_fields.extend(page_fields)
        actual_page_count = len(text)
    else:
        # Split by page if markers are present
        page_markers = re.split(r'\f|--- ?Page \d+ ?---', text, flags=re.IGNORECASE)
        
        if len(page_markers) > 1:
            # Multi-page document with markers
            for page_idx, page_text in enumerate(page_markers):
                if page_text.strip():
                    page_fields = detect_signature_fields_enhanced(
                        text=page_text,
                        page_number=page_idx + 1,
                        page_size=page_size,
                        doc_type=doc_type,
                    )
                    all_fields.extend(page_fields)
            actual_page_count = len(page_markers)
        else:
            # Single page or no markers
            for page_num in range(1, page_count + 1):
                page_fields = detect_signature_fields_enhanced(
                    text=text,
                    page_number=page_num,
                    page_size=page_size,
                    doc_type=doc_type,
                )
                all_fields.extend(page_fields)
                # Only process once if no page markers
                if page_count == 1:
                    break
            actual_page_count = page_count
    
    # Calculate stats
    stats = {
        "total_fields": len(all_fields),
        "by_type": {},
        "by_role": {},
        "by_page": {},
    }
    
    for f in all_fields:
        # Count by type
        type_name = f.field_type.value
        if type_name not in stats["by_type"]:
            stats["by_type"][type_name] = 0
        stats["by_type"][type_name] += 1
        
        # Count by role
        if f.role not in stats["by_role"]:
            stats["by_role"][f.role] = 0
        stats["by_role"][f.role] += 1
        
        # Count by page
        page = f.coordinates.page_number
        if page not in stats["by_page"]:
            stats["by_page"][page] = 0
        stats["by_page"][page] += 1
    
    return SignatureDetectionResult(
        fields=all_fields,
        page_count=actual_page_count,
        detection_stats=stats,
    )


def group_signature_fields_by_role(
    fields: List[DetectedSignatureField]
) -> Dict[str, List[DetectedSignatureField]]:
    """Group signature fields by their assigned role."""
    grouped: Dict[str, List[DetectedSignatureField]] = {}
    for f in fields:
        if f.role not in grouped:
            grouped[f.role] = []
        grouped[f.role].append(f)
    return grouped


def find_paired_fields(
    fields: List[DetectedSignatureField],
    max_y_distance: float = 50.0,  # Max vertical distance in points
) -> List[Tuple[DetectedSignatureField, ...]]:
    """
    Find pairs/groups of related fields (e.g., signature + date on same line).
    
    Returns list of tuples containing related fields.
    """
    paired: List[Tuple[DetectedSignatureField, ...]] = []
    used_indices: Set[int] = set()
    
    for i, field1 in enumerate(fields):
        if i in used_indices:
            continue
            
        group = [field1]
        used_indices.add(i)
        
        for j, field2 in enumerate(fields):
            if j in used_indices:
                continue
            
            # Check if on same page and close vertically
            if (field1.coordinates.page_number == field2.coordinates.page_number and
                abs(field1.coordinates.y - field2.coordinates.y) <= max_y_distance):
                group.append(field2)
                used_indices.add(j)
        
        if len(group) > 1:
            # Sort by x position
            group.sort(key=lambda f: f.coordinates.x)
        
        paired.append(tuple(group))
    
    return paired


# ============================================================================
# Signature Detection Patterns (Legacy - kept for backward compatibility)
# ============================================================================

# Common patterns that indicate signature locations in real estate docs
SIGNATURE_PATTERNS = {
    # Buyer signatures
    r"(?i)buyer['']?s?\s*signature": {"role": "BUYER", "field_type": "signature"},
    r"(?i)buyer['']?s?\s*signature": {"role": "BUYER", "field_type": "signature"},
    r"(?i)purchaser['']?s?\s*signature": {"role": "BUYER", "field_type": "signature"},
    r"(?i)buyer['']?s?\s*initials?": {"role": "BUYER", "field_type": "initial"},
    r"(?i)buyer\s*:?\s*_{3,}": {"role": "BUYER", "field_type": "signature"},
    
    # Seller signatures
    r"(?i)seller['']?s?\s*signature": {"role": "SELLER", "field_type": "signature"},
    r"(?i)seller['']?s?\s*initials?": {"role": "SELLER", "field_type": "initial"},
    r"(?i)seller\s*:?\s*_{3,}": {"role": "SELLER", "field_type": "signature"},
    
    # Agent signatures
    r"(?i)listing\s*agent['']?s?\s*signature": {"role": "LISTING_AGENT", "field_type": "signature"},
    r"(?i)buyer['']?s?\s*agent\s*signature": {"role": "BUYING_AGENT", "field_type": "signature"},
    r"(?i)selling\s*agent['']?s?\s*signature": {"role": "BUYING_AGENT", "field_type": "signature"},
    r"(?i)agent\s*signature": {"role": "LISTING_AGENT", "field_type": "signature"},
    
    # Broker signatures
    r"(?i)broker['']?s?\s*signature": {"role": "LISTING_BROKER", "field_type": "signature"},
    r"(?i)listing\s*broker": {"role": "LISTING_BROKER", "field_type": "signature"},
    r"(?i)selling\s*broker": {"role": "BUYING_BROKER", "field_type": "signature"},
    
    # Date fields (often paired with signatures)
    r"(?i)date\s*:?\s*_{3,}": {"role": "UNKNOWN", "field_type": "date"},
    r"(?i)dated?\s*this": {"role": "UNKNOWN", "field_type": "date"},
    
    # Witness/Notary
    r"(?i)witness\s*signature": {"role": "OTHER", "field_type": "signature"},
    r"(?i)notary": {"role": "OTHER", "field_type": "signature"},
    
    # Generic signature lines
    r"(?i)sign\s*here": {"role": "UNKNOWN", "field_type": "signature"},
    r"(?i)signature\s*:?\s*_{3,}": {"role": "UNKNOWN", "field_type": "signature"},
    r"(?i)x\s*_{5,}": {"role": "UNKNOWN", "field_type": "signature"},
}

# Document type to folder mapping for Dotloop
DOC_TYPE_TO_FOLDER = {
    "Buy-Sell": "Contracts",
    "Purchase Agreement": "Contracts",
    "Counter Offer": "Contracts",
    "Addendum": "Addenda",
    "Amendment": "Addenda",
    "Disclosure": "Disclosures",
    "Mold Disclosure": "Disclosures",
    "Lead Paint Disclosure": "Disclosures",
    "Property Disclosure": "Disclosures",
    "Inspection": "Inspections",
    "Title": "Title",
    "Escrow": "Escrow",
    "Financing": "Financing",
    "Unknown": "Other Documents",
}

# Role context patterns - helps determine role from surrounding text
ROLE_CONTEXT_PATTERNS = {
    "BUYER": [
        r"(?i)buyer\s*\d?\s*[:.]",
        r"(?i)purchaser",
        r"(?i)purchasing\s*party",
    ],
    "SELLER": [
        r"(?i)seller\s*\d?\s*[:.]",
        r"(?i)vendor",
        r"(?i)selling\s*party",
    ],
    "LISTING_AGENT": [
        r"(?i)listing\s*agent",
        r"(?i)seller['']?s?\s*agent",
    ],
    "BUYING_AGENT": [
        r"(?i)buyer['']?s?\s*agent",
        r"(?i)selling\s*agent",
        r"(?i)cooperating\s*agent",
    ],
}


# ============================================================================
# Signature Detection Functions (Legacy wrapper for backward compatibility)
# ============================================================================

def detect_signature_fields(
    text: str, 
    page_number: int = 1,
    doc_type: str = "Unknown"
) -> List[SignatureField]:
    """
    Detect signature fields in document text.
    
    This is a wrapper around detect_signature_fields_enhanced for backward
    compatibility. For full detection with coordinates, use detect_signature_fields_enhanced
    or detect_all_signature_fields.
    
    Returns list of SignatureField TypedDicts.
    """
    detected = detect_signature_fields_enhanced(
        text=text,
        page_number=page_number,
        page_size=PageSize.LETTER,
        doc_type=doc_type,
    )
    return [f.to_signature_field() for f in detected]


def _infer_role_from_context(lines: List[str], current_idx: int) -> str:
    """
    Look at surrounding lines to infer who should sign.
    """
    # Check 5 lines before and after for context
    start = max(0, current_idx - 5)
    end = min(len(lines), current_idx + 5)
    context = ' '.join(lines[start:end])
    
    for role, patterns in ROLE_CONTEXT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, context):
                return role
    
    return "UNKNOWN"


def _generate_field_label(field_info: Dict, role: str) -> str:
    """
    Generate a human-readable label for the signature field.
    """
    field_type = field_info["field_type"]
    role_name = role.replace("_", " ").title()
    
    if field_type == "signature":
        return f"{role_name} Signature"
    elif field_type == "initial":
        return f"{role_name} Initials"
    elif field_type == "date":
        return f"{role_name} Date"
    else:
        return f"{role_name} Field"


def group_signatures_by_role(
    signature_fields: List[SignatureField]
) -> Dict[str, List[SignatureField]]:
    """
    Group detected signature fields by their assigned role.
    This makes it easy to map to Dotloop participants.
    """
    grouped: Dict[str, List[SignatureField]] = {}
    
    for field in signature_fields:
        role = field["assigned_role"]
        if role not in grouped:
            grouped[role] = []
        grouped[role].append(field)
    
    return grouped


# ============================================================================
# Data Extraction Functions
# ============================================================================

def extract_property_address(text: str) -> Optional[PropertyAddress]:
    """
    Extract structured property address from document text.
    
    In production, this would use LLM extraction with structured output.
    """
    # Pattern for common address formats
    address_pattern = r"(\d+)\s+([A-Za-z\s]+(?:St|Street|Ave|Avenue|Ln|Lane|Dr|Drive|Rd|Road|Blvd|Boulevard|Way|Ct|Court|Pl|Place)\.?),?\s*(?:(?:Unit|Apt|Suite|#)\s*(\w+))?,?\s*([A-Za-z\s]+),?\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)"
    
    match = re.search(address_pattern, text, re.IGNORECASE)
    if match:
        street_number, street_name, unit, city, state, zip_code = match.groups()
        return {
            "street_number": street_number.strip(),
            "street_name": street_name.strip(),
            "unit": unit.strip() if unit else None,
            "city": city.strip(),
            "state": state.strip().upper(),
            "zip_code": zip_code.strip(),
            "county": None,  # Would need additional extraction
            "country": "US",
            "mls_number": None,
            "parcel_tax_id": None,
            "full_address": f"{street_number} {street_name}, {city}, {state} {zip_code}",
        }
    
    return None


def extract_financial_details(text: str) -> Optional[FinancialDetails]:
    """
    Extract financial information from document text.
    
    In production, this would use LLM extraction with structured output.
    """
    financials: FinancialDetails = {}
    
    # Purchase price patterns
    price_patterns = [
        r"(?i)purchase\s*price[:\s]*\$?([\d,]+(?:\.\d{2})?)",
        r"(?i)sale\s*price[:\s]*\$?([\d,]+(?:\.\d{2})?)",
        r"(?i)total\s*price[:\s]*\$?([\d,]+(?:\.\d{2})?)",
        r"(?i)price[:\s]*\$?([\d,]+(?:\.\d{2})?)",
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, text)
        if match:
            price_str = match.group(1).replace(",", "")
            financials["purchase_sale_price"] = float(price_str)
            break
    
    # Earnest money patterns
    earnest_patterns = [
        r"(?i)earnest\s*money[:\s]*\$?([\d,]+(?:\.\d{2})?)",
        r"(?i)deposit[:\s]*\$?([\d,]+(?:\.\d{2})?)",
        r"(?i)good\s*faith\s*deposit[:\s]*\$?([\d,]+(?:\.\d{2})?)",
    ]
    
    for pattern in earnest_patterns:
        match = re.search(pattern, text)
        if match:
            earnest_str = match.group(1).replace(",", "")
            financials["earnest_money_amount"] = float(earnest_str)
            break
    
    # Earnest money held by
    held_by_pattern = r"(?i)(?:earnest\s*money|deposit)\s*(?:to\s*be\s*)?held\s*by[:\s]*([A-Za-z\s&]+)"
    match = re.search(held_by_pattern, text)
    if match:
        financials["earnest_money_held_by"] = match.group(1).strip()
    
    return financials if financials else None


def extract_contract_dates(text: str) -> Optional[ContractDates]:
    """
    Extract contract dates from document text.
    
    In production, this would use LLM extraction with structured output.
    """
    dates: ContractDates = {}
    
    # Date pattern (various formats)
    date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})"
    
    # Closing date
    closing_patterns = [
        r"(?i)closing\s*date[:\s]*" + date_pattern,
        r"(?i)close\s*(?:of\s*)?escrow[:\s]*" + date_pattern,
    ]
    for pattern in closing_patterns:
        match = re.search(pattern, text)
        if match:
            dates["closing_date"] = match.group(1)
            break
    
    # Offer date
    offer_pattern = r"(?i)(?:offer|contract)\s*date[:\s]*" + date_pattern
    match = re.search(offer_pattern, text)
    if match:
        dates["offer_date"] = match.group(1)
    
    # Inspection date
    inspection_pattern = r"(?i)inspection\s*(?:due\s*)?date[:\s]*" + date_pattern
    match = re.search(inspection_pattern, text)
    if match:
        dates["inspection_date"] = match.group(1)
    
    return dates if dates else None


def extract_participants(text: str) -> List[ParticipantInfo]:
    """
    Extract participant information from document text.
    
    This is a wrapper around the enhanced extract_participants_with_confidence
    for backward compatibility. Returns only accepted participants.
    
    For full extraction with confidence metadata, use extract_participants_with_confidence.
    """
    result = extract_participants_with_confidence(text)
    return result.to_participant_info_list()


# ============================================================================
# Main Extraction Node
# ============================================================================

def extractor_router(state: DealState) -> str:
    """
    Conditional logic to determine which extractors to run.
    This isn't a node itself, but a function used in add_conditional_edges.
    """
    return "mapper"


def extraction_node(state: DealState) -> dict:
    """
    Node D: Universal Extractor
    
    Iterates through classified docs and:
    1. Extracts structured data (property, financials, dates, participants)
    2. Detects signature field locations
    3. Maps signature fields to participant roles
    4. Prepares data for Dotloop API payload
    """
    print("--- NODE: Extraction ---")
    
    docs = state.get("split_docs", [])
    
    # Initialize extraction results
    all_signature_fields: List[SignatureField] = []
    all_participants: List[ParticipantInfo] = []
    property_details: Optional[PropertyAddress] = None
    financial_details: Optional[FinancialDetails] = None
    contract_dates: Optional[ContractDates] = None
    processed_docs: List[DocumentWithSignatures] = []
    
    # Process each document
    for doc in docs:
        doc_type = doc.get("type", "Unknown")
        raw_text = doc.get("raw_text", "")
        page_range = doc.get("page_range", [1])
        
        print(f"   Processing {doc_type} document...")
        
        # Detect signature fields in this document
        for page_num in range(page_range[0], page_range[-1] + 1):
            page_signatures = detect_signature_fields(
                raw_text, 
                page_number=page_num,
                doc_type=doc_type
            )
            all_signature_fields.extend(page_signatures)
        
        # Extract data based on document type
        if doc_type in ["Buy-Sell", "Purchase Agreement"]:
            print("   Extracting fields from Buy-Sell Agreement...")
            
            # Extract property address
            extracted_address = extract_property_address(raw_text)
            if extracted_address:
                property_details = extracted_address
            
            # Extract financial details
            extracted_financials = extract_financial_details(raw_text)
            if extracted_financials:
                financial_details = extracted_financials
            
            # Extract contract dates
            extracted_dates = extract_contract_dates(raw_text)
            if extracted_dates:
                contract_dates = extracted_dates
            
            # Extract participants
            extracted_participants = extract_participants(raw_text)
            all_participants.extend(extracted_participants)
            
        elif doc_type == "Disclosure":
            print("   Checking compliance on Disclosure...")
            # Disclosures typically just need signature detection
            # which we've already done above
            
        elif doc_type == "Counter Offer":
            print("   Processing Counter Offer...")
            # May have updated price/terms
            counter_financials = extract_financial_details(raw_text)
            if counter_financials and counter_financials.get("purchase_sale_price"):
                financial_details = counter_financials
        
        # Create processed document with signatures
        doc_with_sigs: DocumentWithSignatures = {
            "id": doc.get("id", 0),
            "page_range": page_range,
            "doc_type": doc_type,
            "raw_text": raw_text,
            "signature_fields": [
                sf for sf in all_signature_fields 
                if sf["page_number"] in range(page_range[0], page_range[-1] + 1)
            ],
            "suggested_folder": DOC_TYPE_TO_FOLDER.get(doc_type, "Other Documents"),
        }
        processed_docs.append(doc_with_sigs)
    
    # Group signatures by role for easy mapping
    signature_mapping = group_signatures_by_role(all_signature_fields)
    
    # Log extraction summary
    print(f"   Found {len(all_signature_fields)} signature fields")
    print(f"   Detected roles: {list(signature_mapping.keys())}")
    if property_details:
        print(f"   Property: {property_details.get('full_address', 'N/A')}")
    if financial_details:
        print(f"   Price: ${financial_details.get('purchase_sale_price', 0):,.2f}")
    
    # Build backward-compatible simple values
    simple_financials = {}
    if financial_details:
        simple_financials = {
            "price": financial_details.get("purchase_sale_price", 0.0),
            "earnest_money": financial_details.get("earnest_money_amount", 0.0),
        }
    
    buyers = [p.get("full_name", "") for p in all_participants if p.get("role") == "BUYER"]
    sellers = [p.get("full_name", "") for p in all_participants if p.get("role") == "SELLER"]
    
    # Return updated state
    return {
        # Updated document list with signatures
        "split_docs": processed_docs,
        
        # Structured extracted data
        "property_details": property_details,
        "property_address": property_details.get("full_address") if property_details else None,
        "financial_details": financial_details,
        "financials": simple_financials,
        "contract_dates": contract_dates,
        
        # Participants
        "participants": all_participants,
        "buyers": buyers,
        "sellers": sellers,
        
        # Signature detection results
        "signature_fields": all_signature_fields,
        "signature_mapping": signature_mapping,
    }