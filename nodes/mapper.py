"""
Field Mapper Node - Normalizes extracted data into target system schemas.

This module handles:
1. Validating extracted data
2. Building Dotloop-compatible payloads (exact API field names)
3. Mapping signature fields to participant assignments
4. Preparing the API request structure

Dotloop API Reference:
- Loop-It API uses camelCase field names
- Loop Details API uses quoted 'Section Name'/'Field Name' structure
- Date format: MM/DD/YYYY
"""

from typing import List, Dict, Optional, Any, Sequence, Union, Set, cast
from enum import Enum
from state import (
    DealState, 
    DotloopPayload, 
    ParticipantInfo,
    SignatureField,
    FinancialDetails,
    BrokerageSettings,
    ValidationError
)


# ============================================================================
# Dotloop API Field Constants (exact API names)
# ============================================================================

class DotloopTransactionType(str, Enum):
    """Valid Dotloop transaction types per Loop-It API."""
    PURCHASE_OFFER = "PURCHASE_OFFER"
    LISTING_FOR_SALE = "LISTING_FOR_SALE"
    LISTING_FOR_LEASE = "LISTING_FOR_LEASE"
    LEASE_OFFER = "LEASE_OFFER"
    REAL_ESTATE_OTHER = "REAL_ESTATE_OTHER"
    OTHER = "OTHER"


class DotloopStatus(str, Enum):
    """Valid Dotloop loop statuses per API."""
    # Purchase offer statuses
    PRE_OFFER = "PRE_OFFER"
    UNDER_CONTRACT = "UNDER_CONTRACT"
    SOLD = "SOLD"
    # Listing statuses
    PRE_LISTING = "PRE_LISTING"
    PRIVATE_LISTING = "PRIVATE_LISTING"
    ACTIVE_LISTING = "ACTIVE_LISTING"
    # Lease statuses
    LEASED = "LEASED"
    # General statuses
    ARCHIVED = "ARCHIVED"
    NEW = "NEW"


class DotloopParticipantRole(str, Enum):
    """Valid Dotloop participant roles per API."""
    # Primary parties
    BUYER = "BUYER"
    SELLER = "SELLER"
    LANDLORD = "LANDLORD"
    TENANT = "TENANT"
    # Agents
    LISTING_AGENT = "LISTING_AGENT"
    BUYING_AGENT = "BUYING_AGENT"
    # Brokers
    LISTING_BROKER = "LISTING_BROKER"
    BUYING_BROKER = "BUYING_BROKER"
    LISTING_TC = "LISTING_TC"
    BUYING_TC = "BUYING_TC"
    # Service providers
    ESCROW_TITLE_REP = "ESCROW_TITLE_REP"
    LOAN_OFFICER = "LOAN_OFFICER"
    APPRAISER = "APPRAISER"
    HOME_INSPECTOR = "HOME_INSPECTOR"
    ATTORNEY = "ATTORNEY"
    REFERRAL_AGENT = "REFERRAL_AGENT"
    # Other
    OTHER = "OTHER"


# ============================================================================
# Dotloop Loop-It API Field Mappings (snake_case -> camelCase)
# ============================================================================

# Top-level Loop-It API fields (exact API names)
LOOP_IT_PROPERTY_FIELDS = {
    "street_name": "streetName",
    "street_number": "streetNumber",
    "unit": "unit",
    "city": "city",
    "state": "state",
    "zip_code": "zipCode",
    "county": "county",
    "country": "country",
}

LOOP_IT_MLS_FIELDS = {
    "mls_number": "mlsPropertyId",
    "mls_property_id": "mlsPropertyId",
    "mls_id": "mlsId",
    "mls_agent_id": "mlsAgentId",
    "nrds_id": "nrdsId",
}

LOOP_IT_PARTICIPANT_FIELDS = {
    "full_name": "fullName",
    "email": "email",
    "role": "role",
    "phone": "Phone",  # Loop Details section uses capitalized
    "cell_phone": "Cell Phone",
    "fax": "Fax",
}

# Required fields for Loop-It API
LOOP_IT_REQUIRED_FIELDS = {"name", "transactionType", "status"}

# Maximum length for loop name
MAX_LOOP_NAME_LENGTH = 200


# ============================================================================
# Dotloop Loop Details API Field Mappings (quoted section/field names)
# ============================================================================

class DotloopDetailSection(str, Enum):
    """Loop Details API section names (exact)."""
    PROPERTY_ADDRESS = "Property Address"
    FINANCIALS = "Financials"
    CONTRACT_DATES = "Contract Dates"
    OFFER_DATES = "Offer Dates"
    CONTRACT_INFO = "Contract Info"
    REFERRAL = "Referral"
    LISTING_INFO = "Listing Information"
    PROPERTY_INFO = "Property"
    GEOGRAPHIC_DESC = "Geographic Description"


# Property Address section fields
DETAIL_PROPERTY_ADDRESS_FIELDS = {
    "street_name": "Street Name",
    "street_number": "Street Number",
    "unit": "Unit Number",
    "city": "City",
    "state": "State/Prov",
    "zip_code": "Zip/Postal Code",
    "county": "County",
    "country": "Country",
    "full_address": "Full Address",
    "mls_number": "MLS Number",
    "parcel_tax_id": "Parcel/Tax ID",
}

# Financials section fields
DETAIL_FINANCIALS_FIELDS = {
    "purchase_sale_price": "Purchase/Sale Price",
    "earnest_money_amount": "Earnest Money Amount",
    "earnest_money_held_by": "Earnest Money Held By",
    "sale_commission_rate": "Sale Commission Rate",
    "sale_commission_total": "Sale Commission Total",
    "commission_split_buy_side_percent": "Commission Split % - Loss Due By Side",
    "commission_split_sell_side_percent": "Commission Split % - Sell Side",
    "list_price": "Listing Price",
    "loan_amount": "Loan Amount",
    "down_payment": "Down Payment",
    "seller_concession": "Seller Concession",
}

# Contract Dates section fields
DETAIL_CONTRACT_DATES_FIELDS = {
    "contract_agreement_date": "Contract Agreement Date",
    "closing_date": "Closing Date",
    "possession_date": "Possession Date",
    "acceptance_date": "Acceptance Date",
}

# Offer Dates section fields
DETAIL_OFFER_DATES_FIELDS = {
    "offer_date": "Offer Date",
    "offer_expiration_date": "Offer Expiration Date",
    "inspection_date": "Inspection Date",
    "financing_deadline": "Financing Deadline",
    "appraisal_date": "Appraisal Date",
    "home_warranty_date": "Home Warranty Date",
    "title_deadline": "Title Deadline",
}


# ============================================================================
# Dotloop Transaction Type Mapping
# ============================================================================

# Map document types to Dotloop transaction types
DOC_TYPE_TO_TRANSACTION: Dict[str, str] = {
    # Purchase-related
    "Buy-Sell": DotloopTransactionType.PURCHASE_OFFER.value,
    "Purchase Agreement": DotloopTransactionType.PURCHASE_OFFER.value,
    "Contract to Buy and Sell": DotloopTransactionType.PURCHASE_OFFER.value,
    "Offer to Purchase": DotloopTransactionType.PURCHASE_OFFER.value,
    "Counter Offer": DotloopTransactionType.PURCHASE_OFFER.value,
    "Counteroffer": DotloopTransactionType.PURCHASE_OFFER.value,
    # Listing-related
    "Listing Agreement": DotloopTransactionType.LISTING_FOR_SALE.value,
    "Exclusive Listing": DotloopTransactionType.LISTING_FOR_SALE.value,
    "Seller Agency": DotloopTransactionType.LISTING_FOR_SALE.value,
    # Lease-related
    "Lease Agreement": DotloopTransactionType.LEASE_OFFER.value,
    "Rental Agreement": DotloopTransactionType.LEASE_OFFER.value,
    "Lease Application": DotloopTransactionType.LEASE_OFFER.value,
    # Other
    "Disclosure": DotloopTransactionType.REAL_ESTATE_OTHER.value,
    "Addendum": DotloopTransactionType.REAL_ESTATE_OTHER.value,
    "Unknown": DotloopTransactionType.REAL_ESTATE_OTHER.value,
}

# Default status based on transaction type
TRANSACTION_DEFAULT_STATUS: Dict[str, str] = {
    DotloopTransactionType.PURCHASE_OFFER.value: DotloopStatus.PRE_OFFER.value,
    DotloopTransactionType.LISTING_FOR_SALE.value: DotloopStatus.PRE_LISTING.value,
    DotloopTransactionType.LISTING_FOR_LEASE.value: DotloopStatus.PRE_LISTING.value,
    DotloopTransactionType.LEASE_OFFER.value: DotloopStatus.PRE_OFFER.value,
    DotloopTransactionType.REAL_ESTATE_OTHER.value: DotloopStatus.NEW.value,
    DotloopTransactionType.OTHER.value: DotloopStatus.NEW.value,
}

# Valid statuses per transaction type
VALID_STATUSES_BY_TRANSACTION: Dict[str, Set[str]] = {
    DotloopTransactionType.PURCHASE_OFFER.value: {
        DotloopStatus.PRE_OFFER.value,
        DotloopStatus.UNDER_CONTRACT.value,
        DotloopStatus.SOLD.value,
        DotloopStatus.ARCHIVED.value,
    },
    DotloopTransactionType.LISTING_FOR_SALE.value: {
        DotloopStatus.PRE_LISTING.value,
        DotloopStatus.PRIVATE_LISTING.value,
        DotloopStatus.ACTIVE_LISTING.value,
        DotloopStatus.UNDER_CONTRACT.value,
        DotloopStatus.SOLD.value,
        DotloopStatus.ARCHIVED.value,
    },
    DotloopTransactionType.LISTING_FOR_LEASE.value: {
        DotloopStatus.PRE_LISTING.value,
        DotloopStatus.ACTIVE_LISTING.value,
        DotloopStatus.LEASED.value,
        DotloopStatus.ARCHIVED.value,
    },
    DotloopTransactionType.LEASE_OFFER.value: {
        DotloopStatus.PRE_OFFER.value,
        DotloopStatus.LEASED.value,
        DotloopStatus.ARCHIVED.value,
    },
}


# ============================================================================
# Signature Role Detection Constants
# ============================================================================

# Keywords in signature labels that indicate participant roles
SIGNATURE_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "BUYER": [
        "buyer", "purchaser", "vendee", "buyer's", "purchaser's",
        "buyer 1", "buyer 2", "buyer #1", "buyer #2",
        "co-buyer", "co buyer", "additional buyer",
    ],
    "SELLER": [
        "seller", "vendor", "seller's", "vendor's",
        "seller 1", "seller 2", "seller #1", "seller #2",
        "co-seller", "co seller", "additional seller",
    ],
    "TENANT": [
        "tenant", "lessee", "renter", "tenant's", "lessee's",
        "tenant 1", "tenant 2", "tenant #1", "tenant #2",
    ],
    "LANDLORD": [
        "landlord", "lessor", "property owner", "landlord's", "lessor's",
    ],
    "LISTING_AGENT": [
        "listing agent", "seller's agent", "seller agent",
        "listing agent's", "la signature",
    ],
    "BUYING_AGENT": [
        "buying agent", "buyer's agent", "buyer agent",
        "selling agent", "buying agent's", "ba signature",
    ],
    "LISTING_BROKER": [
        "listing broker", "seller's broker", "listing broker's",
    ],
    "BUYING_BROKER": [
        "buying broker", "buyer's broker", "selling broker",
    ],
    "ESCROW_TITLE_REP": [
        "escrow", "title", "closing agent", "settlement agent",
        "escrow officer", "title officer", "title company",
    ],
    "LOAN_OFFICER": [
        "loan officer", "lender", "mortgage", "loan officer's",
    ],
    "ATTORNEY": [
        "attorney", "lawyer", "counsel", "legal",
    ],
    "HOME_INSPECTOR": [
        "inspector", "home inspector", "inspection",
    ],
    "APPRAISER": [
        "appraiser", "appraisal",
    ],
    "WITNESS": [
        "witness", "witness signature", "witnessed by",
    ],
    "NOTARY": [
        "notary", "notary public", "notarized",
    ],
}

# Signature field type priorities (for routing order)
SIGNATURE_FIELD_PRIORITY: Dict[str, int] = {
    "signature": 1,  # Primary signature
    "initial": 2,    # Initials
    "date": 3,       # Date fields
    "text": 4,       # Text fields
}


# ============================================================================
# Signature Role Detection Functions
# ============================================================================

def detect_role_from_label(label: Optional[str], context_text: Optional[str] = None) -> Optional[str]:
    """
    Detect participant role from signature field label and context.
    
    Args:
        label: The signature field label (e.g., "Buyer Signature")
        context_text: Optional nearby text for additional context
    
    Returns:
        Detected Dotloop role or None if not detected
    """
    if not label:
        return None
    
    # Combine label and context for searching
    search_text = label.lower()
    if context_text:
        search_text = f"{search_text} {context_text.lower()}"
    
    # Build a list of (keyword, role) pairs sorted by keyword length descending
    # This ensures longer/more specific keywords match first
    # e.g., "buyer's agent" matches before "buyer"
    keyword_role_pairs: List[tuple] = []
    for role, keywords in SIGNATURE_ROLE_KEYWORDS.items():
        for keyword in keywords:
            keyword_role_pairs.append((keyword, role))
    
    # Sort by keyword length descending (longest first)
    keyword_role_pairs.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Check keywords in order of specificity (longest first)
    for keyword, role in keyword_role_pairs:
        if keyword in search_text:
            return role
    
    return None


def detect_role_index_from_label(label: Optional[str]) -> int:
    """
    Detect which participant index a signature refers to (for multiple buyers/sellers).
    
    Args:
        label: The signature field label
    
    Returns:
        0-based index (0 for first buyer, 1 for buyer 2, etc.)
    """
    if not label:
        return 0
    
    label_lower = label.lower()
    
    # Check for explicit numbering
    import re
    
    # Match patterns like "Buyer 2", "Buyer #2", "Seller 1", etc.
    match = re.search(r'(?:buyer|seller|tenant|landlord|agent|broker)\s*#?\s*(\d+)', label_lower)
    if match:
        num = int(match.group(1))
        return max(0, num - 1)  # Convert to 0-based index
    
    # Check for "co-" prefix (usually second participant)
    if any(prefix in label_lower for prefix in ['co-', 'co ', 'additional', 'second', '2nd']):
        return 1
    
    # Check for "first" or "primary"
    if any(prefix in label_lower for prefix in ['first', 'primary', '1st']):
        return 0
    
    return 0  # Default to first participant


def enrich_signature_with_role(
    sig_field: Dict[str, Any],
    default_role: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enrich a signature field with detected role information.
    
    If the signature already has an assigned_role, validates it.
    Otherwise, attempts to detect the role from label/context.
    
    Args:
        sig_field: The signature field dictionary
        default_role: Default role if detection fails
    
    Returns:
        Enriched signature field with role information
    """
    enriched = dict(sig_field)
    
    # Get existing role
    existing_role = enriched.get("assigned_role", "")
    label = enriched.get("label", "")
    context = enriched.get("context_text")
    
    # Try to detect role if not assigned
    if not existing_role or existing_role == "UNKNOWN":
        detected_role = detect_role_from_label(label, context)
        if detected_role:
            enriched["assigned_role"] = detected_role
            enriched["role_detected"] = True
        elif default_role:
            enriched["assigned_role"] = default_role
            enriched["role_detected"] = False
    else:
        # Normalize existing role
        enriched["assigned_role"] = normalize_participant_role(existing_role)
        enriched["role_detected"] = False
    
    # Detect participant index for multi-participant scenarios
    enriched["participant_index"] = detect_role_index_from_label(label)
    
    return enriched


def enrich_all_signatures(
    signature_fields: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich all signature fields with role detection.
    
    Args:
        signature_fields: List of signature field dictionaries
    
    Returns:
        List of enriched signature fields
    """
    return [enrich_signature_with_role(sig) for sig in signature_fields]


# ============================================================================
# Validation Functions
# ============================================================================

def get_valid_participant_roles() -> Set[str]:
    """Return set of all valid Dotloop participant roles."""
    return {role.value for role in DotloopParticipantRole}


def validate_participant_role(role: str) -> bool:
    """Check if a role is valid for Dotloop API."""
    return role in get_valid_participant_roles()


def normalize_participant_role(role: str) -> str:
    """
    Normalize a participant role to valid Dotloop format.
    Handles common variations and returns valid role or OTHER.
    """
    if not role:
        return DotloopParticipantRole.OTHER.value
    
    # Standardize
    normalized = role.upper().strip().replace(" ", "_").replace("-", "_")
    
    # Common aliases
    role_aliases = {
        "PURCHASER": "BUYER",
        "VENDEE": "BUYER",
        "VENDOR": "SELLER",
        "LESSOR": "LANDLORD",
        "LESSEE": "TENANT",
        "RENTER": "TENANT",
        "BUYERS_AGENT": "BUYING_AGENT",
        "SELLERS_AGENT": "LISTING_AGENT",
        "SELLER_AGENT": "LISTING_AGENT",
        "BUYER_AGENT": "BUYING_AGENT",
        "TITLE_COMPANY": "ESCROW_TITLE_REP",
        "TITLE_REP": "ESCROW_TITLE_REP",
        "ESCROW_OFFICER": "ESCROW_TITLE_REP",
        "LENDER": "LOAN_OFFICER",
        "MORTGAGE_BROKER": "LOAN_OFFICER",
        "INSPECTOR": "HOME_INSPECTOR",
        "LAWYER": "ATTORNEY",
        "COUNSEL": "ATTORNEY",
    }
    
    if normalized in role_aliases:
        normalized = role_aliases[normalized]
    
    # Check if valid
    if validate_participant_role(normalized):
        return normalized
    
    return DotloopParticipantRole.OTHER.value


def validate_participants(participants: List[Dict[str, Any]]) -> List[str]:
    """
    Validate participant data and return list of issues.
    """
    issues = []
    
    # Check for required roles
    roles = {p.get("role") for p in participants}
    
    if "BUYER" not in roles and "TENANT" not in roles:
        issues.append("Missing buyer/tenant information")
    if "SELLER" not in roles and "LANDLORD" not in roles:
        issues.append("Missing seller/landlord information")
    
    # Validate each participant
    for i, p in enumerate(participants):
        role = p.get("role", "")
        name = p.get("full_name", "")
        
        if not name:
            issues.append(f"Participant #{i+1} with role {role} missing name")
        
        # Validate role
        if role and not validate_participant_role(role):
            issues.append(f"Invalid role '{role}' for {name or f'participant #{i+1}'}")
    
    return issues


def validate_transaction_type(transaction_type: str) -> bool:
    """Check if transaction type is valid."""
    return transaction_type in {t.value for t in DotloopTransactionType}


def validate_status(status: str, transaction_type: Optional[str] = None) -> bool:
    """Check if status is valid, optionally for specific transaction type."""
    valid_statuses = {s.value for s in DotloopStatus}
    if status not in valid_statuses:
        return False
    
    if transaction_type and transaction_type in VALID_STATUSES_BY_TRANSACTION:
        return status in VALID_STATUSES_BY_TRANSACTION[transaction_type]
    
    return True


def validate_loop_name(name: str) -> List[str]:
    """Validate loop name per API requirements."""
    issues = []
    if not name:
        issues.append("Loop name is required")
    elif len(name) > MAX_LOOP_NAME_LENGTH:
        issues.append(f"Loop name exceeds {MAX_LOOP_NAME_LENGTH} characters")
    return issues


def validate_detailed_for_loop_it(state: DealState) -> List[ValidationError]:
    """
    Validate state against Loop-It requirements and return detailed errors.
    """
    errors: List[ValidationError] = []
    
    # 1. Property Details
    prop = state.get("property_details") or {}
    
    if not prop.get("street_name"):
        errors.append({
            "field": "streetName",
            "message": "Missing property street name",
            "expected_format": "String (e.g. 'Main St')",
            "severity": "Error"
        })
        
    if not prop.get("city"):
        errors.append({
            "field": "city",
            "message": "Missing property city",
            "expected_format": "String",
            "severity": "Error" 
        })
        
    if not prop.get("state"):
        errors.append({
            "field": "state",
            "message": "Missing property state/region",
            "expected_format": "2-letter code (e.g. 'CO')",
            "severity": "Error"
        })
    
    if not prop.get("zip_code"):
         errors.append({
            "field": "zipCode",
            "message": "Missing property zip code",
            "expected_format": "5-digit string (e.g. '80202')",
            "severity": "Error"
        })

    # 2. Participants
    participants = state.get("participants") or []
    has_buyer = False
    has_seller = False
    
    for idx, p in enumerate(participants):
        role = p.get("role", "").upper()
        name = p.get("full_name")
        email = p.get("email")
        
        if "BUYER" in role or "TENANT" in role:
            has_buyer = True
        if "SELLER" in role or "LANDLORD" in role:
            has_seller = True
            
        if not name:
            errors.append({
                "field": f"participants[{idx}].fullName",
                "message": f"Participant {idx+1} is missing a name",
                "expected_format": "String",
                "severity": "Error"
            })
            
        if not email:
            # Email is critical for e-signature routing
            errors.append({
                "field": f"participants[{idx}].email",
                "message": f"Participant {name or idx+1} is missing email (required for signing)",
                "expected_format": "Valid email address",
                "severity": "Warning" # Warning because loop can be created without it
            })

    if not has_buyer:
         errors.append({
            "field": "participants",
            "message": "No Buyer or Tenant identified",
            "expected_format": "At least one participant with role BUYER or TENANT",
            "severity": "Error"
        })

    if not has_seller:
         errors.append({
            "field": "participants",
            "message": "No Seller or Landlord identified",
            "expected_format": "At least one participant with role SELLER or LANDLORD",
            "severity": "Warning" # Sometimes we just represent the buyer side
        })

    # 3. Financials
    fin = state.get("financial_details") or {}
    if not fin.get("purchase_sale_price") and not fin.get("listing_price"):
          errors.append({
            "field": "purchasePrice",
            "message": "No Purchase Price or Listing Price found",
            "expected_format": "Decimal/Number",
            "severity": "Warning"
        })

    return errors


def validate_for_dotloop(state: DealState) -> List[str]:
    """
    Validate that we have all required data for Dotloop API.
    Returns list of missing/invalid fields (backward compatibility).
    """
    detailed_errors = validate_detailed_for_loop_it(state)
    return [e["message"] for e in detailed_errors if e["severity"] == "Error"]


# ============================================================================
# Field Conversion Functions (snake_case to camelCase)
# ============================================================================

def convert_property_to_loop_it(property_details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert property details from snake_case to Dotloop Loop-It camelCase format.
    """
    if not property_details:
        return {}
    
    result = {}
    
    # Map property fields
    for snake_key, camel_key in LOOP_IT_PROPERTY_FIELDS.items():
        value = property_details.get(snake_key)
        if value:
            result[camel_key] = str(value)
    
    # Map MLS fields
    for snake_key, camel_key in LOOP_IT_MLS_FIELDS.items():
        value = property_details.get(snake_key)
        if value:
            result[camel_key] = str(value)
    
    return result


def convert_participant_to_loop_it(participant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert participant from snake_case to Dotloop Loop-It API format.
    """
    result: Dict[str, Any] = {}
    
    # Required fields
    full_name = participant.get("full_name", "")
    if full_name:
        result["fullName"] = full_name
    
    role = participant.get("role", "")
    if role:
        result["role"] = normalize_participant_role(role)
    
    # Optional fields
    email = participant.get("email")
    if email:
        result["email"] = email
    
    return result


def convert_participants_to_loop_it(
    participants: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert all participants to Dotloop Loop-It API format.
    """
    return [convert_participant_to_loop_it(p) for p in participants if p.get("full_name")]


# ============================================================================
# Payload Building Functions
# ============================================================================

def determine_transaction_type(state: Dict[str, Any]) -> str:
    """
    Determine the Dotloop transaction type from deal state.
    """
    # Check split docs for document types
    docs = state.get("split_docs", [])
    for doc in docs:
        doc_type = doc.get("doc_type", doc.get("type", ""))
        if doc_type in DOC_TYPE_TO_TRANSACTION:
            return DOC_TYPE_TO_TRANSACTION[doc_type]
    
    # Default to purchase offer
    return DotloopTransactionType.PURCHASE_OFFER.value


def build_loop_name(
    property_details: Optional[Dict[str, Any]],
    participants: List[ParticipantInfo],
    max_length: int = MAX_LOOP_NAME_LENGTH
) -> str:
    """
    Build a descriptive loop name (max 200 chars per API).
    Priority: Full address > Street address > First buyer name > Default
    """
    # Try full address first
    if property_details:
        full_addr = property_details.get("full_address", "")
        if full_addr:
            return full_addr[:max_length]
        
        # Build from components
        parts = []
        street_num = property_details.get("street_number", "")
        street_name = property_details.get("street_name", "")
        city = property_details.get("city", "")
        state_val = property_details.get("state", "")
        
        if street_num or street_name:
            parts.append(f"{street_num} {street_name}".strip())
        if city:
            parts.append(city)
        if state_val:
            parts.append(state_val)
        
        if parts:
            address = ", ".join(parts)
            return address[:max_length]
    
    # Fall back to first buyer name
    if participants:
        buyers = [p for p in participants if p.get("role") in ("BUYER", "TENANT")]
        if buyers:
            name = buyers[0].get("full_name", "")
            if name:
                return name[:max_length]
    
    return "New Transaction"


def build_loop_it_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete Dotloop Loop-It API payload with exact camelCase field names.
    
    This is the payload sent to POST /loop-it?profile_id=<profile_id>
    
    Required fields: name, transactionType, status
    """
    property_details = state.get("property_details", {}) or {}
    participants = state.get("participants", [])
    
    # Determine transaction type and status
    transaction_type = determine_transaction_type(state)
    status = TRANSACTION_DEFAULT_STATUS.get(transaction_type, DotloopStatus.NEW.value)
    
    # Build loop name
    loop_name = build_loop_name(property_details, participants or [])
    
    # Build payload with required fields (exact API field names)
    payload: Dict[str, Any] = {
        "name": loop_name,
        "transactionType": transaction_type,
        "status": status,
    }
    
    # Add property fields (camelCase)
    property_fields = convert_property_to_loop_it(property_details)
    payload.update(property_fields)
    
    # Add participants
    if participants:
        api_participants = convert_participants_to_loop_it(participants)
        if api_participants:
            payload["participants"] = api_participants
    
    return payload


def build_dotloop_payload(state: DealState) -> DotloopPayload:
    """
    Build a complete Dotloop Loop-It API payload from extracted state.
    
    DEPRECATED: Use build_loop_it_payload() for exact API format.
    This function maintains backward compatibility with snake_case TypedDict.
    """
    # Use new function and convert back to snake_case for TypedDict
    loop_it = build_loop_it_payload(cast(Dict[str, Any], state))
    
    # Build backward-compatible payload
    property_details = state.get("property_details", {}) or {}
    participants = state.get("participants", [])
    
    # Determine transaction type from documents
    docs = state.get("split_docs", [])
    transaction_type = "PURCHASE_OFFER"  # Default
    for doc in docs:
        doc_type = doc.get("doc_type", doc.get("type", "Unknown"))
        if doc_type in DOC_TYPE_TO_TRANSACTION:
            transaction_type = DOC_TYPE_TO_TRANSACTION[doc_type]
            break
    
    # Build loop name (usually property address or lead name)
    loop_name = property_details.get("full_address", "")
    if not loop_name and participants:
        # Use first buyer name if no address
        buyers = [p for p in participants if p.get("role") == "BUYER"]
        if buyers:
            loop_name = buyers[0].get("full_name", "New Transaction")
    
    # Build participant list for API
    api_participants = []
    for p in participants:
        api_participant: ParticipantInfo = {
            "full_name": p.get("full_name", ""),
            "email": p.get("email"),
            "role": p.get("role", "OTHER"),
        }
        phone = p.get("phone")
        if phone:
            api_participant["phone"] = phone
        company = p.get("company_name")
        if company:
            api_participant["company_name"] = company
        api_participants.append(api_participant)
    
    # Build the payload
    payload: DotloopPayload = {
        "name": loop_name or "New Transaction",
        "transaction_type": transaction_type,
        "status": TRANSACTION_DEFAULT_STATUS.get(transaction_type, "NEW"),
        "participants": api_participants,
    }
    
    # Add property details if available
    if property_details:
        street_name = property_details.get("street_name")
        if street_name:
            payload["street_name"] = street_name
        street_number = property_details.get("street_number")
        if street_number:
            payload["street_number"] = street_number
        unit = property_details.get("unit")
        if unit:
            payload["unit"] = unit
        city = property_details.get("city")
        if city:
            payload["city"] = city
        state_val = property_details.get("state")
        if state_val:
            payload["state"] = state_val
        zip_code = property_details.get("zip_code")
        if zip_code:
            payload["zip_code"] = zip_code
        county = property_details.get("county")
        if county:
            payload["county"] = county
        payload["country"] = property_details.get("country", "US")
        
        mls_number = property_details.get("mls_number")
        if mls_number:
            payload["mls_property_id"] = mls_number
    
    return payload


# ============================================================================
# Loop Details Payload Builder (for PATCH /loop/:loop_id/detail)
# ============================================================================

def build_loop_details_payload(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build payload for Dotloop Loop Details API (PATCH /loop/:loop_id/detail).
    
    This API uses a nested structure with quoted section and field names:
    {
        "Property Address": {"Street Name": "...", "City": "..."},
        "Financials": {"Purchase/Sale Price": "..."},
        "Contract Dates": {"Closing Date": "..."}
    }
    
    Date format: MM/DD/YYYY
    """
    payload: Dict[str, Dict[str, Any]] = {}
    
    # Property Address section
    property_details = state.get("property_details", {}) or {}
    if property_details:
        property_section: Dict[str, Any] = {}
        for snake_key, detail_key in DETAIL_PROPERTY_ADDRESS_FIELDS.items():
            value = property_details.get(snake_key)
            if value:
                property_section[detail_key] = str(value)
        if property_section:
            payload[DotloopDetailSection.PROPERTY_ADDRESS.value] = property_section
    
    # Financials section
    financial_details = state.get("financial_details", {}) or {}
    if financial_details:
        financials_section: Dict[str, Any] = {}
        for snake_key, detail_key in DETAIL_FINANCIALS_FIELDS.items():
            value = financial_details.get(snake_key)
            if value is not None:
                # Format currency values
                if isinstance(value, (int, float)) and "price" in snake_key.lower() or "amount" in snake_key.lower():
                    financials_section[detail_key] = f"{value:.2f}"
                else:
                    financials_section[detail_key] = str(value)
        if financials_section:
            payload[DotloopDetailSection.FINANCIALS.value] = financials_section
    
    # Contract Dates section
    contract_dates = state.get("contract_dates", {}) or {}
    if contract_dates:
        # Contract Dates section (subset)
        contract_section: Dict[str, Any] = {}
        for snake_key, detail_key in DETAIL_CONTRACT_DATES_FIELDS.items():
            value = contract_dates.get(snake_key)
            if value:
                contract_section[detail_key] = str(value)
        if contract_section:
            payload[DotloopDetailSection.CONTRACT_DATES.value] = contract_section
        
        # Offer Dates section (subset)
        offer_section: Dict[str, Any] = {}
        for snake_key, detail_key in DETAIL_OFFER_DATES_FIELDS.items():
            value = contract_dates.get(snake_key)
            if value:
                offer_section[detail_key] = str(value)
        if offer_section:
            payload[DotloopDetailSection.OFFER_DATES.value] = offer_section
    
    return payload


def build_participant_detail(participant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Loop Details format for a single participant.
    Used when updating participant info after loop creation.
    """
    detail: Dict[str, Any] = {}
    
    full_name = participant.get("full_name")
    if full_name:
        detail["Name"] = full_name
    email = participant.get("email")
    if email:
        detail["Email"] = email
    phone = participant.get("phone")
    if phone:
        detail["Phone"] = phone
    company_name = participant.get("company_name")
    if company_name:
        detail["Company"] = company_name
    license_number = participant.get("license_number")
    if license_number:
        detail["License #"] = license_number
    
    # Address fields
    address_parts: List[str] = []
    street_number = participant.get("street_number")
    if street_number:
        address_parts.append(str(street_number))
    street_name = participant.get("street_name")
    if street_name:
        address_parts.append(str(street_name))
    if address_parts:
        detail["Street Address"] = " ".join(address_parts)
    city = participant.get("city")
    if city:
        detail["City"] = city
    state_val = participant.get("state")
    if state_val:
        detail["State"] = state_val
    zip_code = participant.get("zip_code")
    if zip_code:
        detail["Zip"] = zip_code
    
    return detail


# ============================================================================
# Signature to Participant Routing
# ============================================================================

def get_participant_for_signature(
    sig_field: Dict[str, Any],
    role_to_participants: Dict[str, List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """
    Get the appropriate participant for a signature field.
    
    Handles:
    - Role matching (BUYER -> buyer participant)
    - Multiple participants with same role (Buyer 1, Buyer 2)
    - Fallback when role not found
    
    Args:
        sig_field: Enriched signature field with role info
        role_to_participants: Dict mapping roles to participant lists
    
    Returns:
        Matching participant or None
    """
    role = sig_field.get("assigned_role", "")
    normalized_role = normalize_participant_role(role)
    participant_index = sig_field.get("participant_index", 0)
    
    # Find participants for this role
    matching_participants = role_to_participants.get(normalized_role, [])
    
    if not matching_participants:
        return None
    
    # Get the specific participant by index (for "Buyer 2", etc.)
    if participant_index < len(matching_participants):
        return matching_participants[participant_index]
    
    # Fall back to first participant if index out of range
    return matching_participants[0]


def build_signature_routing(
    signature_fields: List[Dict[str, Any]],
    participants: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build complete signature routing information for document signing.
    
    This creates a comprehensive routing structure that can be used
    for Dotloop, DocuSign, or other e-signature platforms.
    
    Returns:
        {
            "by_participant": {participant_name: [signature_fields]},
            "by_role": {role: [signature_fields]},
            "unassigned": [signature_fields without matching participants],
            "routing_order": [participants in signing order],
            "summary": {role: count}
        }
    """
    # Enrich signatures with role detection
    enriched_signatures = enrich_all_signatures(signature_fields)
    
    # Group participants by role
    role_to_participants: Dict[str, List[Dict[str, Any]]] = {}
    for p in participants:
        role = p.get("role", "UNKNOWN")
        normalized_role = normalize_participant_role(role)
        if normalized_role not in role_to_participants:
            role_to_participants[normalized_role] = []
        role_to_participants[normalized_role].append(p)
    
    # Build routing structures
    by_participant: Dict[str, List[Dict[str, Any]]] = {}
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    unassigned: List[Dict[str, Any]] = []
    summary: Dict[str, int] = {}
    
    for sig_field in enriched_signatures:
        role = sig_field.get("assigned_role", "UNKNOWN")
        normalized_role = normalize_participant_role(role)
        
        # Add to by_role
        if normalized_role not in by_role:
            by_role[normalized_role] = []
        by_role[normalized_role].append(sig_field)
        
        # Update summary
        summary[normalized_role] = summary.get(normalized_role, 0) + 1
        
        # Find matching participant
        participant = get_participant_for_signature(sig_field, role_to_participants)
        
        if participant:
            name = participant.get("full_name", "Unknown")
            if name not in by_participant:
                by_participant[name] = []
            by_participant[name].append({
                "page": sig_field.get("page_number"),
                "x": sig_field.get("x_position"),
                "y": sig_field.get("y_position"),
                "width": sig_field.get("width"),
                "height": sig_field.get("height"),
                "type": sig_field.get("field_type"),
                "label": sig_field.get("label"),
                "required": sig_field.get("required", True),
                "role": normalized_role,
            })
        else:
            unassigned.append(sig_field)
    
    # Determine routing order (primary parties first, then agents, then others)
    routing_priority = {
        "BUYER": 1, "TENANT": 1,
        "SELLER": 2, "LANDLORD": 2,
        "BUYING_AGENT": 3, "LISTING_AGENT": 4,
        "BUYING_BROKER": 5, "LISTING_BROKER": 6,
        "ESCROW_TITLE_REP": 7, "LOAN_OFFICER": 8,
        "ATTORNEY": 9, "OTHER": 10,
    }
    
    routing_order = sorted(
        [p for p in participants if p.get("full_name") in by_participant],
        key=lambda p: routing_priority.get(
            normalize_participant_role(p.get("role", "")), 
            99
        )
    )
    
    return {
        "by_participant": by_participant,
        "by_role": by_role,
        "unassigned": unassigned,
        "routing_order": [p.get("full_name") for p in routing_order],
        "summary": summary,
    }


def map_signatures_to_participants(
    signature_fields: List[Dict[str, Any]],
    participants: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Map detected signature fields to specific participants.
    
    This creates a mapping that can be used when setting up
    signature routing in Dotloop or DocuSign.
    
    Signature Assignment Rules:
    - BUYER signatures → buyer participant(s)
    - SELLER signatures → seller participant(s)
    - LISTING_AGENT → listing agent
    - BUYING_AGENT → buying agent
    - Numbered signatures (Buyer 2) → corresponding participant index
    
    Returns a dict with participant names as keys and their
    assigned signature fields as values.
    """
    # Enrich signatures with role detection
    enriched_signatures = enrich_all_signatures(signature_fields)
    
    # Group participants by role
    role_to_participants: Dict[str, List[Dict[str, Any]]] = {}
    for p in participants:
        role = p.get("role", "UNKNOWN")
        normalized_role = normalize_participant_role(role)
        if normalized_role not in role_to_participants:
            role_to_participants[normalized_role] = []
        role_to_participants[normalized_role].append(p)
    
    # Map signatures to specific participants
    participant_signatures: Dict[str, List[Dict[str, Any]]] = {}
    
    for sig_field in enriched_signatures:
        # Get the appropriate participant for this signature
        participant = get_participant_for_signature(sig_field, role_to_participants)
        
        if participant:
            name = participant.get("full_name", "Unknown")
            
            if name not in participant_signatures:
                participant_signatures[name] = []
            
            participant_signatures[name].append({
                "page": sig_field.get("page_number"),
                "x": sig_field.get("x_position"),
                "y": sig_field.get("y_position"),
                "type": sig_field.get("field_type"),
                "label": sig_field.get("label"),
                "required": sig_field.get("required", True),
            })
    
    return participant_signatures


def validate_signature_routing(
    routing: Dict[str, Any]
) -> List[str]:
    """
    Validate signature routing and return any issues.
    
    Args:
        routing: Output from build_signature_routing()
    
    Returns:
        List of validation issues
    """
    issues = []
    
    unassigned = routing.get("unassigned", [])
    if unassigned:
        issues.append(f"{len(unassigned)} signature field(s) could not be assigned to participants")
    
    by_participant = routing.get("by_participant", {})
    summary = routing.get("summary", {})
    
    # Check for expected roles that have signatures but no participants
    for role, count in summary.items():
        if role not in ["OTHER", "UNKNOWN"] and count > 0:
            # Check if any participant has this role
            has_participant = any(
                p for p in by_participant.keys() 
            )
            if not has_participant:
                issues.append(f"Found {count} {role} signature(s) but no {role} participant")
    
    return issues


# ============================================================================
# Main Mapper Node
# ============================================================================

def apply_brokerage_defaults(
    financials: Optional[FinancialDetails], 
    settings: Optional[BrokerageSettings]
) -> Optional[FinancialDetails]:
    """
    Applies brokerage default settings to financial details if values are missing.
    """
    if not settings:
        return financials
        
    # If no financials but we have settings, initialize empty
    if financials is None:
        financials = cast(FinancialDetails, {})
        
    # Helper to check if value exists (not None and not empty string)
    def has_value(d: Any, k: str) -> bool:
        val = d.get(k)
        return val is not None and val != ""

    # Apply defaults
    def apply_setting(fin_key: str, setting_key: str) -> None:
        # Pylance safe access: check setting existence first
        setting_val = settings.get(setting_key) if settings else None
        
        # If we have a setting value and the financial field is empty
        if setting_val and not has_value(financials, fin_key):
            financials[fin_key] = setting_val # type: ignore

    apply_setting("sale_commission_rate", "default_commission_rate")
    apply_setting("commission_split_buy_side_percent", "default_commission_split_buy_side")
    apply_setting("commission_split_sell_side_percent", "default_commission_split_sell_side")
    apply_setting("earnest_money_held_by", "default_earnest_money_held_by")
        
    return financials


def field_mapper_node(state: DealState) -> dict:
    """
    Node E: Field & Data Mapper
    
    Validates extracted data and builds target system payloads.
    For Dotloop:
    - Validates required fields
    - Builds Loop-It API payload (exact camelCase field names)
    - Builds Loop Details payload (for detailed field updates)
    - Maps signature fields to participants
    """
    print("--- NODE: Field Mapper ---")
    
    target_system = state.get("target_system", "dotloop")
    
    # Apply brokerage defaults to financials
    if state.get("brokerage_settings"):
        financials = state.get("financial_details")
        updated_financials = apply_brokerage_defaults(
            financials, 
            state.get("brokerage_settings")
        )
        if updated_financials:
            state["financial_details"] = updated_financials
            print("   Applied brokerage defaults to financial details")
    
    # Validate extracted data
    detailed_errors = validate_detailed_for_loop_it(state)
    state["validation_errors"] = detailed_errors
    
    # Backward compatibility for 'missing_docs' (list of strings)
    validation_issues = [f"{e['message']} ({e['field']})" for e in detailed_errors if e['severity'] == "Error"]
    
    if detailed_errors:
        print(f"   Validation issues found: {len(detailed_errors)}")
        for err in detailed_errors:
            print(f"      - [{err['severity']}] {err['message']} (Field: {err['field']})")
        
        # Set status to needs review if critical issues
        critical_issues = [e for e in detailed_errors if e['severity'] == 'Error']

        if critical_issues:
            return {
                "human_approval_status": "Pending",
                "missing_docs": validation_issues,
                "validation_errors": detailed_errors,
                "status": "Needs_Review",
                # Return partial payloads so user can see what we have
                "dotloop_payload": build_dotloop_payload(state) if target_system == "dotloop" else None, 
                "loop_it_payload": build_loop_it_payload(cast(Dict[str, Any], state)) if target_system == "dotloop" else None
            }
    
    # Build Dotloop payloads
    dotloop_payload = None
    loop_it_payload = None
    loop_details_payload = None
    
    if target_system == "dotloop":
        # Build Loop-It payload (exact API format with camelCase)
        loop_it_payload = build_loop_it_payload(cast(Dict[str, Any], state))
        print(f"   Built Loop-It payload for: {loop_it_payload.get('name', 'Unknown')}")
        print(f"   Transaction type: {loop_it_payload.get('transactionType')}")
        print(f"   Participants: {len(loop_it_payload.get('participants', []))}")
        
        # Build Loop Details payload (for updating after creation)
        loop_details_payload = build_loop_details_payload(cast(Dict[str, Any], state))
        if loop_details_payload:
            print(f"   Loop Details sections: {list(loop_details_payload.keys())}")
        
        # Also build backward-compatible payload
        dotloop_payload = build_dotloop_payload(state)
    
    # Map signatures to participants
    signature_fields = state.get("signature_fields", [])
    participants = state.get("participants", [])
    participant_signature_map = map_signatures_to_participants(
        cast(List[Dict[str, Any]], signature_fields), 
        cast(List[Dict[str, Any]], participants)
    )
    
    if participant_signature_map:
        print(f"   Signature assignments:")
        for name, sigs in participant_signature_map.items():
            print(f"      - {name}: {len(sigs)} signature fields")
    
    return {
        "human_approval_status": "Pending",
        "financial_details": state.get("financial_details"),
        "dotloop_payload": dotloop_payload,
        "loop_it_payload": loop_it_payload,
        "loop_details_payload": loop_details_payload,
        "participant_signature_map": participant_signature_map,
        "status": "Ready_For_Review" if not validation_issues else "Needs_Review",
        "missing_docs": validation_issues,
    }