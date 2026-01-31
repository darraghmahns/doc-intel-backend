from typing import TypedDict, List, Dict, Optional, Any
import operator
from typing import Annotated

# ============================================================================
# Dotloop-Compatible Data Models
# ============================================================================

class ParticipantInfo(TypedDict, total=False):
    """
    Represents a participant in a real estate transaction.
    Maps directly to Dotloop Loop Participant structure.
    
    Dotloop Roles: BUYER, SELLER, LISTING_AGENT, BUYING_AGENT, 
                   LISTING_BROKER, BUYING_BROKER, ESCROW_TITLE_REP,
                   LOAN_OFFICER, etc.
    """
    full_name: str
    email: Optional[str]
    phone: Optional[str]
    role: str  # Dotloop participant role
    company_name: Optional[str]
    license_number: Optional[str]
    # Address fields (optional)
    street_number: Optional[str]
    street_name: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip_code: Optional[str]


class PropertyAddress(TypedDict, total=False):
    """
    Structured property address matching Dotloop 'Property Address' detail fields.
    """
    street_number: str
    street_name: str
    unit: Optional[str]
    city: str
    state: str
    zip_code: str
    county: Optional[str]
    country: str  # Default: "US"
    mls_number: Optional[str]
    parcel_tax_id: Optional[str]
    # Full formatted string for display
    full_address: str


class FinancialDetails(TypedDict, total=False):
    """
    Financial information matching Dotloop 'Financials' detail fields.
    """
    purchase_sale_price: float
    earnest_money_amount: float
    earnest_money_held_by: Optional[str]
    sale_commission_rate: Optional[str]
    sale_commission_total: Optional[float]
    commission_split_buy_side_percent: Optional[str]
    commission_split_sell_side_percent: Optional[str]


class ContractDates(TypedDict, total=False):
    """
    Contract dates matching Dotloop 'Contract Dates' and 'Offer Dates' fields.
    Format: MM/DD/YYYY
    """
    contract_agreement_date: Optional[str]
    closing_date: Optional[str]
    offer_date: Optional[str]
    offer_expiration_date: Optional[str]
    inspection_date: Optional[str]
    occupancy_date: Optional[str]


class ApprovalChainConfig(TypedDict, total=False):
    """
    Configuration for approval chain rules.
    """
    deal_value_threshold: float  # Deals above this require manager approval
    deal_types_requiring_approval: List[str]  # e.g., ['commercial', 'high_value']
    required_approvers: List[str]  # e.g., ['manager', 'broker']
    enabled: bool


class BrokerageSettings(TypedDict, total=False):
    """
    User/Brokerage preferences for default field population.
    """
    default_commission_rate: Optional[str]
    default_commission_split_buy_side: Optional[str]
    default_commission_split_sell_side: Optional[str]
    default_earnest_money_held_by: Optional[str]
    # Approval chain configuration (H3)
    approval_chain_config: Optional[ApprovalChainConfig]


class ValidationError(TypedDict):
    """
    Structured validation error for UI feedback.
    """
    field: str
    message: str
    expected_format: Optional[str]
    severity: str  # 'Error' | 'Warning'


class ApproverInfo(TypedDict, total=False):
    """
    Represents an approver in the approval chain.
    """
    user_id: str
    name: str
    role: str  # 'manager', 'broker', 'compliance'
    approved: bool
    approved_at: Optional[str]
    notes: Optional[str]


class ApprovalStatus(TypedDict, total=False):
    """
    Current approval chain status for a deal.
    """
    requires_chain_approval: bool
    chain_config: Optional[ApprovalChainConfig]
    approvers: List[ApproverInfo]
    current_level: int  # 0-indexed, which approver we're waiting on
    fully_approved: bool


class RejectionInfo(TypedDict, total=False):
    """
    Details about a rejection for H4 user story.
    """
    reason: str  # 'incorrect_extraction', 'missing_data', 'invalid_document', 'other'
    action: str  # 're_extract', 'manual_edit', 'request_new_document'
    notes: Optional[str]
    rejected_by: str
    rejected_at: str


class SignatureField(TypedDict):
    """
    Represents a detected signature location in a document.
    Used to map to Dotloop/DocuSign signature placement.
    """
    # Location in document
    page_number: int
    x_position: float  # Percentage or absolute position
    y_position: float
    width: float
    height: float
    
    # Signature metadata
    field_type: str  # 'signature', 'initial', 'date', 'text'
    label: str  # e.g., "Buyer Signature", "Seller Initial"
    required: bool
    
    # Role assignment (maps to Dotloop participant roles)
    assigned_role: str  # BUYER, SELLER, LISTING_AGENT, etc.
    
    # Context from document
    context_text: Optional[str]  # Nearby text that identified this field


class DocumentWithSignatures(TypedDict):
    """
    A processed document with extracted signature fields.
    """
    id: int
    page_range: List[int]
    doc_type: str  # Buy-Sell, Disclosure, Counter Offer, etc.
    raw_text: str
    signature_fields: List[SignatureField]
    # Dotloop folder mapping
    suggested_folder: str  # e.g., "Contracts", "Disclosures", "Addenda"


class DotloopPayload(TypedDict, total=False):
    """
    Complete payload structure for Dotloop Loop-It API.
    This is what we build up and send to create a loop.
    """
    # Required
    name: str
    transaction_type: str  # PURCHASE_OFFER, LISTING_FOR_SALE, etc.
    status: str  # PRE_OFFER, UNDER_CONTRACT, SOLD, etc.
    
    # Property
    street_name: Optional[str]
    street_number: Optional[str]
    unit: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip_code: Optional[str]
    county: Optional[str]
    country: Optional[str]
    
    # Participants
    participants: List[ParticipantInfo]
    
    # Optional
    template_id: Optional[int]
    mls_property_id: Optional[str]


# ============================================================================
# Main Deal State
# ============================================================================

class DealState(TypedDict):
    """
    The central state of the Doc Intel Agent.
    This dict is passed and updated by every node in the graph.
    """
    # Meta Information
    deal_id: str
    status: str  # 'Processing', 'Needs_Review', 'Synced', 'Failed'
    
    # Source Context
    email_metadata: Dict[str, str]  # {'sender': '...', 'subject': '...', 'msg_id': '...'}
    
    # File Handling
    raw_pdf_path: str
    # List of split sub-documents found within the raw PDF
    split_docs: List[Dict[str, Any]]  # Will contain DocumentWithSignatures after extraction
    
    # The "Truth" (Extracted & Normalized Data)
    property_address: Optional[str]  # Keep for backward compat
    property_details: Optional[PropertyAddress]  # Structured address
    buyers: List[str]  # Simple names list (backward compat)
    sellers: List[str]  # Simple names list (backward compat)
    participants: List[ParticipantInfo]  # Full participant details
    financials: Dict[str, float]  # Simple dict (backward compat)
    financial_details: Optional[FinancialDetails]  # Structured financials
    contract_dates: Optional[ContractDates]  # Contract timeline
    validation_errors: List[ValidationError]
    
    # Signature Detection Results
    
    # User Preferences
    brokerage_settings: Optional[BrokerageSettings]
    signature_fields: List[SignatureField]  # All detected signature locations
    signature_mapping: Dict[str, List[SignatureField]]  # Grouped by role
    
    # Compliance & Logic
    missing_docs: List[str]
    human_approval_status: str  # 'Pending', 'Approved', 'Rejected'
    
    # Approval Chain (H3)
    approval_status: Optional[ApprovalStatus]
    
    # Rejection History (H4)
    rejection_history: List[RejectionInfo]
    current_rejection: Optional[RejectionInfo]
    
    # Target System Configuration
    target_system: str  # 'dotloop', 'skyslope', 'docusign'
    dotloop_payload: Optional[DotloopPayload]  # Ready-to-send payload
    dotloop_loop_id: Optional[str]
    dotloop_loop_url: Optional[str]
    sync_errors: List[str]
    dotloop_sync_action: Optional[str]