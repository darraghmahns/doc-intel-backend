"""
Tests for Field Mapper Node - Dotloop API Field Mapping

M1 Acceptance Criteria: Field mapping matches Dotloop API schema exactly
- Loop-It API uses camelCase field names
- Loop Details API uses quoted section/field names
- Date format: MM/DD/YYYY
- All participant roles validated

M2 Acceptance Criteria: Signature fields assigned to correct participant roles
- BUYER signatures → buyer participant
- SELLER signatures → seller participant
- Agent, broker, and other role signatures → appropriate participants
"""

import pytest
from typing import Dict, Any, List, cast

from state import FinancialDetails
from nodes.mapper import (
    # Enums and constants
    DotloopTransactionType,
    DotloopStatus,
    DotloopParticipantRole,
    DotloopDetailSection,
    DOC_TYPE_TO_TRANSACTION,
    TRANSACTION_DEFAULT_STATUS,
    VALID_STATUSES_BY_TRANSACTION,
    LOOP_IT_PROPERTY_FIELDS,
    LOOP_IT_MLS_FIELDS,
    LOOP_IT_PARTICIPANT_FIELDS,
    LOOP_IT_REQUIRED_FIELDS,
    validate_detailed_for_loop_it,
    MAX_LOOP_NAME_LENGTH,
    DETAIL_PROPERTY_ADDRESS_FIELDS,
    DETAIL_FINANCIALS_FIELDS,
    DETAIL_CONTRACT_DATES_FIELDS,
    DETAIL_OFFER_DATES_FIELDS,
    SIGNATURE_ROLE_KEYWORDS,
    # Validation functions
    get_valid_participant_roles,
    validate_participant_role,
    normalize_participant_role,
    validate_participants,
    validate_transaction_type,
    validate_status,
    validate_loop_name,
    validate_for_dotloop,
    # Conversion functions
    convert_property_to_loop_it,
    convert_participant_to_loop_it,
    convert_participants_to_loop_it,
    apply_brokerage_defaults,
    # Payload builders
    determine_transaction_type,
    build_loop_name,
    build_loop_it_payload,
    build_dotloop_payload,
    build_loop_details_payload,
    build_participant_detail,
    # Signature mapping (M2)
    detect_role_from_label,
    detect_role_index_from_label,
    enrich_signature_with_role,
    enrich_all_signatures,
    get_participant_for_signature,
    build_signature_routing,
    map_signatures_to_participants,
    validate_signature_routing,
    # Main node
    field_mapper_node,
)
from state import DealState, ParticipantInfo, SignatureField


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_property_details() -> Dict[str, Any]:
    """Sample property details in snake_case (internal format)."""
    return {
        "street_number": "123",
        "street_name": "Main Street",
        "unit": "A",
        "city": "Denver",
        "state": "CO",
        "zip_code": "80202",
        "county": "Denver",
        "country": "US",
        "mls_number": "MLS12345",
        "full_address": "123 Main Street, Unit A, Denver, CO 80202",
    }


@pytest.fixture
def sample_participants() -> List[ParticipantInfo]:
    """Sample participants list."""
    return [
        {"full_name": "John Buyer", "email": "john@buyer.com", "role": "BUYER"},
        {"full_name": "Jane Seller", "email": "jane@seller.com", "role": "SELLER"},
        {"full_name": "Bob Agent", "email": "bob@realty.com", "role": "BUYING_AGENT"},
    ]


@pytest.fixture
def sample_financial_details() -> Dict[str, Any]:
    """Sample financial details."""
    return {
        "purchase_sale_price": 450000.00,
        "earnest_money_amount": 5000.00,
        "earnest_money_held_by": "Title Company",
        "sale_commission_rate": "6%",
    }


@pytest.fixture
def sample_contract_dates() -> Dict[str, Any]:
    """Sample contract dates."""
    return {
        "contract_agreement_date": "01/15/2024",
        "closing_date": "02/28/2024",
        "offer_date": "01/10/2024",
        "offer_expiration_date": "01/12/2024",
        "inspection_date": "01/20/2024",
    }


@pytest.fixture
def sample_deal_state(
    sample_property_details,
    sample_participants,
    sample_financial_details,
    sample_contract_dates,
) -> DealState:
    """Complete sample deal state."""
    return {
        "deal_id": "DEAL-001",
        "status": "Processing",
        "email_metadata": {"sender": "agent@realty.com", "subject": "New Contract"},
        "raw_pdf_path": "/path/to/contract.pdf",
        "split_docs": [{"id": 1, "doc_type": "Buy-Sell", "page_range": [1, 10]}],
        "property_address": "123 Main Street, Denver, CO 80202",
        "property_details": sample_property_details,
        "buyers": ["John Buyer"],
        "sellers": ["Jane Seller"],
        "participants": sample_participants,
        "financials": {"purchase_price": 450000.00},
        "financial_details": sample_financial_details,
        "contract_dates": sample_contract_dates,
        "signature_fields": [],
        "signature_mapping": {},
        "missing_docs": [],
        "validation_errors": [],
        "human_approval_status": "Pending",
        "target_system": "dotloop",
        "dotloop_payload": None,
        "brokerage_settings": None,
    }


# ============================================================================
# Transaction Type Enum Tests
# ============================================================================

class TestDotloopTransactionType:
    """Tests for DotloopTransactionType enum."""

    def test_purchase_offer_value(self):
        """PURCHASE_OFFER has correct value."""
        assert DotloopTransactionType.PURCHASE_OFFER.value == "PURCHASE_OFFER"

    def test_listing_for_sale_value(self):
        """LISTING_FOR_SALE has correct value."""
        assert DotloopTransactionType.LISTING_FOR_SALE.value == "LISTING_FOR_SALE"

    def test_listing_for_lease_value(self):
        """LISTING_FOR_LEASE has correct value."""
        assert DotloopTransactionType.LISTING_FOR_LEASE.value == "LISTING_FOR_LEASE"

    def test_lease_offer_value(self):
        """LEASE_OFFER has correct value."""
        assert DotloopTransactionType.LEASE_OFFER.value == "LEASE_OFFER"

    def test_real_estate_other_value(self):
        """REAL_ESTATE_OTHER has correct value."""
        assert DotloopTransactionType.REAL_ESTATE_OTHER.value == "REAL_ESTATE_OTHER"

    def test_other_value(self):
        """OTHER has correct value."""
        assert DotloopTransactionType.OTHER.value == "OTHER"

    def test_all_transaction_types_are_strings(self):
        """All transaction types are string enums."""
        for t_type in DotloopTransactionType:
            assert isinstance(t_type.value, str)


# ============================================================================
# Status Enum Tests
# ============================================================================

class TestDotloopStatus:
    """Tests for DotloopStatus enum."""

    def test_pre_offer_value(self):
        """PRE_OFFER has correct value."""
        assert DotloopStatus.PRE_OFFER.value == "PRE_OFFER"

    def test_under_contract_value(self):
        """UNDER_CONTRACT has correct value."""
        assert DotloopStatus.UNDER_CONTRACT.value == "UNDER_CONTRACT"

    def test_sold_value(self):
        """SOLD has correct value."""
        assert DotloopStatus.SOLD.value == "SOLD"

    def test_pre_listing_value(self):
        """PRE_LISTING has correct value."""
        assert DotloopStatus.PRE_LISTING.value == "PRE_LISTING"

    def test_active_listing_value(self):
        """ACTIVE_LISTING has correct value."""
        assert DotloopStatus.ACTIVE_LISTING.value == "ACTIVE_LISTING"

    def test_archived_value(self):
        """ARCHIVED has correct value."""
        assert DotloopStatus.ARCHIVED.value == "ARCHIVED"

    def test_leased_value(self):
        """LEASED has correct value."""
        assert DotloopStatus.LEASED.value == "LEASED"


# ============================================================================
# Participant Role Enum Tests
# ============================================================================

class TestDotloopParticipantRole:
    """Tests for DotloopParticipantRole enum."""

    def test_buyer_value(self):
        """BUYER has correct value."""
        assert DotloopParticipantRole.BUYER.value == "BUYER"

    def test_seller_value(self):
        """SELLER has correct value."""
        assert DotloopParticipantRole.SELLER.value == "SELLER"

    def test_listing_agent_value(self):
        """LISTING_AGENT has correct value."""
        assert DotloopParticipantRole.LISTING_AGENT.value == "LISTING_AGENT"

    def test_buying_agent_value(self):
        """BUYING_AGENT has correct value."""
        assert DotloopParticipantRole.BUYING_AGENT.value == "BUYING_AGENT"

    def test_escrow_title_rep_value(self):
        """ESCROW_TITLE_REP has correct value."""
        assert DotloopParticipantRole.ESCROW_TITLE_REP.value == "ESCROW_TITLE_REP"

    def test_loan_officer_value(self):
        """LOAN_OFFICER has correct value."""
        assert DotloopParticipantRole.LOAN_OFFICER.value == "LOAN_OFFICER"

    def test_landlord_value(self):
        """LANDLORD has correct value."""
        assert DotloopParticipantRole.LANDLORD.value == "LANDLORD"

    def test_tenant_value(self):
        """TENANT has correct value."""
        assert DotloopParticipantRole.TENANT.value == "TENANT"

    def test_all_roles_have_values(self):
        """All roles have non-empty values."""
        for role in DotloopParticipantRole:
            assert role.value
            assert isinstance(role.value, str)


# ============================================================================
# Detail Section Enum Tests
# ============================================================================

class TestDotloopDetailSection:
    """Tests for DotloopDetailSection enum (exact API section names)."""

    def test_property_address_exact_name(self):
        """Property Address section has exact API name."""
        assert DotloopDetailSection.PROPERTY_ADDRESS.value == "Property Address"

    def test_financials_exact_name(self):
        """Financials section has exact API name."""
        assert DotloopDetailSection.FINANCIALS.value == "Financials"

    def test_contract_dates_exact_name(self):
        """Contract Dates section has exact API name."""
        assert DotloopDetailSection.CONTRACT_DATES.value == "Contract Dates"

    def test_offer_dates_exact_name(self):
        """Offer Dates section has exact API name."""
        assert DotloopDetailSection.OFFER_DATES.value == "Offer Dates"


# ============================================================================
# Field Mapping Constants Tests
# ============================================================================

class TestFieldMappingConstants:
    """Tests for field mapping constants (snake_case -> API format)."""

    def test_property_street_name_camel_case(self):
        """street_name maps to camelCase streetName."""
        assert LOOP_IT_PROPERTY_FIELDS["street_name"] == "streetName"

    def test_property_street_number_camel_case(self):
        """street_number maps to camelCase streetNumber."""
        assert LOOP_IT_PROPERTY_FIELDS["street_number"] == "streetNumber"

    def test_property_zip_code_camel_case(self):
        """zip_code maps to camelCase zipCode."""
        assert LOOP_IT_PROPERTY_FIELDS["zip_code"] == "zipCode"

    def test_mls_property_id_camel_case(self):
        """mls_number maps to camelCase mlsPropertyId."""
        assert LOOP_IT_MLS_FIELDS["mls_number"] == "mlsPropertyId"

    def test_participant_full_name_camel_case(self):
        """full_name maps to camelCase fullName."""
        assert LOOP_IT_PARTICIPANT_FIELDS["full_name"] == "fullName"

    def test_loop_it_required_fields(self):
        """Loop-It API requires name, transactionType, status."""
        assert "name" in LOOP_IT_REQUIRED_FIELDS
        assert "transactionType" in LOOP_IT_REQUIRED_FIELDS
        assert "status" in LOOP_IT_REQUIRED_FIELDS

    def test_max_loop_name_length(self):
        """Max loop name is 200 characters per API."""
        assert MAX_LOOP_NAME_LENGTH == 200


class TestDetailFieldMappings:
    """Tests for Loop Details API field mappings."""

    def test_property_street_name_detail(self):
        """street_name maps to 'Street Name' for details."""
        assert DETAIL_PROPERTY_ADDRESS_FIELDS["street_name"] == "Street Name"

    def test_property_city_detail(self):
        """city maps to 'City' for details."""
        assert DETAIL_PROPERTY_ADDRESS_FIELDS["city"] == "City"

    def test_zip_code_detail(self):
        """zip_code maps to 'Zip/Postal Code' for details."""
        assert DETAIL_PROPERTY_ADDRESS_FIELDS["zip_code"] == "Zip/Postal Code"

    def test_purchase_price_detail(self):
        """purchase_sale_price maps to 'Purchase/Sale Price'."""
        assert DETAIL_FINANCIALS_FIELDS["purchase_sale_price"] == "Purchase/Sale Price"

    def test_earnest_money_detail(self):
        """earnest_money_amount maps to 'Earnest Money Amount'."""
        assert DETAIL_FINANCIALS_FIELDS["earnest_money_amount"] == "Earnest Money Amount"

    def test_closing_date_detail(self):
        """closing_date maps to 'Closing Date'."""
        assert DETAIL_CONTRACT_DATES_FIELDS["closing_date"] == "Closing Date"

    def test_offer_date_detail(self):
        """offer_date maps to 'Offer Date'."""
        assert DETAIL_OFFER_DATES_FIELDS["offer_date"] == "Offer Date"

    def test_inspection_date_detail(self):
        """inspection_date maps to 'Inspection Date'."""
        assert DETAIL_OFFER_DATES_FIELDS["inspection_date"] == "Inspection Date"


# ============================================================================
# Role Validation Tests
# ============================================================================

class TestRoleValidation:
    """Tests for participant role validation."""

    def test_get_valid_roles_includes_buyer(self):
        """Valid roles includes BUYER."""
        roles = get_valid_participant_roles()
        assert "BUYER" in roles

    def test_get_valid_roles_includes_seller(self):
        """Valid roles includes SELLER."""
        roles = get_valid_participant_roles()
        assert "SELLER" in roles

    def test_validate_buyer_role(self):
        """BUYER is a valid role."""
        assert validate_participant_role("BUYER") is True

    def test_validate_invalid_role(self):
        """Random string is not a valid role."""
        assert validate_participant_role("RANDOM_ROLE") is False

    def test_validate_empty_role(self):
        """Empty string is not valid."""
        assert validate_participant_role("") is False

    def test_normalize_purchaser_to_buyer(self):
        """PURCHASER normalizes to BUYER."""
        assert normalize_participant_role("PURCHASER") == "BUYER"

    def test_normalize_buyers_agent_to_buying_agent(self):
        """BUYERS_AGENT normalizes to BUYING_AGENT."""
        assert normalize_participant_role("BUYERS_AGENT") == "BUYING_AGENT"

    def test_normalize_sellers_agent_to_listing_agent(self):
        """SELLERS_AGENT normalizes to LISTING_AGENT."""
        assert normalize_participant_role("SELLERS_AGENT") == "LISTING_AGENT"

    def test_normalize_lender_to_loan_officer(self):
        """LENDER normalizes to LOAN_OFFICER."""
        assert normalize_participant_role("LENDER") == "LOAN_OFFICER"

    def test_normalize_title_company_to_escrow_title_rep(self):
        """TITLE_COMPANY normalizes to ESCROW_TITLE_REP."""
        assert normalize_participant_role("TITLE_COMPANY") == "ESCROW_TITLE_REP"

    def test_normalize_lessor_to_landlord(self):
        """LESSOR normalizes to LANDLORD."""
        assert normalize_participant_role("LESSOR") == "LANDLORD"

    def test_normalize_renter_to_tenant(self):
        """RENTER normalizes to TENANT."""
        assert normalize_participant_role("RENTER") == "TENANT"

    def test_normalize_unknown_to_other(self):
        """Unknown role normalizes to OTHER."""
        assert normalize_participant_role("UNKNOWN_ROLE") == "OTHER"

    def test_normalize_empty_to_other(self):
        """Empty role normalizes to OTHER."""
        assert normalize_participant_role("") == "OTHER"

    def test_normalize_case_insensitive(self):
        """Role normalization is case insensitive."""
        assert normalize_participant_role("buyer") == "BUYER"
        assert normalize_participant_role("Seller") == "SELLER"


# ============================================================================
# Transaction Type Validation Tests
# ============================================================================

class TestTransactionTypeValidation:
    """Tests for transaction type validation."""

    def test_purchase_offer_valid(self):
        """PURCHASE_OFFER is valid."""
        assert validate_transaction_type("PURCHASE_OFFER") is True

    def test_listing_for_sale_valid(self):
        """LISTING_FOR_SALE is valid."""
        assert validate_transaction_type("LISTING_FOR_SALE") is True

    def test_invalid_type_rejected(self):
        """Invalid transaction type is rejected."""
        assert validate_transaction_type("INVALID_TYPE") is False


class TestStatusValidation:
    """Tests for status validation."""

    def test_pre_offer_valid(self):
        """PRE_OFFER is valid status."""
        assert validate_status("PRE_OFFER") is True

    def test_under_contract_valid(self):
        """UNDER_CONTRACT is valid status."""
        assert validate_status("UNDER_CONTRACT") is True

    def test_invalid_status_rejected(self):
        """Invalid status is rejected."""
        assert validate_status("INVALID_STATUS") is False

    def test_pre_offer_valid_for_purchase(self):
        """PRE_OFFER is valid for PURCHASE_OFFER transaction."""
        assert validate_status("PRE_OFFER", "PURCHASE_OFFER") is True

    def test_active_listing_invalid_for_purchase(self):
        """ACTIVE_LISTING is not valid for PURCHASE_OFFER."""
        assert validate_status("ACTIVE_LISTING", "PURCHASE_OFFER") is False

    def test_active_listing_valid_for_listing_for_sale(self):
        """ACTIVE_LISTING is valid for LISTING_FOR_SALE."""
        assert validate_status("ACTIVE_LISTING", "LISTING_FOR_SALE") is True


class TestLoopNameValidation:
    """Tests for loop name validation."""

    def test_valid_name_passes(self):
        """Valid loop name returns no issues."""
        issues = validate_loop_name("123 Main Street, Denver, CO")
        assert len(issues) == 0

    def test_empty_name_fails(self):
        """Empty loop name returns issue."""
        issues = validate_loop_name("")
        assert len(issues) > 0
        assert "required" in issues[0].lower()

    def test_too_long_name_fails(self):
        """Loop name over 200 chars returns issue."""
        long_name = "A" * 201
        issues = validate_loop_name(long_name)
        assert len(issues) > 0
        assert "200" in issues[0]

    def test_exactly_200_chars_passes(self):
        """Loop name of exactly 200 chars passes."""
        exact_name = "A" * 200
        issues = validate_loop_name(exact_name)
        assert len(issues) == 0


# ============================================================================
# Property Conversion Tests
# ============================================================================

class TestPropertyConversion:
    """Tests for property details to Loop-It format conversion."""

    def test_converts_street_name_to_camel_case(self, sample_property_details):
        """street_name converts to streetName."""
        result = convert_property_to_loop_it(sample_property_details)
        assert result.get("streetName") == "Main Street"

    def test_converts_street_number_to_camel_case(self, sample_property_details):
        """street_number converts to streetNumber."""
        result = convert_property_to_loop_it(sample_property_details)
        assert result.get("streetNumber") == "123"

    def test_converts_zip_code_to_camel_case(self, sample_property_details):
        """zip_code converts to zipCode."""
        result = convert_property_to_loop_it(sample_property_details)
        assert result.get("zipCode") == "80202"

    def test_converts_mls_number_to_mls_property_id(self, sample_property_details):
        """mls_number converts to mlsPropertyId."""
        result = convert_property_to_loop_it(sample_property_details)
        assert result.get("mlsPropertyId") == "MLS12345"

    def test_city_unchanged(self, sample_property_details):
        """city stays as city."""
        result = convert_property_to_loop_it(sample_property_details)
        assert result.get("city") == "Denver"

    def test_state_unchanged(self, sample_property_details):
        """state stays as state."""
        result = convert_property_to_loop_it(sample_property_details)
        assert result.get("state") == "CO"

    def test_empty_property_returns_empty_dict(self):
        """Empty property returns empty dict."""
        result = convert_property_to_loop_it({})
        assert result == {}

    def test_none_property_returns_empty_dict(self):
        """None property returns empty dict."""
        result = convert_property_to_loop_it(None)
        assert result == {}

    def test_omits_missing_fields(self):
        """Fields not in input are omitted from output."""
        result = convert_property_to_loop_it({"city": "Denver"})
        assert "streetName" not in result
        assert "city" in result


# ============================================================================
# Participant Conversion Tests
# ============================================================================

class TestParticipantConversion:
    """Tests for participant to Loop-It format conversion."""

    def test_converts_full_name_to_camel_case(self):
        """full_name converts to fullName."""
        participant = {"full_name": "John Doe", "role": "BUYER"}
        result = convert_participant_to_loop_it(participant)
        assert result.get("fullName") == "John Doe"

    def test_email_unchanged(self):
        """email stays as email."""
        participant = {"full_name": "John Doe", "email": "john@example.com", "role": "BUYER"}
        result = convert_participant_to_loop_it(participant)
        assert result.get("email") == "john@example.com"

    def test_role_normalized(self):
        """role is normalized to valid Dotloop role."""
        participant = {"full_name": "John Doe", "role": "PURCHASER"}
        result = convert_participant_to_loop_it(participant)
        assert result.get("role") == "BUYER"

    def test_empty_participant_returns_minimal_dict(self):
        """Empty participant returns minimal dict."""
        result = convert_participant_to_loop_it({})
        assert "fullName" not in result

    def test_convert_multiple_participants(self, sample_participants):
        """Multiple participants converted correctly."""
        results = convert_participants_to_loop_it(sample_participants)
        assert len(results) == 3
        assert results[0]["fullName"] == "John Buyer"
        assert results[0]["role"] == "BUYER"

    def test_skips_participants_without_name(self):
        """Participants without full_name are skipped."""
        participants = [
            {"full_name": "John Doe", "role": "BUYER"},
            {"role": "SELLER"},  # No name
        ]
        results = convert_participants_to_loop_it(participants)
        assert len(results) == 1


# ============================================================================
# Loop-It Payload Builder Tests
# ============================================================================

class TestBuildLoopItPayload:
    """Tests for Loop-It API payload builder."""

    def test_payload_has_required_name(self, sample_deal_state):
        """Payload has required 'name' field."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "name" in payload
        assert payload["name"]

    def test_payload_has_required_transaction_type(self, sample_deal_state):
        """Payload has required 'transactionType' field (camelCase)."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "transactionType" in payload
        assert payload["transactionType"] == "PURCHASE_OFFER"

    def test_payload_has_required_status(self, sample_deal_state):
        """Payload has required 'status' field."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "status" in payload
        assert payload["status"] == "PRE_OFFER"

    def test_payload_has_camel_case_street_name(self, sample_deal_state):
        """Payload uses camelCase for streetName."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "streetName" in payload
        assert payload["streetName"] == "Main Street"
        assert "street_name" not in payload

    def test_payload_has_camel_case_street_number(self, sample_deal_state):
        """Payload uses camelCase for streetNumber."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "streetNumber" in payload
        assert payload["streetNumber"] == "123"

    def test_payload_has_camel_case_zip_code(self, sample_deal_state):
        """Payload uses camelCase for zipCode."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "zipCode" in payload
        assert payload["zipCode"] == "80202"

    def test_payload_participants_use_camel_case(self, sample_deal_state):
        """Participants use camelCase fullName."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "participants" in payload
        assert len(payload["participants"]) > 0
        first_participant = payload["participants"][0]
        assert "fullName" in first_participant
        assert "full_name" not in first_participant

    def test_name_uses_full_address(self, sample_deal_state):
        """Loop name prefers full_address when available."""
        payload = build_loop_it_payload(sample_deal_state)
        assert "123 Main Street" in payload["name"]

    def test_name_truncated_to_max_length(self, sample_deal_state):
        """Loop name is truncated to 200 characters."""
        sample_deal_state["property_details"]["full_address"] = "A" * 250
        payload = build_loop_it_payload(sample_deal_state)
        assert len(payload["name"]) <= 200

    def test_name_falls_back_to_buyer_name(self, sample_deal_state):
        """Loop name falls back to buyer name if no address."""
        sample_deal_state["property_details"] = None
        payload = build_loop_it_payload(sample_deal_state)
        assert payload["name"] == "John Buyer"

    def test_name_falls_back_to_default(self):
        """Loop name uses default when no address or participants."""
        state = {
            "property_details": None,
            "participants": [],
            "split_docs": [],
        }
        payload = build_loop_it_payload(state)
        assert payload["name"] == "New Transaction"


class TestDetermineTransactionType:
    """Tests for transaction type determination."""

    def test_buy_sell_returns_purchase_offer(self):
        """Buy-Sell document returns PURCHASE_OFFER."""
        state = {"split_docs": [{"doc_type": "Buy-Sell"}]}
        assert determine_transaction_type(state) == "PURCHASE_OFFER"

    def test_listing_agreement_returns_listing_for_sale(self):
        """Listing Agreement returns LISTING_FOR_SALE."""
        state = {"split_docs": [{"doc_type": "Listing Agreement"}]}
        assert determine_transaction_type(state) == "LISTING_FOR_SALE"

    def test_lease_agreement_returns_lease_offer(self):
        """Lease Agreement returns LEASE_OFFER."""
        state = {"split_docs": [{"doc_type": "Lease Agreement"}]}
        assert determine_transaction_type(state) == "LEASE_OFFER"

    def test_unknown_doc_returns_real_estate_other(self):
        """Unknown document returns REAL_ESTATE_OTHER."""
        state = {"split_docs": [{"doc_type": "Unknown"}]}
        assert determine_transaction_type(state) == "REAL_ESTATE_OTHER"

    def test_empty_docs_returns_purchase_offer_default(self):
        """Empty docs list returns PURCHASE_OFFER default."""
        state = {"split_docs": []}
        assert determine_transaction_type(state) == "PURCHASE_OFFER"


# ============================================================================
# Loop Details Payload Builder Tests
# ============================================================================

class TestBuildLoopDetailsPayload:
    """Tests for Loop Details API payload builder."""

    def test_has_property_address_section(self, sample_deal_state):
        """Payload has 'Property Address' section."""
        payload = build_loop_details_payload(sample_deal_state)
        assert "Property Address" in payload

    def test_property_address_uses_quoted_field_names(self, sample_deal_state):
        """Property Address uses exact API field names."""
        payload = build_loop_details_payload(sample_deal_state)
        prop_section = payload["Property Address"]
        assert "Street Name" in prop_section
        assert "City" in prop_section
        assert "Zip/Postal Code" in prop_section

    def test_has_financials_section(self, sample_deal_state):
        """Payload has 'Financials' section."""
        payload = build_loop_details_payload(sample_deal_state)
        assert "Financials" in payload

    def test_financials_uses_quoted_field_names(self, sample_deal_state):
        """Financials uses exact API field names."""
        payload = build_loop_details_payload(sample_deal_state)
        fin_section = payload["Financials"]
        assert "Purchase/Sale Price" in fin_section

    def test_has_contract_dates_section(self, sample_deal_state):
        """Payload has 'Contract Dates' section."""
        payload = build_loop_details_payload(sample_deal_state)
        assert "Contract Dates" in payload

    def test_contract_dates_uses_quoted_field_names(self, sample_deal_state):
        """Contract Dates uses exact API field names."""
        payload = build_loop_details_payload(sample_deal_state)
        dates_section = payload["Contract Dates"]
        assert "Contract Agreement Date" in dates_section
        assert "Closing Date" in dates_section

    def test_has_offer_dates_section(self, sample_deal_state):
        """Payload has 'Offer Dates' section."""
        payload = build_loop_details_payload(sample_deal_state)
        assert "Offer Dates" in payload

    def test_offer_dates_uses_quoted_field_names(self, sample_deal_state):
        """Offer Dates uses exact API field names."""
        payload = build_loop_details_payload(sample_deal_state)
        offer_section = payload["Offer Dates"]
        assert "Offer Date" in offer_section
        assert "Inspection Date" in offer_section

    def test_empty_state_returns_empty_dict(self):
        """Empty state returns empty payload."""
        state = {
            "property_details": None,
            "financial_details": None,
            "contract_dates": None,
        }
        payload = build_loop_details_payload(state)
        assert payload == {}

    def test_partial_state_only_includes_available_sections(self):
        """Only sections with data are included."""
        state = {
            "property_details": {"city": "Denver"},
            "financial_details": None,
            "contract_dates": None,
        }
        payload = build_loop_details_payload(state)
        assert "Property Address" in payload
        assert "Financials" not in payload
        assert "Contract Dates" not in payload


# ============================================================================
# Participant Detail Builder Tests
# ============================================================================

class TestBuildParticipantDetail:
    """Tests for participant detail builder."""

    def test_includes_name(self):
        """Detail includes Name field."""
        participant = {"full_name": "John Doe"}
        detail = build_participant_detail(participant)
        assert detail.get("Name") == "John Doe"

    def test_includes_email(self):
        """Detail includes Email field."""
        participant = {"full_name": "John Doe", "email": "john@example.com"}
        detail = build_participant_detail(participant)
        assert detail.get("Email") == "john@example.com"

    def test_includes_phone(self):
        """Detail includes Phone field."""
        participant = {"full_name": "John Doe", "phone": "555-1234"}
        detail = build_participant_detail(participant)
        assert detail.get("Phone") == "555-1234"

    def test_includes_company(self):
        """Detail includes Company field."""
        participant = {"full_name": "John Doe", "company_name": "Acme Realty"}
        detail = build_participant_detail(participant)
        assert detail.get("Company") == "Acme Realty"

    def test_builds_street_address(self):
        """Detail builds Street Address from components."""
        participant = {
            "full_name": "John Doe",
            "street_number": "123",
            "street_name": "Main St",
        }
        detail = build_participant_detail(participant)
        assert detail.get("Street Address") == "123 Main St"


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidateParticipants:
    """Tests for participant validation."""

    def test_valid_participants_no_issues(self, sample_participants):
        """Valid participants return no issues."""
        issues = validate_participants(sample_participants)
        assert len(issues) == 0

    def test_missing_buyer_reported(self):
        """Missing buyer is reported."""
        participants = [{"full_name": "Jane Seller", "role": "SELLER"}]
        issues = validate_participants(participants)
        assert any("buyer" in i.lower() or "tenant" in i.lower() for i in issues)

    def test_missing_seller_reported(self):
        """Missing seller is reported."""
        participants = [{"full_name": "John Buyer", "role": "BUYER"}]
        issues = validate_participants(participants)
        assert any("seller" in i.lower() or "landlord" in i.lower() for i in issues)

    def test_tenant_satisfies_buyer_requirement(self):
        """TENANT satisfies buyer/tenant requirement."""
        participants = [
            {"full_name": "John Tenant", "role": "TENANT"},
            {"full_name": "Jane Landlord", "role": "LANDLORD"},
        ]
        issues = validate_participants(participants)
        # Should not report missing buyer/tenant or seller/landlord
        buyer_issues = [i for i in issues if "buyer" in i.lower() or "tenant" in i.lower()]
        seller_issues = [i for i in issues if "seller" in i.lower() or "landlord" in i.lower()]
        assert len(buyer_issues) == 0
        assert len(seller_issues) == 0

    def test_missing_name_reported(self):
        """Missing participant name is reported."""
        participants = [
            {"full_name": "John Buyer", "role": "BUYER"},
            {"role": "SELLER"},  # No name
        ]
        issues = validate_participants(participants)
        assert any("missing name" in i.lower() for i in issues)

    def test_invalid_role_reported(self):
        """Invalid role is reported."""
        participants = [
            {"full_name": "John Buyer", "role": "BUYER"},
            {"full_name": "Jane Seller", "role": "SELLER"},
            {"full_name": "Bob", "role": "INVALID_ROLE"},
        ]
        issues = validate_participants(participants)
        assert any("invalid role" in i.lower() for i in issues)


class TestValidateForDotloop:
    """Tests for full Dotloop validation."""

    def test_valid_state_minimal_issues(self, sample_deal_state):
        """Valid state returns minimal or no issues."""
        issues = validate_for_dotloop(sample_deal_state)
        # May have "recommended" warnings but no critical issues
        critical = [i for i in issues if "recommended" not in i.lower()]
        assert len(critical) == 0

    def test_missing_property_details_reported(self, sample_deal_state):
        """Missing property details is reported."""
        sample_deal_state["property_details"] = None
        issues = validate_for_dotloop(sample_deal_state)
        # Validation now itemizes errors (e.g. missing street, city, etc.)
        assert any("streetname" in i.lower().replace(" ", "") for i in issues)

    def test_missing_street_name_reported(self, sample_deal_state):
        """Missing street name is reported."""
        sample_deal_state["property_details"]["street_name"] = None
        issues = validate_for_dotloop(sample_deal_state)
        assert any("street name" in i.lower() for i in issues)

    def test_missing_city_reported(self, sample_deal_state):
        """Missing city is reported."""
        sample_deal_state["property_details"]["city"] = None
        issues = validate_for_dotloop(sample_deal_state)
        assert any("city" in i.lower() for i in issues)


# ============================================================================
# Signature Mapping Tests
# ============================================================================

class TestMapSignaturesToParticipants:
    """Tests for signature to participant mapping."""

    def test_maps_buyer_signature(self, sample_participants):
        """Buyer signature mapped to buyer participant."""
        sig_fields = [
            {
                "page_number": 1,
                "x_position": 100.0,
                "y_position": 200.0,
                "width": 150.0,
                "height": 30.0,
                "field_type": "signature",
                "label": "Buyer Signature",
                "required": True,
                "assigned_role": "BUYER",
                "context_text": "Buyer:",
            }
        ]
        result = map_signatures_to_participants(sig_fields, sample_participants)
        assert "John Buyer" in result
        assert len(result["John Buyer"]) == 1

    def test_maps_seller_signature(self, sample_participants):
        """Seller signature mapped to seller participant."""
        sig_fields = [
            {
                "page_number": 2,
                "x_position": 100.0,
                "y_position": 200.0,
                "width": 150.0,
                "height": 30.0,
                "field_type": "signature",
                "label": "Seller Signature",
                "required": True,
                "assigned_role": "SELLER",
                "context_text": "Seller:",
            }
        ]
        result = map_signatures_to_participants(sig_fields, sample_participants)
        assert "Jane Seller" in result

    def test_normalizes_role_aliases(self, sample_participants):
        """Role aliases are normalized before mapping."""
        sig_fields = [
            {
                "page_number": 1,
                "x_position": 100.0,
                "y_position": 200.0,
                "width": 150.0,
                "height": 30.0,
                "field_type": "signature",
                "label": "Purchaser Signature",
                "required": True,
                "assigned_role": "PURCHASER",  # Alias for BUYER
                "context_text": "Purchaser:",
            }
        ]
        result = map_signatures_to_participants(sig_fields, sample_participants)
        # Should map to John Buyer (BUYER role)
        assert "John Buyer" in result

    def test_empty_signatures_returns_empty_dict(self, sample_participants):
        """Empty signature list returns empty mapping."""
        result = map_signatures_to_participants([], sample_participants)
        assert result == {}


# ============================================================================
# Field Mapper Node Integration Tests
# ============================================================================

class TestFieldMapperNode:
    """Integration tests for field_mapper_node."""

    def test_returns_pending_approval_status(self, sample_deal_state):
        """Node returns pending human_approval_status."""
        result = field_mapper_node(sample_deal_state)
        assert result["human_approval_status"] == "Pending"

    def test_returns_dotloop_payload(self, sample_deal_state):
        """Node returns dotloop_payload (backward compat)."""
        result = field_mapper_node(sample_deal_state)
        assert "dotloop_payload" in result

    def test_returns_loop_it_payload(self, sample_deal_state):
        """Node returns loop_it_payload with camelCase."""
        result = field_mapper_node(sample_deal_state)
        assert "loop_it_payload" in result
        payload = result["loop_it_payload"]
        assert "transactionType" in payload
        assert "streetName" in payload

    def test_returns_loop_details_payload(self, sample_deal_state):
        """Node returns loop_details_payload."""
        result = field_mapper_node(sample_deal_state)
        assert "loop_details_payload" in result

    def test_returns_participant_signature_map(self, sample_deal_state):
        """Node returns participant_signature_map for routing."""
        result = field_mapper_node(sample_deal_state)
        assert "participant_signature_map" in result
        
    def test_ready_for_review_status_when_valid(self, sample_deal_state):
        """Status is Ready_For_Review when no critical issues."""
        result = field_mapper_node(sample_deal_state)
        assert result["status"] == "Ready_For_Review"

    def test_needs_review_status_when_missing_data(self, sample_deal_state):
        """Status is Needs_Review when critical data missing."""
        sample_deal_state["property_details"] = None
        result = field_mapper_node(sample_deal_state)
        assert result["status"] == "Needs_Review"
        assert "missing_docs" in result


# ============================================================================
# Doc Type to Transaction Mapping Tests
# ============================================================================

class TestDocTypeToTransactionMapping:
    """Tests for document type to transaction type mapping."""

    def test_buy_sell_maps_to_purchase_offer(self):
        """Buy-Sell maps to PURCHASE_OFFER."""
        assert DOC_TYPE_TO_TRANSACTION["Buy-Sell"] == "PURCHASE_OFFER"

    def test_purchase_agreement_maps_to_purchase_offer(self):
        """Purchase Agreement maps to PURCHASE_OFFER."""
        assert DOC_TYPE_TO_TRANSACTION["Purchase Agreement"] == "PURCHASE_OFFER"

    def test_listing_agreement_maps_to_listing_for_sale(self):
        """Listing Agreement maps to LISTING_FOR_SALE."""
        assert DOC_TYPE_TO_TRANSACTION["Listing Agreement"] == "LISTING_FOR_SALE"

    def test_lease_agreement_maps_to_lease_offer(self):
        """Lease Agreement maps to LEASE_OFFER."""
        assert DOC_TYPE_TO_TRANSACTION["Lease Agreement"] == "LEASE_OFFER"

    def test_counter_offer_maps_to_purchase_offer(self):
        """Counter Offer maps to PURCHASE_OFFER."""
        assert DOC_TYPE_TO_TRANSACTION["Counter Offer"] == "PURCHASE_OFFER"


class TestTransactionDefaultStatus:
    """Tests for transaction type to default status mapping."""

    def test_purchase_offer_defaults_to_pre_offer(self):
        """PURCHASE_OFFER defaults to PRE_OFFER."""
        assert TRANSACTION_DEFAULT_STATUS["PURCHASE_OFFER"] == "PRE_OFFER"

    def test_listing_for_sale_defaults_to_pre_listing(self):
        """LISTING_FOR_SALE defaults to PRE_LISTING."""
        assert TRANSACTION_DEFAULT_STATUS["LISTING_FOR_SALE"] == "PRE_LISTING"

    def test_lease_offer_defaults_to_pre_offer(self):
        """LEASE_OFFER defaults to PRE_OFFER."""
        assert TRANSACTION_DEFAULT_STATUS["LEASE_OFFER"] == "PRE_OFFER"


class TestValidStatusesByTransaction:
    """Tests for valid statuses per transaction type."""

    def test_purchase_offer_valid_statuses(self):
        """PURCHASE_OFFER has correct valid statuses."""
        valid = VALID_STATUSES_BY_TRANSACTION["PURCHASE_OFFER"]
        assert "PRE_OFFER" in valid
        assert "UNDER_CONTRACT" in valid
        assert "SOLD" in valid
        assert "ARCHIVED" in valid

    def test_listing_for_sale_valid_statuses(self):
        """LISTING_FOR_SALE has correct valid statuses."""
        valid = VALID_STATUSES_BY_TRANSACTION["LISTING_FOR_SALE"]
        assert "PRE_LISTING" in valid
        assert "ACTIVE_LISTING" in valid
        assert "UNDER_CONTRACT" in valid
        assert "SOLD" in valid


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_none_participants(self):
        """Handles None participants gracefully."""
        state = {
            "property_details": {"city": "Denver"},
            "participants": None,
            "split_docs": [],
        }
        # Should not raise
        payload = build_loop_it_payload(state)
        assert "participants" not in payload or payload["participants"] == []

    def test_handles_empty_string_values(self):
        """Empty strings are excluded from payload."""
        property_details = {
            "city": "Denver",
            "street_name": "",  # Empty string
        }
        result = convert_property_to_loop_it(property_details)
        assert "streetName" not in result
        assert result["city"] == "Denver"

    def test_handles_unicode_in_names(self):
        """Unicode characters in names are handled."""
        participant = {"full_name": "José García", "role": "BUYER"}
        result = convert_participant_to_loop_it(participant)
        assert result["fullName"] == "José García"

    def test_handles_very_long_address(self):
        """Very long addresses are truncated for loop name."""
        property_details = {
            "full_address": "A" * 500,
        }
        name = build_loop_name(property_details, [])
        assert len(name) <= 200

    def test_handles_special_characters_in_address(self):
        """Special characters in address are preserved."""
        property_details = {
            "street_name": "O'Connor's Way",
            "city": "San José",
        }
        result = convert_property_to_loop_it(property_details)
        assert result["streetName"] == "O'Connor's Way"
        assert result["city"] == "San José"


# ============================================================================
# M2: Signature Role Detection Tests
# ============================================================================

class TestSignatureRoleKeywords:
    """Tests for signature role keyword constants."""

    def test_buyer_keywords_exist(self):
        """BUYER keywords include common variations."""
        keywords = SIGNATURE_ROLE_KEYWORDS.get("BUYER", [])
        assert "buyer" in keywords
        assert "purchaser" in keywords

    def test_seller_keywords_exist(self):
        """SELLER keywords include common variations."""
        keywords = SIGNATURE_ROLE_KEYWORDS.get("SELLER", [])
        assert "seller" in keywords
        assert "vendor" in keywords

    def test_tenant_keywords_exist(self):
        """TENANT keywords include lease terms."""
        keywords = SIGNATURE_ROLE_KEYWORDS.get("TENANT", [])
        assert "tenant" in keywords
        assert "lessee" in keywords

    def test_agent_keywords_exist(self):
        """Agent keywords exist for both listing and buying."""
        assert "LISTING_AGENT" in SIGNATURE_ROLE_KEYWORDS
        assert "BUYING_AGENT" in SIGNATURE_ROLE_KEYWORDS


class TestDetectRoleFromLabel:
    """Tests for detect_role_from_label function."""

    def test_detects_buyer_from_label(self):
        """Detects BUYER from 'Buyer Signature' label."""
        assert detect_role_from_label("Buyer Signature") == "BUYER"

    def test_detects_buyer_from_purchaser(self):
        """Detects BUYER from 'Purchaser' label."""
        assert detect_role_from_label("Purchaser Sign Here") == "BUYER"

    def test_detects_seller_from_label(self):
        """Detects SELLER from 'Seller Signature' label."""
        assert detect_role_from_label("Seller Signature") == "SELLER"

    def test_detects_seller_from_vendor(self):
        """Detects SELLER from 'Vendor' label."""
        assert detect_role_from_label("Vendor Initial") == "SELLER"

    def test_detects_tenant_from_label(self):
        """Detects TENANT from 'Tenant Signature' label."""
        assert detect_role_from_label("Tenant Signature") == "TENANT"

    def test_detects_lessee_as_tenant(self):
        """Detects TENANT from 'Lessee' label."""
        assert detect_role_from_label("Lessee Sign Here") == "TENANT"

    def test_detects_landlord_from_label(self):
        """Detects LANDLORD from 'Landlord Signature' label."""
        assert detect_role_from_label("Landlord Signature") == "LANDLORD"

    def test_detects_lessor_as_landlord(self):
        """Detects LANDLORD from 'Lessor' label."""
        assert detect_role_from_label("Lessor Initial") == "LANDLORD"

    def test_detects_listing_agent(self):
        """Detects LISTING_AGENT from label."""
        assert detect_role_from_label("Listing Agent Signature") == "LISTING_AGENT"

    def test_detects_sellers_agent_as_listing_agent(self):
        """Detects LISTING_AGENT from 'Seller's Agent' label."""
        assert detect_role_from_label("Seller's Agent") == "LISTING_AGENT"

    def test_detects_buying_agent(self):
        """Detects BUYING_AGENT from label."""
        assert detect_role_from_label("Buying Agent Signature") == "BUYING_AGENT"

    def test_detects_buyers_agent_as_buying_agent(self):
        """Detects BUYING_AGENT from 'Buyer's Agent' label."""
        assert detect_role_from_label("Buyer's Agent") == "BUYING_AGENT"

    def test_detects_escrow_title_rep(self):
        """Detects ESCROW_TITLE_REP from 'Escrow Officer' label."""
        assert detect_role_from_label("Escrow Officer") == "ESCROW_TITLE_REP"

    def test_detects_title_company(self):
        """Detects ESCROW_TITLE_REP from 'Title Company' label."""
        assert detect_role_from_label("Title Company Representative") == "ESCROW_TITLE_REP"

    def test_detects_loan_officer(self):
        """Detects LOAN_OFFICER from label."""
        assert detect_role_from_label("Loan Officer Signature") == "LOAN_OFFICER"

    def test_detects_lender_as_loan_officer(self):
        """Detects LOAN_OFFICER from 'Lender' label."""
        assert detect_role_from_label("Lender Approval") == "LOAN_OFFICER"

    def test_detects_attorney(self):
        """Detects ATTORNEY from label."""
        assert detect_role_from_label("Attorney Signature") == "ATTORNEY"

    def test_uses_context_text_for_detection(self):
        """Uses context_text when label is generic."""
        assert detect_role_from_label("Sign Here", "Buyer must sign below") == "BUYER"

    def test_returns_none_for_unknown(self):
        """Returns None when role cannot be detected."""
        assert detect_role_from_label("X_______________") is None

    def test_case_insensitive(self):
        """Role detection is case insensitive."""
        assert detect_role_from_label("BUYER SIGNATURE") == "BUYER"
        assert detect_role_from_label("buyer signature") == "BUYER"

    def test_handles_empty_label(self):
        """Handles empty label gracefully."""
        assert detect_role_from_label("") is None
        assert detect_role_from_label(None) is None


class TestDetectRoleIndexFromLabel:
    """Tests for detecting participant index from label."""

    def test_buyer_1_returns_index_0(self):
        """'Buyer 1' returns index 0."""
        assert detect_role_index_from_label("Buyer 1 Signature") == 0

    def test_buyer_2_returns_index_1(self):
        """'Buyer 2' returns index 1."""
        assert detect_role_index_from_label("Buyer 2 Signature") == 1

    def test_buyer_hash_2_returns_index_1(self):
        """'Buyer #2' returns index 1."""
        assert detect_role_index_from_label("Buyer #2 Signature") == 1

    def test_seller_1_returns_index_0(self):
        """'Seller 1' returns index 0."""
        assert detect_role_index_from_label("Seller 1 Initial") == 0

    def test_seller_2_returns_index_1(self):
        """'Seller 2' returns index 1."""
        assert detect_role_index_from_label("Seller 2 Initial") == 1

    def test_co_buyer_returns_index_1(self):
        """'Co-Buyer' returns index 1 (second buyer)."""
        assert detect_role_index_from_label("Co-Buyer Signature") == 1

    def test_additional_buyer_returns_index_1(self):
        """'Additional Buyer' returns index 1."""
        assert detect_role_index_from_label("Additional Buyer Sign Here") == 1

    def test_first_buyer_returns_index_0(self):
        """'First Buyer' returns index 0."""
        assert detect_role_index_from_label("First Buyer Signature") == 0

    def test_primary_seller_returns_index_0(self):
        """'Primary Seller' returns index 0."""
        assert detect_role_index_from_label("Primary Seller") == 0

    def test_default_returns_index_0(self):
        """Default is index 0 for plain labels."""
        assert detect_role_index_from_label("Buyer Signature") == 0


class TestEnrichSignatureWithRole:
    """Tests for signature enrichment with role detection."""

    def test_enriches_with_detected_role(self):
        """Enriches signature with detected role from label."""
        sig = {
            "label": "Buyer Signature",
            "page_number": 1,
            "x_position": 100,
            "y_position": 200,
        }
        result = enrich_signature_with_role(sig)
        assert result["assigned_role"] == "BUYER"
        assert result["role_detected"] is True

    def test_preserves_existing_role(self):
        """Preserves and normalizes existing assigned_role."""
        sig = {
            "label": "Sign Here",
            "assigned_role": "SELLER",
            "page_number": 1,
        }
        result = enrich_signature_with_role(sig)
        assert result["assigned_role"] == "SELLER"
        assert result["role_detected"] is False

    def test_normalizes_existing_role(self):
        """Normalizes existing role (e.g., purchaser -> BUYER)."""
        sig = {
            "label": "Sign Here",
            "assigned_role": "purchaser",
            "page_number": 1,
        }
        result = enrich_signature_with_role(sig)
        assert result["assigned_role"] == "BUYER"

    def test_uses_default_role_when_undetected(self):
        """Uses default role when detection fails."""
        sig = {
            "label": "X___",
            "page_number": 1,
        }
        result = enrich_signature_with_role(sig, default_role="OTHER")
        assert result["assigned_role"] == "OTHER"
        assert result["role_detected"] is False

    def test_adds_participant_index(self):
        """Adds participant_index based on label."""
        sig = {
            "label": "Buyer 2 Signature",
            "page_number": 1,
        }
        result = enrich_signature_with_role(sig)
        assert result["participant_index"] == 1


class TestEnrichAllSignatures:
    """Tests for enriching multiple signatures."""

    def test_enriches_all_signatures(self):
        """Enriches a list of signatures."""
        sigs = [
            {"label": "Buyer Signature", "page_number": 1},
            {"label": "Seller Signature", "page_number": 2},
        ]
        results = enrich_all_signatures(sigs)
        assert len(results) == 2
        assert results[0]["assigned_role"] == "BUYER"
        assert results[1]["assigned_role"] == "SELLER"


class TestGetParticipantForSignature:
    """Tests for matching signatures to participants."""

    def test_matches_buyer_signature_to_buyer(self):
        """BUYER signature matches buyer participant."""
        sig = {"assigned_role": "BUYER", "participant_index": 0}
        role_to_participants = {
            "BUYER": [{"full_name": "John Buyer", "role": "BUYER"}],
            "SELLER": [{"full_name": "Jane Seller", "role": "SELLER"}],
        }
        result = get_participant_for_signature(sig, role_to_participants)
        assert result is not None
        assert result["full_name"] == "John Buyer"

    def test_matches_seller_signature_to_seller(self):
        """SELLER signature matches seller participant."""
        sig = {"assigned_role": "SELLER", "participant_index": 0}
        role_to_participants = {
            "BUYER": [{"full_name": "John Buyer", "role": "BUYER"}],
            "SELLER": [{"full_name": "Jane Seller", "role": "SELLER"}],
        }
        result = get_participant_for_signature(sig, role_to_participants)
        assert result is not None
        assert result["full_name"] == "Jane Seller"

    def test_matches_buyer_2_to_second_buyer(self):
        """Buyer 2 signature matches second buyer participant."""
        sig = {"assigned_role": "BUYER", "participant_index": 1}
        role_to_participants = {
            "BUYER": [
                {"full_name": "John Buyer", "role": "BUYER"},
                {"full_name": "Mary Buyer", "role": "BUYER"},
            ],
        }
        result = get_participant_for_signature(sig, role_to_participants)
        assert result is not None
        assert result["full_name"] == "Mary Buyer"

    def test_falls_back_to_first_when_index_out_of_range(self):
        """Falls back to first participant when index exceeds count."""
        sig = {"assigned_role": "BUYER", "participant_index": 5}
        role_to_participants = {
            "BUYER": [{"full_name": "John Buyer", "role": "BUYER"}],
        }
        result = get_participant_for_signature(sig, role_to_participants)
        assert result is not None
        assert result["full_name"] == "John Buyer"

    def test_returns_none_when_no_matching_role(self):
        """Returns None when no participant has the role."""
        sig = {"assigned_role": "ATTORNEY", "participant_index": 0}
        role_to_participants = {
            "BUYER": [{"full_name": "John Buyer", "role": "BUYER"}],
        }
        result = get_participant_for_signature(sig, role_to_participants)
        assert result is None


class TestBuildSignatureRouting:
    """Tests for comprehensive signature routing."""

    @pytest.fixture
    def sample_signatures(self) -> List[Dict[str, Any]]:
        """Sample signature fields for testing."""
        return [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "width": 150,
                "height": 30,
                "field_type": "signature",
                "label": "Buyer Signature",
                "required": True,
                "context_text": "Buyer must sign below",
            },
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 300,
                "width": 150,
                "height": 30,
                "field_type": "signature",
                "label": "Seller Signature",
                "required": True,
                "context_text": "Seller must sign below",
            },
            {
                "page_number": 2,
                "x_position": 100,
                "y_position": 200,
                "width": 150,
                "height": 30,
                "field_type": "initial",
                "label": "Buyer Initial",
                "required": True,
                "context_text": None,
            },
        ]

    @pytest.fixture
    def sample_participants_multi(self) -> List[Dict[str, Any]]:
        """Sample participants including multiple buyers."""
        return [
            {"full_name": "John Buyer", "email": "john@buyer.com", "role": "BUYER"},
            {"full_name": "Mary Buyer", "email": "mary@buyer.com", "role": "BUYER"},
            {"full_name": "Jane Seller", "email": "jane@seller.com", "role": "SELLER"},
            {"full_name": "Bob Agent", "email": "bob@realty.com", "role": "BUYING_AGENT"},
        ]

    def test_builds_by_participant_mapping(self, sample_signatures, sample_participants_multi):
        """Routing includes by_participant mapping."""
        routing = build_signature_routing(sample_signatures, sample_participants_multi)
        assert "by_participant" in routing
        assert "John Buyer" in routing["by_participant"]
        assert "Jane Seller" in routing["by_participant"]

    def test_builds_by_role_mapping(self, sample_signatures, sample_participants_multi):
        """Routing includes by_role mapping."""
        routing = build_signature_routing(sample_signatures, sample_participants_multi)
        assert "by_role" in routing
        assert "BUYER" in routing["by_role"]
        assert "SELLER" in routing["by_role"]

    def test_includes_unassigned_signatures(self, sample_participants_multi):
        """Routing includes unassigned signatures."""
        sigs = [
            {"label": "Unknown Signature", "page_number": 1, "field_type": "signature"},
        ]
        routing = build_signature_routing(sigs, sample_participants_multi)
        assert "unassigned" in routing

    def test_includes_routing_order(self, sample_signatures, sample_participants_multi):
        """Routing includes signing order."""
        routing = build_signature_routing(sample_signatures, sample_participants_multi)
        assert "routing_order" in routing
        assert isinstance(routing["routing_order"], list)

    def test_includes_summary(self, sample_signatures, sample_participants_multi):
        """Routing includes role summary counts."""
        routing = build_signature_routing(sample_signatures, sample_participants_multi)
        assert "summary" in routing
        assert routing["summary"]["BUYER"] == 2  # signature + initial
        assert routing["summary"]["SELLER"] == 1


class TestMapSignaturesToParticipantsM2:
    """M2 Acceptance Criteria: Signature fields assigned to correct roles."""

    @pytest.fixture
    def multi_participant_list(self) -> List[Dict[str, Any]]:
        """Multiple buyers and sellers for testing."""
        return [
            {"full_name": "John Buyer", "role": "BUYER"},
            {"full_name": "Mary CoBuyer", "role": "BUYER"},
            {"full_name": "Jane Seller", "role": "SELLER"},
            {"full_name": "Tom CoSeller", "role": "SELLER"},
            {"full_name": "Alice Agent", "role": "LISTING_AGENT"},
            {"full_name": "Bob Agent", "role": "BUYING_AGENT"},
        ]

    def test_buyer_signature_goes_to_buyer(self, multi_participant_list):
        """BUYER signatures → buyer participant."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Buyer Signature",
                "required": True,
                "assigned_role": "BUYER",
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "John Buyer" in result
        assert len(result["John Buyer"]) == 1

    def test_seller_signature_goes_to_seller(self, multi_participant_list):
        """SELLER signatures → seller participant."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Seller Signature",
                "required": True,
                "assigned_role": "SELLER",
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "Jane Seller" in result
        assert len(result["Jane Seller"]) == 1

    def test_buyer_2_goes_to_second_buyer(self, multi_participant_list):
        """Buyer 2 signature → second buyer participant."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Buyer 2 Signature",
                "required": True,
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "Mary CoBuyer" in result

    def test_seller_2_goes_to_second_seller(self, multi_participant_list):
        """Seller 2 signature → second seller participant."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Seller #2 Initial",
                "required": True,
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "Tom CoSeller" in result

    def test_listing_agent_signature_routes_correctly(self, multi_participant_list):
        """LISTING_AGENT signatures → listing agent participant."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Listing Agent Signature",
                "required": True,
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "Alice Agent" in result

    def test_buying_agent_signature_routes_correctly(self, multi_participant_list):
        """BUYING_AGENT signatures → buying agent participant."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Buyer's Agent Sign Here",
                "required": True,
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "Bob Agent" in result

    def test_co_buyer_goes_to_second_buyer(self, multi_participant_list):
        """Co-Buyer signature → second buyer."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Co-Buyer Signature",
                "required": True,
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "Mary CoBuyer" in result

    def test_tenant_signature_routes_to_tenant(self):
        """TENANT signatures → tenant participant (lease context)."""
        participants = [
            {"full_name": "Tom Tenant", "role": "TENANT"},
            {"full_name": "Larry Landlord", "role": "LANDLORD"},
        ]
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Tenant Signature",
                "required": True,
            }
        ]
        result = map_signatures_to_participants(sigs, participants)
        assert "Tom Tenant" in result

    def test_landlord_signature_routes_to_landlord(self):
        """LANDLORD signatures → landlord participant."""
        participants = [
            {"full_name": "Tom Tenant", "role": "TENANT"},
            {"full_name": "Larry Landlord", "role": "LANDLORD"},
        ]
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Landlord Signature",
                "required": True,
            }
        ]
        result = map_signatures_to_participants(sigs, participants)
        assert "Larry Landlord" in result

    def test_multiple_signatures_same_participant(self, multi_participant_list):
        """Multiple signatures can be assigned to same participant."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Buyer Signature",
                "required": True,
            },
            {
                "page_number": 2,
                "x_position": 100,
                "y_position": 200,
                "field_type": "initial",
                "label": "Buyer Initial",
                "required": True,
            },
            {
                "page_number": 3,
                "x_position": 100,
                "y_position": 200,
                "field_type": "date",
                "label": "Buyer Date",
                "required": True,
            },
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "John Buyer" in result
        assert len(result["John Buyer"]) == 3

    def test_detects_role_from_label_when_not_assigned(self, multi_participant_list):
        """Detects role from label when assigned_role is missing."""
        sigs = [
            {
                "page_number": 1,
                "x_position": 100,
                "y_position": 200,
                "field_type": "signature",
                "label": "Purchaser Signature",  # Should detect as BUYER
                "required": True,
                # No assigned_role
            }
        ]
        result = map_signatures_to_participants(sigs, multi_participant_list)
        assert "John Buyer" in result


class TestValidateSignatureRouting:
    """Tests for signature routing validation."""

    def test_reports_unassigned_signatures(self):
        """Reports when signatures cannot be assigned."""
        routing = {
            "by_participant": {},
            "unassigned": [{"label": "Unknown Sig"}],
            "summary": {},
        }
        issues = validate_signature_routing(routing)
        assert any("could not be assigned" in i for i in issues)

    def test_no_issues_when_all_assigned(self):
        """No issues when all signatures are assigned."""
        routing = {
            "by_participant": {"John Buyer": [{"label": "Buyer Sig"}]},
            "unassigned": [],
            "summary": {"BUYER": 1},
        }
        issues = validate_signature_routing(routing)
        # May or may not have issues, but no "could not be assigned" issue
        assert not any("could not be assigned" in i for i in issues)


# ============================================================================
# Brokerage Defaults Tests
# ============================================================================

class TestBrokerageDefaults:
    """Tests for applying brokerage default settings."""

    @pytest.fixture
    def sample_settings(self):
        return {
            "default_commission_rate": "6%",
            "default_commission_split_buy_side": "3%",
            "default_commission_split_sell_side": "3%",
            "default_earnest_money_held_by": "Title Company",
        }

    def test_apply_defaults_to_empty_financials(self, sample_settings):
        """Defaults are applied when financials are missing."""
        financials = None
        result = apply_brokerage_defaults(financials, sample_settings)
        assert result is not None
        assert result.get("sale_commission_rate") == "6%"
        assert result.get("commission_split_buy_side_percent") == "3%"
        assert result.get("earnest_money_held_by") == "Title Company"

    def test_defaults_do_not_overwrite_existing(self, sample_settings):
        """Defaults do not overwrite existing values."""
        financials = cast(FinancialDetails, {
            "sale_commission_rate": "5%",
            "earnest_money_held_by": "Attorney",
        })
        result = apply_brokerage_defaults(financials, sample_settings)
        assert result is not None
        assert result.get("sale_commission_rate") == "5%"
        assert result.get("earnest_money_held_by") == "Attorney"
        # Missing fields should still be populated
        assert result.get("commission_split_buy_side_percent") == "3%"

    def test_defaults_overwrite_empty_strings(self, sample_settings):
        """Defaults overwrite empty strings."""
        financials = cast(FinancialDetails, {
            "sale_commission_rate": "",
        })
        result = apply_brokerage_defaults(financials, sample_settings)
        assert result is not None
        assert result.get("sale_commission_rate") == "6%"

    def test_no_settings_makes_no_changes(self):
        """No changes if settings are missing."""
        financials = cast(FinancialDetails, {})
        result = apply_brokerage_defaults(financials, None)
        assert result == {}

    def test_field_mapper_node_applies_defaults(self, sample_deal_state, sample_settings):
        """Field mapper node applies defaults from state."""
        # Setup state with settings and minimal financials
        sample_deal_state["brokerage_settings"] = sample_settings
        sample_deal_state["financial_details"] = {}
        
        result = field_mapper_node(sample_deal_state)
        
        # Check persisted state (in return value)
        financials = result["financial_details"]
        assert financials["sale_commission_rate"] == "6%"
        
        # Check Loop Details payload
        details = result["loop_details_payload"]
        assert "Financials" in details
        # Check mapping to API field
        assert details["Financials"]["Sale Commission Rate"] == "6%"

class TestDetailedValidation:
    """Tests for Detailed Validation Logic (itemized error objects)."""

    @pytest.fixture
    def sample_settings(self):
        return {
            "default_commission_rate": "6%",
            "default_commission_split_buy_side": "3%",
            "default_commission_split_sell_side": "3%",
            "default_earnest_money_held_by": "Title Company",
        }

    def test_validate_detailed_structure(self):
        """Test that validate_detailed_for_loop_it returns correct structure."""
        # Scenario: Missing Address fields and Participants
        state = {
            "property_details": {"street_number": "123"},
            "participants": []
        }
        
        errors = validate_detailed_for_loop_it(cast(DealState, state))
        
        # Expect errors for street_name, city, state, zip
        fields = [e["field"] for e in errors]
        assert "streetName" in fields
        assert "city" in fields
        
        # Check structure of one error
        err = errors[0]
        assert "field" in err
        assert "message" in err
        assert "severity" in err
        assert "expected_format" in err

    def test_validate_reports_expected_format(self):
        """Test that errors include format guidance."""
        # Zip code missing
        state = {
            "property_details": {
                "street_number": "123", "street_name": "Main", 
                "city": "Anytown", "state": "CA", 
                # "zip_code" Missing
            },
            "participants": [{"role": "buyer", "full_name": "John"}]
        }
        
        errors = validate_detailed_for_loop_it(cast(DealState, state))
        
        zip_err = next((e for e in errors if e["field"] == "zipCode"), None)
        assert zip_err is not None
        assert "expected_format" in zip_err
        # Verify the hint text
        expected = zip_err["expected_format"]
        assert expected is not None
        assert "5-digit" in expected

    def test_mapper_node_populates_validation_errors(self, sample_deal_state, sample_settings):
        """Test that the node populates state['validation_errors']."""
        # Setup state with minimal invalid data
        sample_deal_state["brokerage_settings"] = sample_settings
        sample_deal_state["property_details"] = {"street_number": "123"} # Missing rest
        sample_deal_state["participants"] = []
        
        result = field_mapper_node(sample_deal_state)
        
        assert "validation_errors" in result
        errors = result["validation_errors"]
        assert len(errors) > 0
        # The severity in mapper.py is 'Error' or 'Warning'
        assert errors[0]["severity"] in ["Error", "Warning"]


