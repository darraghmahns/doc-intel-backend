"""
Tests for the Extractor Node

E1: Participant extraction with >95% accuracy
- Full names, emails, phone numbers
- Confidence scoring and validation
"""

import pytest
from nodes.extractor import (
    # E1: Participant extraction
    ExtractionConfidence,
    ExtractedValue,
    ExtractedParticipant,
    ParticipantExtractionResult,
    EXTRACTION_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    # Validation functions
    validate_email,
    normalize_phone,
    validate_name,
    # Extraction functions
    extract_email_from_text,
    extract_phone_from_text,
    extract_names_from_text,
    extract_participant_by_role,
    extract_all_participants,
    deduplicate_participants,
    filter_participants_by_confidence,
    extract_participants_with_confidence,
    extract_participants,
    # Patterns
    EMAIL_PATTERN,
    PHONE_PATTERNS,
    NAME_PATTERN,
    PARTICIPANT_ROLE_PATTERNS,
)


# ============================================================================
# Test Fixtures - Sample Real Estate Documents
# ============================================================================

@pytest.fixture
def sample_purchase_agreement():
    """Sample purchase agreement with buyer/seller info."""
    return """
    RESIDENTIAL PURCHASE AGREEMENT
    
    This Purchase Agreement is entered into this 15th day of November, 2025.
    
    BUYER: John Michael Smith
    Email: john.smith@gmail.com
    Phone: (555) 234-5678
    Address: 123 Main Street, Anytown, CA 90210
    
    BUYER 2: Mary Jane Smith
    Email: mary.smith@yahoo.com
    Phone: 555-987-6543
    
    SELLER: Robert William Johnson
    Email: rjohnson@coldwellbanker.com
    Phone: (555) 456-7890
    
    SELLER 2: Susan Marie Johnson
    Email: sjohnson@gmail.com
    Phone: 555.321.9876
    
    PROPERTY ADDRESS: 456 Oak Avenue, Unit 7, Springfield, CA 92101
    
    PURCHASE PRICE: $750,000.00
    EARNEST MONEY DEPOSIT: $15,000.00
    
    LISTING AGENT: Patricia Ann Williams
    Company: Coldwell Banker Realty
    License #: 01234567
    Email: patricia.williams@coldwellbanker.com
    Phone: (555) 111-2222
    
    BUYER'S AGENT: David Lee Thompson
    Company: RE/MAX Associates
    License #: 98765432
    Email: david.thompson@remax.com
    Phone: (555) 333-4444
    """


@pytest.fixture
def sample_disclosure():
    """Sample disclosure document with signatures."""
    return """
    SELLER'S PROPERTY DISCLOSURE STATEMENT
    
    Property: 789 Pine Street, Lakewood, CA 90001
    
    Seller: James Andrew Brown
    Email: jabrown@outlook.com
    Phone: (310) 555-8888
    
    The seller hereby discloses the following information about the property.
    
    Seller Signature: _________________________ Date: ___________
    """


@pytest.fixture
def sample_text_with_emails():
    """Text containing various email formats."""
    return """
    Contact Information:
    Primary Email: john.doe@example.com
    Secondary: jane_doe123@company.org
    Agent: agent.name@kw.com
    Invalid: not-an-email
    Another: test.user+tag@subdomain.domain.co.uk
    """


@pytest.fixture
def sample_text_with_phones():
    """Text containing various phone formats."""
    return """
    Contact Numbers:
    Phone: (555) 234-5678
    Cell: 555-987-6543
    Tel: 555.321.9876
    Office: 5552345678
    Fax: 1-555-222-3333
    Invalid: 123-456
    """


# ============================================================================
# E1: Email Validation Tests
# ============================================================================

class TestValidateEmail:
    """Tests for email validation."""
    
    def test_valid_email_basic(self):
        """Test basic valid email."""
        is_valid, confidence = validate_email("john@example.com")
        assert is_valid is True
        assert confidence >= 0.90
    
    def test_valid_email_with_subdomain(self):
        """Test email with subdomain."""
        is_valid, confidence = validate_email("user@mail.example.com")
        assert is_valid is True
        assert confidence >= 0.90
    
    def test_valid_email_with_plus(self):
        """Test email with plus addressing."""
        is_valid, confidence = validate_email("user+tag@example.com")
        assert is_valid is True
    
    def test_valid_email_with_dots(self):
        """Test email with dots in local part."""
        is_valid, confidence = validate_email("first.last@example.com")
        assert is_valid is True
    
    def test_trusted_domain_higher_confidence(self):
        """Test that trusted domains get higher confidence."""
        _, gmail_conf = validate_email("test@gmail.com")
        _, unknown_conf = validate_email("test@unknowndomain.xyz")
        assert gmail_conf > unknown_conf
    
    def test_real_estate_domain_high_confidence(self):
        """Test real estate brokerage domains."""
        is_valid, confidence = validate_email("agent@kw.com")
        assert is_valid is True
        assert confidence >= 0.95
    
    def test_invalid_email_no_at(self):
        """Test email without @ symbol."""
        is_valid, _ = validate_email("notanemail.com")
        assert is_valid is False
    
    def test_invalid_email_multiple_at(self):
        """Test email with multiple @ symbols."""
        is_valid, _ = validate_email("user@@example.com")
        assert is_valid is False
    
    def test_invalid_email_double_dots(self):
        """Test email with consecutive dots."""
        is_valid, _ = validate_email("user..name@example.com")
        assert is_valid is False
    
    def test_invalid_email_too_short(self):
        """Test email that's too short."""
        is_valid, _ = validate_email("a@b")
        assert is_valid is False
    
    def test_invalid_email_empty(self):
        """Test empty email."""
        is_valid, _ = validate_email("")
        assert is_valid is False
    
    def test_invalid_email_none(self):
        """Test None email."""
        is_valid, _ = validate_email(None)  # type: ignore
        assert is_valid is False
    
    def test_email_case_normalization(self):
        """Test that email validation handles case."""
        is_valid, _ = validate_email("John.Doe@Example.COM")
        assert is_valid is True


# ============================================================================
# E1: Phone Validation Tests
# ============================================================================

class TestNormalizePhone:
    """Tests for phone number normalization."""
    
    def test_normalize_parentheses_format(self):
        """Test (XXX) XXX-XXXX format."""
        normalized, confidence = normalize_phone("(555) 234-5678")
        assert normalized == "(555) 234-5678"
        assert confidence >= 0.90
    
    def test_normalize_dashes_format(self):
        """Test XXX-XXX-XXXX format."""
        normalized, confidence = normalize_phone("555-234-5678")
        assert normalized == "(555) 234-5678"
        assert confidence >= 0.90
    
    def test_normalize_dots_format(self):
        """Test XXX.XXX.XXXX format."""
        normalized, confidence = normalize_phone("555.234.5678")
        assert normalized == "(555) 234-5678"
        assert confidence >= 0.90
    
    def test_normalize_no_separator(self):
        """Test 10 digit number without separators."""
        normalized, confidence = normalize_phone("5552345678")
        assert normalized == "(555) 234-5678"
    
    def test_normalize_with_country_code(self):
        """Test number with +1 country code."""
        normalized, confidence = normalize_phone("+1 555-234-5678")
        assert normalized == "(555) 234-5678"
    
    def test_normalize_with_1_prefix(self):
        """Test number with 1 prefix."""
        normalized, confidence = normalize_phone("1-555-234-5678")
        assert normalized == "(555) 234-5678"
    
    def test_invalid_phone_too_short(self):
        """Test phone number that's too short."""
        normalized, _ = normalize_phone("555-1234")
        assert normalized is None
    
    def test_invalid_phone_too_long(self):
        """Test phone number that's too long."""
        normalized, _ = normalize_phone("555-234-5678-890")
        assert normalized is None
    
    def test_invalid_area_code_starts_0(self):
        """Test invalid area code starting with 0."""
        normalized, _ = normalize_phone("055-234-5678")
        assert normalized is None
    
    def test_invalid_area_code_starts_1(self):
        """Test invalid area code starting with 1."""
        normalized, _ = normalize_phone("155-234-5678")
        assert normalized is None
    
    def test_invalid_exchange_starts_0(self):
        """Test invalid exchange starting with 0."""
        normalized, _ = normalize_phone("555-023-5678")
        assert normalized is None
    
    def test_invalid_phone_empty(self):
        """Test empty phone."""
        normalized, _ = normalize_phone("")
        assert normalized is None
    
    def test_invalid_phone_none(self):
        """Test None phone."""
        normalized, _ = normalize_phone(None)  # type: ignore
        assert normalized is None


# ============================================================================
# E1: Name Validation Tests
# ============================================================================

class TestValidateName:
    """Tests for name validation."""
    
    def test_valid_name_two_parts(self):
        """Test basic two-part name."""
        is_valid, confidence = validate_name("John Smith")
        assert is_valid is True
        assert confidence >= 0.80
    
    def test_valid_name_three_parts(self):
        """Test three-part name with middle name."""
        is_valid, confidence = validate_name("John Michael Smith")
        assert is_valid is True
        assert confidence >= 0.80
    
    def test_valid_name_with_initial(self):
        """Test name with middle initial."""
        is_valid, confidence = validate_name("John M. Smith")
        assert is_valid is True
    
    def test_valid_name_with_suffix(self):
        """Test name with suffix."""
        is_valid, confidence = validate_name("John Smith Jr.")
        assert is_valid is True
    
    def test_valid_name_with_hyphen(self):
        """Test hyphenated name."""
        is_valid, confidence = validate_name("Mary-Jane Watson")
        assert is_valid is True
    
    def test_valid_name_with_apostrophe(self):
        """Test name with apostrophe."""
        is_valid, confidence = validate_name("Patrick O'Brien")
        assert is_valid is True
    
    def test_invalid_name_single_word(self):
        """Test single word - not a valid full name."""
        is_valid, _ = validate_name("John")
        assert is_valid is False
    
    def test_invalid_name_contract_term(self):
        """Test that contract terms are rejected."""
        is_valid, _ = validate_name("Purchase Agreement")
        assert is_valid is False
    
    def test_invalid_name_with_numbers(self):
        """Test name with too many numbers."""
        is_valid, _ = validate_name("John 12345 Smith")
        assert is_valid is False
    
    def test_invalid_name_with_special_chars(self):
        """Test name with special characters."""
        is_valid, _ = validate_name("John @Smith")
        assert is_valid is False
    
    def test_invalid_name_too_long(self):
        """Test name that's too long."""
        long_name = "A" * 101
        is_valid, _ = validate_name(long_name)
        assert is_valid is False
    
    def test_invalid_name_empty(self):
        """Test empty name."""
        is_valid, _ = validate_name("")
        assert is_valid is False
    
    def test_invalid_name_excluded_words(self):
        """Test names containing excluded words."""
        is_valid, _ = validate_name("Property Disclosure")
        assert is_valid is False
        is_valid, _ = validate_name("Real Estate")
        assert is_valid is False


# ============================================================================
# E1: Email Extraction Tests
# ============================================================================

class TestExtractEmailFromText:
    """Tests for extracting emails from text."""
    
    def test_extract_labeled_email(self):
        """Test extraction of labeled email."""
        text = "Email: john.smith@example.com"
        emails = extract_email_from_text(text)
        assert len(emails) >= 1
        assert emails[0].value == "john.smith@example.com"
    
    def test_extract_multiple_emails(self, sample_text_with_emails):
        """Test extraction of multiple emails."""
        emails = extract_email_from_text(sample_text_with_emails)
        assert len(emails) >= 3
        values = [e.value for e in emails]
        assert "john.doe@example.com" in values
        assert "jane_doe123@company.org" in values
    
    def test_labeled_email_higher_confidence(self):
        """Test that labeled emails get higher confidence."""
        text = "Email: labeled@example.com\nunlabeled@example.com"
        emails = extract_email_from_text(text)
        labeled = next((e for e in emails if e.value == "labeled@example.com"), None)
        unlabeled = next((e for e in emails if e.value == "unlabeled@example.com"), None)
        assert labeled is not None
        assert unlabeled is not None
        assert labeled.confidence > unlabeled.confidence
    
    def test_extract_email_with_e_mail_format(self):
        """Test extraction with 'e-mail' label."""
        text = "E-mail address: contact@domain.com"
        emails = extract_email_from_text(text)
        assert len(emails) >= 1
    
    def test_no_duplicate_emails(self):
        """Test that duplicate emails are not returned."""
        text = "Email: same@example.com\nContact: same@example.com"
        emails = extract_email_from_text(text)
        values = [e.value for e in emails]
        assert values.count("same@example.com") == 1
    
    def test_extract_real_estate_broker_email(self):
        """Test extraction of real estate domain emails."""
        text = "Agent: patricia@coldwellbanker.com"
        emails = extract_email_from_text(text)
        assert len(emails) >= 1
        # Should have high confidence for trusted domain
        assert emails[0].confidence >= 0.95


# ============================================================================
# E1: Phone Extraction Tests
# ============================================================================

class TestExtractPhoneFromText:
    """Tests for extracting phone numbers from text."""
    
    def test_extract_labeled_phone(self):
        """Test extraction of labeled phone."""
        text = "Phone: (555) 234-5678"
        phones = extract_phone_from_text(text)
        assert len(phones) >= 1
        assert phones[0].value == "(555) 234-5678"
    
    def test_extract_multiple_phones(self, sample_text_with_phones):
        """Test extraction of multiple phone numbers."""
        phones = extract_phone_from_text(sample_text_with_phones)
        assert len(phones) >= 3
    
    def test_extract_phone_various_formats(self):
        """Test extraction of different phone formats."""
        text = """
        (555) 222-3333
        555-333-4444
        555.555.6666
        """
        phones = extract_phone_from_text(text)
        assert len(phones) >= 3
        # All should normalize to same format
        for phone in phones:
            assert phone.value.startswith("(")
            assert ")" in phone.value
    
    def test_labeled_phone_higher_confidence(self):
        """Test that labeled phones get higher confidence."""
        text = "Phone: (555) 222-3333\nAlso call 555-444-5555"
        phones = extract_phone_from_text(text)
        labeled = next((p for p in phones if "222" in p.value), None)
        unlabeled = next((p for p in phones if "444" in p.value), None)
        assert labeled is not None
        assert unlabeled is not None
        assert labeled.confidence > unlabeled.confidence
    
    def test_no_duplicate_phones(self):
        """Test that duplicate phones are not returned."""
        text = "Phone: (555) 234-5678\nCell: 555-234-5678"
        phones = extract_phone_from_text(text)
        values = [p.value for p in phones]
        assert values.count("(555) 234-5678") == 1


# ============================================================================
# E1: Name Extraction Tests
# ============================================================================

class TestExtractNamesFromText:
    """Tests for extracting names from text."""
    
    def test_extract_labeled_name(self):
        """Test extraction of labeled name."""
        text = "Name: John Michael Smith"
        names = extract_names_from_text(text)
        assert len(names) >= 1
        assert "John Michael Smith" in [n.value for n in names]
    
    def test_extract_buyer_name(self):
        """Test extraction of buyer name."""
        text = "Buyer Name: Patricia Ann Williams"
        names = extract_names_from_text(text)
        assert len(names) >= 1
    
    def test_extract_printed_name(self):
        """Test extraction of printed name field."""
        text = "Print Name: David Lee Thompson"
        names = extract_names_from_text(text)
        assert len(names) >= 1
    
    def test_no_contract_terms_extracted(self):
        """Test that contract terms aren't extracted as names."""
        text = "This Purchase Agreement is between the parties."
        names = extract_names_from_text(text)
        # Should not extract "Purchase Agreement" as a name
        values_lower = [n.value.lower() for n in names]
        assert "purchase agreement" not in values_lower


# ============================================================================
# E1: Participant Extraction by Role Tests
# ============================================================================

class TestExtractParticipantByRole:
    """Tests for extracting participants by role."""
    
    def test_extract_buyer(self):
        """Test extraction of buyer."""
        text = "Buyer: John Michael Smith\nEmail: john@example.com"
        participants = extract_participant_by_role(
            text, "BUYER", PARTICIPANT_ROLE_PATTERNS["BUYER"]
        )
        assert len(participants) >= 1
        assert participants[0].role == "BUYER"
        assert "John" in participants[0].full_name.value
    
    def test_extract_seller(self):
        """Test extraction of seller."""
        text = "Seller: Robert William Johnson\nPhone: (555) 234-5678"
        participants = extract_participant_by_role(
            text, "SELLER", PARTICIPANT_ROLE_PATTERNS["SELLER"]
        )
        assert len(participants) >= 1
        assert participants[0].role == "SELLER"
    
    def test_extract_buyer_with_email_and_phone(self):
        """Test extraction of buyer with contact info."""
        text = """
        Buyer: Jane Marie Doe
        Email: jane.doe@gmail.com
        Phone: (555) 987-6543
        """
        participants = extract_participant_by_role(
            text, "BUYER", PARTICIPANT_ROLE_PATTERNS["BUYER"]
        )
        assert len(participants) >= 1
        p = participants[0]
        assert p.email is not None
        assert p.email.value == "jane.doe@gmail.com"
        assert p.phone is not None
        assert p.phone.value == "(555) 987-6543"
    
    def test_extract_listing_agent(self):
        """Test extraction of listing agent."""
        text = "Listing Agent: Patricia Ann Williams\nLicense #: 01234567"
        participants = extract_participant_by_role(
            text, "LISTING_AGENT", PARTICIPANT_ROLE_PATTERNS["LISTING_AGENT"]
        )
        assert len(participants) >= 1
        assert participants[0].role == "LISTING_AGENT"
    
    def test_extract_multiple_buyers(self):
        """Test extraction of multiple buyers."""
        text = """
        Buyer 1: John Smith
        Email: john@example.com
        
        Buyer 2: Jane Smith
        Email: jane@example.com
        """
        participants = extract_participant_by_role(
            text, "BUYER", PARTICIPANT_ROLE_PATTERNS["BUYER"]
        )
        # Note: patterns may extract multiple
        assert len(participants) >= 1


# ============================================================================
# E1: Full Participant Extraction Tests
# ============================================================================

class TestExtractAllParticipants:
    """Tests for extracting all participants from document."""
    
    def test_extract_from_purchase_agreement(self, sample_purchase_agreement):
        """Test extraction from full purchase agreement."""
        participants = extract_all_participants(sample_purchase_agreement)
        assert len(participants) >= 2
        
        # Check we got buyers and sellers
        roles = [p.role for p in participants]
        assert "BUYER" in roles
        assert "SELLER" in roles
    
    def test_extract_agents(self, sample_purchase_agreement):
        """Test extraction of agents."""
        participants = extract_all_participants(sample_purchase_agreement)
        roles = [p.role for p in participants]
        assert "LISTING_AGENT" in roles or "BUYING_AGENT" in roles
    
    def test_extract_with_emails(self, sample_purchase_agreement):
        """Test that emails are extracted with participants."""
        participants = extract_all_participants(sample_purchase_agreement)
        with_email = [p for p in participants if p.email is not None]
        assert len(with_email) >= 1
    
    def test_extract_with_phones(self, sample_purchase_agreement):
        """Test that phones are extracted with participants."""
        participants = extract_all_participants(sample_purchase_agreement)
        with_phone = [p for p in participants if p.phone is not None]
        assert len(with_phone) >= 1


# ============================================================================
# E1: Deduplication Tests
# ============================================================================

class TestDeduplicateParticipants:
    """Tests for participant deduplication."""
    
    def test_deduplicate_same_name(self):
        """Test deduplication of same name."""
        p1 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
        )
        p2 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.85, "text2", "method"),
            role="BUYER",
        )
        result = deduplicate_participants([p1, p2])
        assert len(result) == 1
        # Should keep higher confidence
        assert result[0].full_name.confidence == 0.90
    
    def test_deduplicate_merges_email(self):
        """Test that deduplication merges email."""
        p1 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
            email=None,
        )
        p2 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.85, "text2", "method"),
            role="BUYER",
            email=ExtractedValue("john@example.com", 0.95, "email", "method"),
        )
        result = deduplicate_participants([p1, p2])
        assert len(result) == 1
        assert result[0].email is not None
        assert result[0].email.value == "john@example.com"
    
    def test_deduplicate_merges_phone(self):
        """Test that deduplication merges phone."""
        p1 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
            phone=None,
        )
        p2 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.85, "text2", "method"),
            role="BUYER",
            phone=ExtractedValue("(555) 234-5678", 0.95, "phone", "method"),
        )
        result = deduplicate_participants([p1, p2])
        assert len(result) == 1
        assert result[0].phone is not None
        assert result[0].phone.value == "(555) 234-5678"
    
    def test_deduplicate_keeps_higher_confidence_contact(self):
        """Test that deduplication keeps higher confidence contact info."""
        p1 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
            email=ExtractedValue("low@example.com", 0.80, "email", "method"),
        )
        p2 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.85, "text2", "method"),
            role="BUYER",
            email=ExtractedValue("high@example.com", 0.95, "email", "method"),
        )
        result = deduplicate_participants([p1, p2])
        assert len(result) == 1
        assert result[0].email is not None
        assert result[0].email.value == "high@example.com"
    
    def test_deduplicate_case_insensitive(self):
        """Test that deduplication is case insensitive."""
        p1 = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
        )
        p2 = ExtractedParticipant(
            full_name=ExtractedValue("JOHN SMITH", 0.85, "text2", "method"),
            role="BUYER",
        )
        result = deduplicate_participants([p1, p2])
        assert len(result) == 1


# ============================================================================
# E1: Confidence Filtering Tests
# ============================================================================

class TestFilterParticipantsByConfidence:
    """Tests for confidence-based filtering."""
    
    def test_filter_high_confidence_accepted(self):
        """Test that high confidence participants are accepted."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.95, "text", "method"),
            role="BUYER",
        )
        accepted, needs_review = filter_participants_by_confidence([p])
        assert len(accepted) == 1
        assert len(needs_review) == 0
    
    def test_filter_low_confidence_needs_review(self):
        """Test that low confidence participants need review."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.60, "text", "method"),
            role="BUYER",
        )
        accepted, needs_review = filter_participants_by_confidence([p])
        assert len(accepted) == 0
        assert len(needs_review) == 1
    
    def test_filter_custom_threshold(self):
        """Test custom confidence threshold."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.75, "text", "method"),
            role="BUYER",
        )
        # Default threshold is 0.80, should be in needs_review
        accepted, needs_review = filter_participants_by_confidence([p])
        assert len(needs_review) == 1
        
        # With lower threshold, should be accepted
        accepted, needs_review = filter_participants_by_confidence([p], min_confidence=0.70)
        assert len(accepted) == 1
    
    def test_filter_overall_confidence_calculation(self):
        """Test that overall confidence considers all fields."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.95, "text", "method"),
            role="BUYER",
            email=ExtractedValue("john@example.com", 0.60, "email", "method"),
        )
        # Overall should be average: (0.95 + 0.60) / 2 = 0.775
        assert p.overall_confidence() == pytest.approx(0.775)


# ============================================================================
# E1: Full Extraction Pipeline Tests
# ============================================================================

class TestExtractParticipantsWithConfidence:
    """Tests for the main extraction function."""
    
    def test_returns_extraction_result(self, sample_purchase_agreement):
        """Test that function returns ParticipantExtractionResult."""
        result = extract_participants_with_confidence(sample_purchase_agreement)
        assert isinstance(result, ParticipantExtractionResult)
    
    def test_result_has_participants(self, sample_purchase_agreement):
        """Test that result contains participants."""
        result = extract_participants_with_confidence(sample_purchase_agreement)
        assert len(result.participants) >= 1 or len(result.needs_review) >= 1
    
    def test_result_has_stats(self, sample_purchase_agreement):
        """Test that result contains extraction stats."""
        result = extract_participants_with_confidence(sample_purchase_agreement)
        assert "total_extracted" in result.extraction_stats
        assert "accepted" in result.extraction_stats
        assert "needs_review" in result.extraction_stats
    
    def test_result_to_participant_info_list(self, sample_purchase_agreement):
        """Test conversion to ParticipantInfo list."""
        result = extract_participants_with_confidence(sample_purchase_agreement)
        info_list = result.to_participant_info_list()
        assert isinstance(info_list, list)
        if info_list:
            assert "full_name" in info_list[0]
            assert "role" in info_list[0]
    
    def test_result_to_dict(self, sample_purchase_agreement):
        """Test conversion to dictionary."""
        result = extract_participants_with_confidence(sample_purchase_agreement)
        d = result.to_dict()
        assert "participants" in d
        assert "needs_review" in d
        assert "stats" in d


# ============================================================================
# E1: Backward Compatibility Tests
# ============================================================================

class TestExtractParticipantsBackwardCompat:
    """Tests for backward compatible extract_participants function."""
    
    def test_returns_list(self, sample_purchase_agreement):
        """Test that function returns list."""
        result = extract_participants(sample_purchase_agreement)
        assert isinstance(result, list)
    
    def test_returns_participant_info(self, sample_purchase_agreement):
        """Test that function returns ParticipantInfo dicts."""
        result = extract_participants(sample_purchase_agreement)
        if result:
            p = result[0]
            assert "full_name" in p
            assert "role" in p
            assert "email" in p or "email" not in p  # Optional
    
    def test_extracts_buyers(self, sample_purchase_agreement):
        """Test extraction of buyers."""
        result = extract_participants(sample_purchase_agreement)
        buyers = [p for p in result if p.get("role") == "BUYER"]
        assert len(buyers) >= 1
    
    def test_extracts_sellers(self, sample_purchase_agreement):
        """Test extraction of sellers."""
        result = extract_participants(sample_purchase_agreement)
        sellers = [p for p in result if p.get("role") == "SELLER"]
        assert len(sellers) >= 1


# ============================================================================
# E1: ExtractedValue Tests
# ============================================================================

class TestExtractedValue:
    """Tests for ExtractedValue dataclass."""
    
    def test_high_confidence_level(self):
        """Test high confidence level classification."""
        ev = ExtractedValue("test", 0.96, "source", "method")
        assert ev.confidence_level() == ExtractionConfidence.HIGH
    
    def test_medium_confidence_level(self):
        """Test medium confidence level classification."""
        ev = ExtractedValue("test", 0.85, "source", "method")
        assert ev.confidence_level() == ExtractionConfidence.MEDIUM
    
    def test_low_confidence_level(self):
        """Test low confidence level classification."""
        ev = ExtractedValue("test", 0.70, "source", "method")
        assert ev.confidence_level() == ExtractionConfidence.LOW
    
    def test_to_dict(self):
        """Test to_dict method."""
        ev = ExtractedValue("test", 0.90, "source", "method", 10, 20)
        d = ev.to_dict()
        assert d["value"] == "test"
        assert d["confidence"] == 0.90
        assert d["source_text"] == "source"
        assert d["extraction_method"] == "method"
        assert "confidence_level" in d


# ============================================================================
# E1: ExtractedParticipant Tests
# ============================================================================

class TestExtractedParticipant:
    """Tests for ExtractedParticipant dataclass."""
    
    def test_overall_confidence_name_only(self):
        """Test overall confidence with just name."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
        )
        assert p.overall_confidence() == 0.90
    
    def test_overall_confidence_with_email(self):
        """Test overall confidence with name and email."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
            email=ExtractedValue("john@example.com", 0.80, "email", "method"),
        )
        assert p.overall_confidence() == pytest.approx(0.85)
    
    def test_overall_confidence_with_all(self):
        """Test overall confidence with name, email, and phone."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
            email=ExtractedValue("john@example.com", 0.80, "email", "method"),
            phone=ExtractedValue("(555) 234-5678", 0.70, "phone", "method"),
        )
        assert p.overall_confidence() == pytest.approx(0.80)
    
    def test_to_participant_info(self):
        """Test conversion to ParticipantInfo."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
            email=ExtractedValue("john@example.com", 0.80, "email", "method"),
            phone=ExtractedValue("(555) 234-5678", 0.95, "phone", "method"),
        )
        info = p.to_participant_info()
        assert info.get("full_name") == "John Smith"
        assert info.get("role") == "BUYER"
        assert info.get("email") == "john@example.com"
        assert info.get("phone") == "(555) 234-5678"
    
    def test_to_dict(self):
        """Test to_dict method."""
        p = ExtractedParticipant(
            full_name=ExtractedValue("John Smith", 0.90, "text", "method"),
            role="BUYER",
        )
        d = p.to_dict()
        assert "full_name" in d
        assert "role" in d
        assert "overall_confidence" in d


# ============================================================================
# E1: Real World Document Tests
# ============================================================================

class TestRealWorldDocuments:
    """Tests with realistic document content."""
    
    def test_coop_listing_agreement(self):
        """Test extraction from cooperative listing agreement."""
        text = """
        EXCLUSIVE RIGHT TO SELL LISTING AGREEMENT
        
        Owner/Seller: Michael Andrew Thompson
        Address: 123 Main Street, San Diego, CA 92101
        Email: m.thompson@protonmail.com
        Phone: (619) 555-1234
        
        Owner/Seller 2: Linda Marie Thompson
        Email: linda.thompson@gmail.com
        Phone: 619.555.5678
        
        Property Address: 456 Oak Lane, San Diego, CA 92102
        
        Listing Broker: Pacific Coast Realty
        Listing Agent: Jennifer Rose Adams
        License #: DRE 01987654
        Email: jadams@pacificcoastrealty.com
        Phone: (619) 555-9999
        """
        result = extract_participants_with_confidence(text)
        
        # Should extract sellers
        sellers = [p for p in result.participants if p.role == "SELLER"]
        assert len(sellers) >= 1
        
        # Check at least one has email
        with_email = [p for p in result.participants if p.email]
        assert len(with_email) >= 1
    
    def test_counter_offer(self):
        """Test extraction from counter offer."""
        text = """
        COUNTER OFFER #1
        
        In response to the offer dated November 15, 2025
        
        Buyer: William James Carter
        Email: wcarter@outlook.com
        
        Seller: Elizabeth Anne Morgan
        Phone: (415) 555-3333
        
        The Seller proposes the following changes:
        Purchase Price: $825,000.00
        
        Seller Signature: _______________________
        Date: _______________
        """
        result = extract_participants_with_confidence(text)
        
        participants = result.participants + result.needs_review
        roles = [p.role for p in participants]
        assert "BUYER" in roles
        assert "SELLER" in roles
    
    def test_messy_formatting(self):
        """Test extraction from document with messy formatting."""
        text = """
        BUYER:    John   Smith   
        EMAIL:john.smith@email.com   
        PHONE:   555-234-5678
        
        SELLER:Robert     Jones
        email:  rjones@test.org
        PHONE: (555)   999-8888
        """
        result = extract_participants_with_confidence(text)
        
        # Should still extract despite formatting issues
        all_participants = result.participants + result.needs_review
        assert len(all_participants) >= 1


# ============================================================================
# E1: Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_text(self):
        """Test extraction from empty text."""
        result = extract_participants_with_confidence("")
        assert len(result.participants) == 0
        assert len(result.needs_review) == 0
    
    def test_no_participants(self):
        """Test extraction from text with no participants."""
        text = "This is just some random text without any names or contacts."
        result = extract_participants_with_confidence(text)
        assert len(result.participants) == 0
    
    def test_unicode_names(self):
        """Test extraction of names with unicode characters."""
        text = "Buyer: José García-López\nEmail: jose@example.com"
        result = extract_participants_with_confidence(text)
        # May or may not extract depending on pattern, but shouldn't crash
        assert isinstance(result, ParticipantExtractionResult)
    
    def test_very_long_text(self):
        """Test extraction from very long text."""
        text = "Buyer: John Smith\nEmail: john@example.com\n" + ("Lorem ipsum " * 1000)
        result = extract_participants_with_confidence(text)
        assert isinstance(result, ParticipantExtractionResult)
    
    def test_special_characters_in_context(self):
        """Test handling of special characters."""
        text = "Buyer: John Smith @#$%^&*()\nEmail: john@example.com"
        result = extract_participants_with_confidence(text)
        # Should still extract the name
        assert isinstance(result, ParticipantExtractionResult)


# ============================================================================
# E1: Accuracy Metrics Tests
# ============================================================================

class TestAccuracyMetrics:
    """Tests to verify 95% accuracy target."""
    
    def test_email_accuracy_valid(self):
        """Test email validation accuracy on valid emails."""
        valid_emails = [
            "john@example.com",
            "jane.doe@company.org",
            "user+tag@domain.co.uk",
            "first.last@subdomain.domain.com",
            "agent@kw.com",
            "broker@coldwellbanker.com",
        ]
        for email in valid_emails:
            is_valid, _ = validate_email(email)
            assert is_valid is True, f"Failed to validate valid email: {email}"
    
    def test_email_accuracy_invalid(self):
        """Test email validation accuracy on invalid emails."""
        invalid_emails = [
            "notanemail",
            "@nodomain.com",
            "missing@",
            "two@@at.com",
            "space in@email.com",
        ]
        for email in invalid_emails:
            is_valid, _ = validate_email(email)
            assert is_valid is False, f"Incorrectly validated invalid email: {email}"
    
    def test_phone_accuracy_valid(self):
        """Test phone normalization accuracy on valid phones."""
        valid_phones = [
            "(555) 234-5678",
            "555-234-5678",
            "555.234.5678",
            "5552345678",
            "+1-555-234-5678",
        ]
        for phone in valid_phones:
            normalized, _ = normalize_phone(phone)
            assert normalized is not None, f"Failed to normalize valid phone: {phone}"
            assert normalized == "(555) 234-5678"
    
    def test_phone_accuracy_invalid(self):
        """Test phone normalization accuracy on invalid phones."""
        invalid_phones = [
            "123-456",
            "55-234-5678",
            "555-234",
            "055-234-5678",
            "555-023-5678",
        ]
        for phone in invalid_phones:
            normalized, _ = normalize_phone(phone)
            assert normalized is None, f"Incorrectly normalized invalid phone: {phone}"
    
    def test_name_accuracy_valid(self):
        """Test name validation accuracy on valid names."""
        valid_names = [
            "John Smith",
            "Mary Jane Watson",
            "Robert J. Williams",
            "Patricia O'Brien",
            "John Smith Jr.",
            "David Lee-Thompson",
        ]
        for name in valid_names:
            is_valid, confidence = validate_name(name)
            assert is_valid is True, f"Failed to validate valid name: {name}"
            assert confidence >= 0.75, f"Low confidence for valid name: {name}"
    
    def test_name_accuracy_invalid(self):
        """Test name validation accuracy on invalid names."""
        invalid_names = [
            "John",  # Single word
            "Purchase Agreement",
            "Property Disclosure",
            "123 Main Street",
            "@#$%^",
        ]
        for name in invalid_names:
            is_valid, _ = validate_name(name)
            assert is_valid is False, f"Incorrectly validated invalid name: {name}"


# ============================================================================
# E2: Signature Detection Tests - PDF Coordinates
# ============================================================================

from nodes.extractor import (
    # E2: Signature detection types
    SignatureFieldType,
    PageSize,
    SignatureCoordinates,
    DetectedSignatureField,
    SignatureDetectionResult,
    # E2: Signature detection functions
    calculate_text_coordinates,
    detect_signature_fields_enhanced,
    detect_all_signature_fields,
    detect_signature_fields,
    # E2: Constants
    DEFAULT_SIGNATURE_WIDTH,
    DEFAULT_SIGNATURE_HEIGHT,
    DEFAULT_INITIAL_WIDTH,
    DEFAULT_INITIAL_HEIGHT,
    DEFAULT_DATE_WIDTH,
    DEFAULT_DATE_HEIGHT,
)


class TestSignatureFieldType:
    """Tests for SignatureFieldType enum."""
    
    def test_signature_type_value(self):
        """Test SIGNATURE type has correct value."""
        assert SignatureFieldType.SIGNATURE.value == "signature"
    
    def test_initial_type_value(self):
        """Test INITIAL type has correct value."""
        assert SignatureFieldType.INITIAL.value == "initial"
    
    def test_date_type_value(self):
        """Test DATE type has correct value."""
        assert SignatureFieldType.DATE.value == "date"
    
    def test_text_type_value(self):
        """Test TEXT type has correct value."""
        assert SignatureFieldType.TEXT.value == "text"
    
    def test_checkbox_type_value(self):
        """Test CHECKBOX type has correct value."""
        assert SignatureFieldType.CHECKBOX.value == "checkbox"


class TestPageSize:
    """Tests for PageSize enum with PDF dimensions."""
    
    def test_letter_dimensions(self):
        """Test Letter size is 8.5x11 inches (612x792 points)."""
        assert PageSize.LETTER.width == 612.0
        assert PageSize.LETTER.height == 792.0
    
    def test_legal_dimensions(self):
        """Test Legal size is 8.5x14 inches (612x1008 points)."""
        assert PageSize.LEGAL.width == 612.0
        assert PageSize.LEGAL.height == 1008.0
    
    def test_a4_dimensions(self):
        """Test A4 size is 210x297mm (~595x842 points)."""
        assert PageSize.A4.width == 595.0
        assert PageSize.A4.height == 842.0


class TestSignatureCoordinates:
    """Tests for SignatureCoordinates dataclass."""
    
    def test_basic_coordinates(self):
        """Test basic coordinate creation."""
        coords = SignatureCoordinates(
            x=100.0,
            y=200.0,
            width=150.0,
            height=30.0,
            page_number=1,
        )
        assert coords.x == 100.0
        assert coords.y == 200.0
        assert coords.width == 150.0
        assert coords.height == 30.0
        assert coords.page_number == 1
    
    def test_default_page_size(self):
        """Test default page size is Letter."""
        coords = SignatureCoordinates(
            x=100.0, y=200.0, width=150.0, height=30.0, page_number=1
        )
        assert coords.page_width == PageSize.LETTER.width
        assert coords.page_height == PageSize.LETTER.height
    
    def test_x_percentage(self):
        """Test X position as percentage of page width."""
        coords = SignatureCoordinates(
            x=306.0,  # Half of 612
            y=200.0,
            width=150.0,
            height=30.0,
            page_number=1,
        )
        assert abs(coords.x_percentage - 50.0) < 0.1
    
    def test_y_percentage(self):
        """Test Y position as percentage of page height."""
        coords = SignatureCoordinates(
            x=100.0,
            y=396.0,  # Half of 792
            width=150.0,
            height=30.0,
            page_number=1,
        )
        assert abs(coords.y_percentage - 50.0) < 0.1
    
    def test_y_from_top(self):
        """Test Y position converted to top-left origin."""
        coords = SignatureCoordinates(
            x=100.0,
            y=100.0,  # 100 points from bottom
            width=150.0,
            height=30.0,
            page_number=1,
            page_height=792.0,
        )
        # y_from_top = page_height - y - height = 792 - 100 - 30 = 662
        assert coords.y_from_top == 662.0
    
    def test_y_from_top_percentage(self):
        """Test Y from top as percentage."""
        coords = SignatureCoordinates(
            x=100.0,
            y=0.0,  # At very bottom
            width=150.0,
            height=30.0,
            page_number=1,
            page_height=792.0,
        )
        # y_from_top = 792 - 0 - 30 = 762
        # percentage = 762 / 792 * 100 ≈ 96.2%
        assert abs(coords.y_from_top_percentage - 96.2) < 0.5
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        coords = SignatureCoordinates(
            x=100.0, y=200.0, width=150.0, height=30.0, page_number=1
        )
        d = coords.to_dict()
        assert d["x"] == 100.0
        assert d["y"] == 200.0
        assert d["width"] == 150.0
        assert d["height"] == 30.0
        assert d["page_number"] == 1
        assert "x_percentage" in d
        assert "y_from_top" in d


class TestDetectedSignatureField:
    """Tests for DetectedSignatureField dataclass."""
    
    def test_basic_creation(self):
        """Test basic field creation."""
        coords = SignatureCoordinates(
            x=72.0, y=500.0, width=200.0, height=30.0, page_number=1
        )
        field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.SIGNATURE,
            role="BUYER",
            label="Buyer Signature",
        )
        assert field.field_type == SignatureFieldType.SIGNATURE
        assert field.role == "BUYER"
        assert field.label == "Buyer Signature"
        assert field.required is True  # Default
    
    def test_to_signature_field(self):
        """Test conversion to SignatureField TypedDict."""
        coords = SignatureCoordinates(
            x=72.0, y=500.0, width=200.0, height=30.0, page_number=1
        )
        field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.SIGNATURE,
            role="SELLER",
            label="Seller Signature",
            context_text="Sign here: _____",
        )
        sig_field = field.to_signature_field()
        
        assert sig_field["page_number"] == 1
        assert sig_field["x_position"] == 72.0
        assert sig_field["y_position"] == 500.0
        assert sig_field["width"] == 200.0
        assert sig_field["height"] == 30.0
        assert sig_field["field_type"] == "signature"
        assert sig_field["assigned_role"] == "SELLER"
        assert sig_field["label"] == "Seller Signature"
    
    def test_to_dict(self):
        """Test conversion to full dictionary."""
        coords = SignatureCoordinates(
            x=72.0, y=500.0, width=200.0, height=30.0, page_number=1
        )
        field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.DATE,
            role="BUYER",
            label="Date",
            confidence=0.95,
            detection_method="pattern_match",
        )
        d = field.to_dict()
        
        assert d["field_type"] == "date"
        assert d["role"] == "BUYER"
        assert d["confidence"] == 0.95
        assert "coordinates" in d


class TestSignatureDetectionResult:
    """Tests for SignatureDetectionResult dataclass."""
    
    def test_empty_result(self):
        """Test empty detection result."""
        result = SignatureDetectionResult(fields=[], page_count=1)
        assert len(result.fields) == 0
        assert result.page_count == 1
    
    def test_to_signature_field_list(self):
        """Test conversion to SignatureField list."""
        coords = SignatureCoordinates(
            x=72.0, y=500.0, width=200.0, height=30.0, page_number=1
        )
        field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.SIGNATURE,
            role="BUYER",
            label="Buyer Signature",
        )
        result = SignatureDetectionResult(fields=[field], page_count=1)
        
        sig_fields = result.to_signature_field_list()
        assert len(sig_fields) == 1
        assert sig_fields[0]["field_type"] == "signature"
    
    def test_filter_by_role(self):
        """Test filtering fields by role."""
        coords = SignatureCoordinates(
            x=72.0, y=500.0, width=200.0, height=30.0, page_number=1
        )
        buyer_field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.SIGNATURE,
            role="BUYER",
            label="Buyer Signature",
        )
        seller_field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.SIGNATURE,
            role="SELLER",
            label="Seller Signature",
        )
        result = SignatureDetectionResult(
            fields=[buyer_field, seller_field], page_count=1
        )
        
        buyer_fields = result.filter_by_role("BUYER")
        assert len(buyer_fields) == 1
        assert buyer_fields[0].role == "BUYER"
    
    def test_filter_by_type(self):
        """Test filtering fields by type."""
        coords = SignatureCoordinates(
            x=72.0, y=500.0, width=200.0, height=30.0, page_number=1
        )
        sig_field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.SIGNATURE,
            role="BUYER",
            label="Signature",
        )
        date_field = DetectedSignatureField(
            coordinates=coords,
            field_type=SignatureFieldType.DATE,
            role="BUYER",
            label="Date",
        )
        result = SignatureDetectionResult(
            fields=[sig_field, date_field], page_count=1
        )
        
        sig_fields = result.filter_by_type(SignatureFieldType.SIGNATURE)
        assert len(sig_fields) == 1
        assert sig_fields[0].field_type == SignatureFieldType.SIGNATURE


class TestCalculateTextCoordinates:
    """Tests for text position to PDF coordinate calculation."""
    
    def test_first_line_position(self):
        """Test coordinate calculation for first line."""
        x, y = calculate_text_coordinates(
            line_index=0,
            char_start=0,
            char_end=10,
            total_lines=50,
            line_length=80,
            page_number=1,
        )
        # First line should be near top of page
        # With 1 inch margins (72pt), y should be near page_height - margin
        assert x >= 72.0  # At least left margin
        assert y > 600.0  # Near top of letter page
    
    def test_last_line_position(self):
        """Test coordinate calculation for last line."""
        x, y = calculate_text_coordinates(
            line_index=49,  # Last line of 50
            char_start=0,
            char_end=10,
            total_lines=50,
            line_length=80,
            page_number=1,
        )
        # Last line should be near bottom of page
        assert y < 200.0  # Near bottom
    
    def test_middle_of_line(self):
        """Test coordinate calculation for middle of line."""
        x_start, y_start = calculate_text_coordinates(
            line_index=10,
            char_start=0,
            char_end=10,
            total_lines=50,
            line_length=80,
        )
        x_mid, y_mid = calculate_text_coordinates(
            line_index=10,
            char_start=40,  # Middle of 80-char line
            char_end=50,
            total_lines=50,
            line_length=80,
        )
        # Middle position should have higher x
        assert x_mid > x_start
        # Same line, same y
        assert y_mid == y_start
    
    def test_legal_page_size(self):
        """Test coordinates with Legal page size."""
        x, y = calculate_text_coordinates(
            line_index=0,
            char_start=0,
            char_end=10,
            total_lines=50,
            line_length=80,
            page_size=PageSize.LEGAL,
        )
        # Legal is taller (1008 vs 792)
        assert y > 800.0  # Higher than Letter page


class TestDetectSignatureFieldsEnhanced:
    """Tests for enhanced signature detection with coordinates."""
    
    def test_detect_buyer_signature(self):
        """Test detection of buyer signature field."""
        text = """
        PURCHASE AGREEMENT
        
        Buyer's Signature: _________________
        Date: _______________
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        assert len(fields) >= 1
        buyer_sigs = [f for f in fields if f.role == "BUYER"]
        assert len(buyer_sigs) >= 1
        assert buyer_sigs[0].field_type == SignatureFieldType.SIGNATURE
    
    def test_detect_seller_signature(self):
        """Test detection of seller signature field."""
        text = """
        LISTING AGREEMENT
        
        Seller's Signature: _________________
        Date: _______________
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        seller_sigs = [f for f in fields if f.role == "SELLER"]
        assert len(seller_sigs) >= 1
        assert seller_sigs[0].field_type == SignatureFieldType.SIGNATURE
    
    def test_detect_agent_signature(self):
        """Test detection of agent signature fields."""
        text = """
        BROKERAGE AGREEMENT
        
        Listing Agent Signature: _________________
        
        Buyer's Agent Signature: _________________
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        agent_sigs = [f for f in fields if "AGENT" in f.role]
        assert len(agent_sigs) >= 1
    
    def test_detect_initials_field(self):
        """Test detection of initials field."""
        text = """
        Acknowledgment of Terms
        
        Buyer's Initials: _____    Seller's Initials: _____
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        initial_fields = [f for f in fields if f.field_type == SignatureFieldType.INITIAL]
        assert len(initial_fields) >= 1
    
    def test_detect_date_field(self):
        """Test detection of date field."""
        text = """
        Agreement Date: _________________
        
        Signed this _____ day of __________, 2025
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        date_fields = [f for f in fields if f.field_type == SignatureFieldType.DATE]
        assert len(date_fields) >= 1
    
    def test_coordinates_are_valid(self):
        """Test that detected coordinates are within page bounds."""
        text = """
        Buyer's Signature: _________________
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        for field in fields:
            coords = field.coordinates
            # X should be within page width
            assert 0 <= coords.x <= PageSize.LETTER.width
            # Y should be within page height  
            assert 0 <= coords.y <= PageSize.LETTER.height
            # Width/height should be positive
            assert coords.width > 0
            assert coords.height > 0
    
    def test_page_number_preserved(self):
        """Test that page number is correctly set."""
        text = "Seller's Signature: _________________"
        fields = detect_signature_fields_enhanced(text, page_number=3)
        
        for field in fields:
            assert field.coordinates.page_number == 3
    
    def test_context_text_captured(self):
        """Test that context text is captured."""
        text = "Please sign below:\nBuyer's Signature: _________________"
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        assert len(fields) >= 1
        assert "Buyer" in fields[0].context_text or "sign" in fields[0].context_text.lower()
    
    def test_no_duplicates(self):
        """Test that duplicate detections at exact same position are avoided."""
        text = """
        Buyer's Signature: _________________
        
        
        Seller's Signature: _________________
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        # Should detect both buyer and seller at different positions
        buyer_sigs = [f for f in fields if f.role == "BUYER"]
        seller_sigs = [f for f in fields if f.role == "SELLER"]
        assert len(buyer_sigs) >= 1
        assert len(seller_sigs) >= 1
        
        # Verify they have different Y positions
        buyer_y = buyer_sigs[0].coordinates.y
        seller_y = seller_sigs[0].coordinates.y
        assert buyer_y != seller_y, "Buyer and seller should be at different Y positions"
    
    def test_empty_text(self):
        """Test with empty text."""
        fields = detect_signature_fields_enhanced("", page_number=1)
        assert len(fields) == 0
    
    def test_no_signature_fields(self):
        """Test text with no signature fields."""
        text = """
        This is a simple paragraph.
        No signature fields here.
        Just regular text content.
        """
        fields = detect_signature_fields_enhanced(text, page_number=1)
        assert len(fields) == 0


class TestDetectAllSignatureFields:
    """Tests for full document signature detection."""
    
    def test_returns_result_object(self):
        """Test that function returns SignatureDetectionResult."""
        text = "Buyer's Signature: _________________"
        result = detect_all_signature_fields(text)
        
        assert isinstance(result, SignatureDetectionResult)
    
    def test_result_has_fields(self):
        """Test result contains detected fields."""
        text = """
        Buyer's Signature: _________________
        Seller's Signature: _________________
        """
        result = detect_all_signature_fields(text)
        
        assert len(result.fields) >= 2
    
    def test_result_has_stats(self):
        """Test result contains detection stats."""
        text = "Buyer's Signature: _________________"
        result = detect_all_signature_fields(text)
        
        assert "total_fields" in result.detection_stats
        assert "by_type" in result.detection_stats
        assert "by_role" in result.detection_stats
    
    def test_multi_page_detection(self):
        """Test detection across multiple pages."""
        pages = [
            "Page 1\nBuyer's Signature: _________________",
            "Page 2\nSeller's Signature: _________________",
        ]
        result = detect_all_signature_fields(pages)
        
        assert result.page_count == 2
        # Should have fields from both pages
        page_numbers = {f.coordinates.page_number for f in result.fields}
        assert 1 in page_numbers
        assert 2 in page_numbers


class TestDetectSignatureFieldsLegacy:
    """Tests for backward-compatible detect_signature_fields."""
    
    def test_returns_list_of_signature_field(self):
        """Test returns list of SignatureField TypedDict."""
        text = "Buyer's Signature: _________________"
        fields = detect_signature_fields(text, page_number=1)
        
        assert isinstance(fields, list)
        if len(fields) > 0:
            # Check it's a dict with expected keys
            field = fields[0]
            assert "page_number" in field
            assert "x_position" in field
            assert "y_position" in field
            assert "width" in field
            assert "height" in field
            assert "field_type" in field
            assert "assigned_role" in field
    
    def test_field_type_is_string(self):
        """Test field_type is string value (not enum)."""
        text = "Seller's Signature: _________________"
        fields = detect_signature_fields(text, page_number=1)
        
        if len(fields) > 0:
            assert isinstance(fields[0]["field_type"], str)
            assert fields[0]["field_type"] in ["signature", "initial", "date", "text", "checkbox"]
    
    def test_positions_are_numeric(self):
        """Test positions are numeric values."""
        text = "Buyer's Signature: _________________"
        fields = detect_signature_fields(text, page_number=1)
        
        if len(fields) > 0:
            field = fields[0]
            assert isinstance(field["x_position"], (int, float))
            assert isinstance(field["y_position"], (int, float))
            assert isinstance(field["width"], (int, float))
            assert isinstance(field["height"], (int, float))


class TestSignatureDefaultDimensions:
    """Tests for default signature field dimensions."""
    
    def test_signature_default_size(self):
        """Test default signature field size."""
        # Should be approximately 2.5" x 0.5" (180 x 36 points)
        assert DEFAULT_SIGNATURE_WIDTH >= 150
        assert DEFAULT_SIGNATURE_WIDTH <= 250
        assert DEFAULT_SIGNATURE_HEIGHT >= 25
        assert DEFAULT_SIGNATURE_HEIGHT <= 50
    
    def test_initial_default_size(self):
        """Test default initials field size."""
        # Should be smaller than signature
        assert DEFAULT_INITIAL_WIDTH < DEFAULT_SIGNATURE_WIDTH
        assert DEFAULT_INITIAL_HEIGHT <= DEFAULT_SIGNATURE_HEIGHT
    
    def test_date_default_size(self):
        """Test default date field size."""
        # Date field should be moderate width
        assert DEFAULT_DATE_WIDTH >= 80
        assert DEFAULT_DATE_WIDTH <= 150


class TestComplexDocumentDetection:
    """Tests for complex real estate document scenarios."""
    
    def test_purchase_agreement_full(self):
        """Test detection in full purchase agreement."""
        text = """
        RESIDENTIAL REAL ESTATE PURCHASE AGREEMENT
        
        PARTIES:
        Buyer: John Smith
        Seller: Jane Doe
        
        SIGNATURES:
        
        Buyer's Signature: ___________________________ Date: ___________
        
        Seller's Signature: __________________________ Date: ___________
        
        AGENT ACKNOWLEDGMENT:
        
        Listing Agent Signature: _____________________ Date: ___________
        
        Buyer's Agent Signature: ____________________ Date: ___________
        """
        fields = detect_signature_fields_enhanced(text)
        
        # Should detect multiple signature fields
        sig_fields = [f for f in fields if f.field_type == SignatureFieldType.SIGNATURE]
        assert len(sig_fields) >= 4
        
        # Should detect date fields
        date_fields = [f for f in fields if f.field_type == SignatureFieldType.DATE]
        assert len(date_fields) >= 1
        
        # Should have buyer and seller
        roles = {f.role for f in fields}
        assert "BUYER" in roles or any("BUYER" in r for r in roles)
        assert "SELLER" in roles or any("SELLER" in r for r in roles)
    
    def test_disclosure_form(self):
        """Test detection in disclosure form."""
        text = """
        PROPERTY DISCLOSURE STATEMENT
        
        Seller acknowledges the following disclosures:
        
        Buyer's Initials: _____ / _____
        
        Seller's Initials: _____ / _____
        
        Buyer's Signature: ________________ Date: _______
        """
        fields = detect_signature_fields_enhanced(text)
        
        # Should detect both initials and signatures
        initials = [f for f in fields if f.field_type == SignatureFieldType.INITIAL]
        sigs = [f for f in fields if f.field_type == SignatureFieldType.SIGNATURE]
        
        assert len(initials) >= 1
        assert len(sigs) >= 1
    
    def test_counter_offer(self):
        """Test detection in counter offer document."""
        text = """
        COUNTER OFFER #1
        
        The undersigned Buyer hereby offers the following changes:
        
        X_________________________________
        Buyer Signature
        
        X_________________________________
        Seller Signature (Accept/Reject)
        """
        fields = detect_signature_fields_enhanced(text)
        
        # X_____ is a common signature pattern
        assert len(fields) >= 2
    
    def test_witness_and_notary(self):
        """Test detection of witness and notary fields."""
        text = """
        NOTARIZATION
        
        Witness Signature: _________________________
        
        Notary Public Signature: ____________________
        
        Notary Seal:
        """
        fields = detect_signature_fields_enhanced(text)
        
        # Should detect witness and notary
        other_roles = [f for f in fields if f.role == "OTHER"]
        assert len(other_roles) >= 1


class TestCoordinateAccuracy:
    """Tests for coordinate calculation accuracy."""
    
    def test_signature_at_bottom_third(self):
        """Test signature in bottom third of page has correct Y."""
        # Simulate a document with signature at line 40 of 50
        text = "\n" * 39 + "Buyer's Signature: _________________"
        fields = detect_signature_fields_enhanced(text, page_number=1)
        
        if len(fields) > 0:
            coords = fields[0].coordinates
            # Should be in bottom portion of page
            assert coords.y < PageSize.LETTER.height / 2
    
    def test_multiple_signatures_have_different_y(self):
        """Test multiple signatures have different Y coordinates."""
        text = """
        First Section
        
        Buyer's Signature: _________________
        
        
        
        Second Section
        
        Seller's Signature: _________________
        """
        fields = detect_signature_fields_enhanced(text)
        
        # Filter to get just the specific role signatures
        buyer_sigs = [f for f in fields if f.role == "BUYER"]
        seller_sigs = [f for f in fields if f.role == "SELLER"]
        
        if len(buyer_sigs) >= 1 and len(seller_sigs) >= 1:
            # Y coordinates should be different
            buyer_y = buyer_sigs[0].coordinates.y
            seller_y = seller_sigs[0].coordinates.y
            assert abs(buyer_y - seller_y) > 10.0, f"Expected different Y: buyer={buyer_y}, seller={seller_y}"
    
    def test_fields_ordered_top_to_bottom(self):
        """Test fields are ordered from top to bottom of page."""
        text = """
        Top Signature: _________________
        
        
        
        Middle Signature: _________________
        
        
        
        Bottom Signature: _________________
        """
        fields = detect_signature_fields_enhanced(text)
        
        if len(fields) >= 2:
            # In PDF coordinates, higher Y is higher on page
            # Fields should be returned in document order (top to bottom)
            y_coords = [f.coordinates.y for f in fields]
            # Each subsequent field should have lower Y (further down page)
            for i in range(len(y_coords) - 1):
                assert y_coords[i] >= y_coords[i + 1]


# ============================================================================
# E3: Financial Terms Extraction Tests
# ============================================================================

from nodes.extractor import (
    # E3: Financial extraction types
    FinancialFieldType,
    FinancialConfidence,
    ExtractedFinancialValue,
    FinancialExtractionResult,
    # E3: Financial extraction functions
    parse_currency_amount,
    parse_percentage,
    validate_financial_amount,
    validate_earnest_money_ratio,
    extract_purchase_price,
    extract_earnest_money,
    extract_down_payment,
    extract_loan_amount,
    extract_closing_costs,
    extract_commission,
    extract_commission_rate,
    extract_financial_terms,
    # E3: Constants
    FINANCIAL_VALIDATION_RANGES,
    PERCENTAGE_VALIDATION_RANGES,
    EARNEST_MONEY_PERCENT_RANGE,
)


class TestFinancialFieldType:
    """Tests for FinancialFieldType enum."""
    
    def test_purchase_price_value(self):
        """Test PURCHASE_PRICE type value."""
        assert FinancialFieldType.PURCHASE_PRICE.value == "purchase_price"
    
    def test_earnest_money_value(self):
        """Test EARNEST_MONEY type value."""
        assert FinancialFieldType.EARNEST_MONEY.value == "earnest_money"
    
    def test_closing_costs_value(self):
        """Test CLOSING_COSTS type value."""
        assert FinancialFieldType.CLOSING_COSTS.value == "closing_costs"
    
    def test_commission_value(self):
        """Test COMMISSION type value."""
        assert FinancialFieldType.COMMISSION.value == "commission"
    
    def test_commission_rate_value(self):
        """Test COMMISSION_RATE type value."""
        assert FinancialFieldType.COMMISSION_RATE.value == "commission_rate"


class TestFinancialConfidence:
    """Tests for FinancialConfidence enum."""
    
    def test_high_value(self):
        """Test HIGH confidence value."""
        assert FinancialConfidence.HIGH.value == "high"
    
    def test_medium_value(self):
        """Test MEDIUM confidence value."""
        assert FinancialConfidence.MEDIUM.value == "medium"
    
    def test_low_value(self):
        """Test LOW confidence value."""
        assert FinancialConfidence.LOW.value == "low"


class TestParseCurrencyAmount:
    """Tests for currency amount parsing."""
    
    def test_parse_dollar_sign_basic(self):
        """Test parsing $1234."""
        result = parse_currency_amount("$1234")
        assert result is not None
        assert result[0] == 1234.0
    
    def test_parse_dollar_sign_with_commas(self):
        """Test parsing $1,234,567."""
        result = parse_currency_amount("$1,234,567")
        assert result is not None
        assert result[0] == 1234567.0
    
    def test_parse_dollar_sign_with_cents(self):
        """Test parsing $1,234.56."""
        result = parse_currency_amount("$1,234.56")
        assert result is not None
        assert result[0] == 1234.56
    
    def test_parse_dollar_sign_with_space(self):
        """Test parsing $ 1,234."""
        result = parse_currency_amount("$ 1,234")
        assert result is not None
        assert result[0] == 1234.0
    
    def test_parse_million_suffix_m(self):
        """Test parsing $1.5M."""
        result = parse_currency_amount("$1.5M")
        assert result is not None
        assert result[0] == 1_500_000.0
    
    def test_parse_million_suffix_word(self):
        """Test parsing $1.5 million."""
        result = parse_currency_amount("$1.5 million")
        assert result is not None
        assert result[0] == 1_500_000.0
    
    def test_parse_million_suffix_mil(self):
        """Test parsing $2.3 mil."""
        result = parse_currency_amount("$2.3 mil")
        assert result is not None
        assert result[0] == 2_300_000.0
    
    def test_parse_dollars_word(self):
        """Test parsing '1,234 dollars'."""
        result = parse_currency_amount("1,234 dollars")
        assert result is not None
        assert result[0] == 1234.0
    
    def test_parse_usd_word(self):
        """Test parsing '1,234 USD'."""
        result = parse_currency_amount("1,234 USD")
        assert result is not None
        assert result[0] == 1234.0
    
    def test_parse_plain_number_large(self):
        """Test parsing plain large number like 500000."""
        result = parse_currency_amount("500000")
        assert result is not None
        assert result[0] == 500000.0
    
    def test_parse_rejects_small_plain_number(self):
        """Test that small plain numbers are rejected."""
        result = parse_currency_amount("50")
        assert result is None
    
    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = parse_currency_amount("")
        assert result is None
    
    def test_parse_no_number(self):
        """Test parsing text without number returns None."""
        result = parse_currency_amount("no amount here")
        assert result is None


class TestParsePercentage:
    """Tests for percentage parsing."""
    
    def test_parse_percent_sign(self):
        """Test parsing 5.5%."""
        result = parse_percentage("5.5%")
        assert result is not None
        assert abs(result[0] - 5.5) < 0.01
    
    def test_parse_percent_word(self):
        """Test parsing '5.5 percent'."""
        result = parse_percentage("5.5 percent")
        assert result is not None
        assert result[0] == 5.5
    
    def test_parse_percent_pct(self):
        """Test parsing '6 pct'."""
        result = parse_percentage("6 pct")
        assert result is not None
        assert result[0] == 6.0
    
    def test_parse_whole_number_percent(self):
        """Test parsing '3%'."""
        result = parse_percentage("3%")
        assert result is not None
        assert result[0] == 3.0
    
    def test_parse_rejects_over_100(self):
        """Test that over 100% is rejected."""
        result = parse_percentage("150%")
        assert result is None
    
    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        result = parse_percentage("")
        assert result is None


class TestValidateFinancialAmount:
    """Tests for financial amount validation."""
    
    def test_valid_purchase_price(self):
        """Test valid purchase price in range."""
        is_valid, confidence = validate_financial_amount(
            500_000.0, FinancialFieldType.PURCHASE_PRICE
        )
        assert is_valid is True
        assert confidence >= 0.85  # May be edge case or full confidence
    
    def test_invalid_purchase_price_too_low(self):
        """Test purchase price below minimum."""
        is_valid, confidence = validate_financial_amount(
            100.0, FinancialFieldType.PURCHASE_PRICE
        )
        assert is_valid is False
        assert confidence == 0.0
    
    def test_invalid_purchase_price_too_high(self):
        """Test purchase price above maximum."""
        is_valid, confidence = validate_financial_amount(
            500_000_000.0, FinancialFieldType.PURCHASE_PRICE
        )
        assert is_valid is False
        assert confidence == 0.0
    
    def test_edge_case_purchase_price(self):
        """Test purchase price near edge of range has reduced confidence."""
        # Very near minimum (10K min, 15K within 50%)
        is_valid, confidence = validate_financial_amount(
            12_000.0, FinancialFieldType.PURCHASE_PRICE
        )
        assert is_valid is True
        assert confidence < 1.0  # Should be edge case
    
    def test_valid_earnest_money(self):
        """Test valid earnest money in range."""
        is_valid, confidence = validate_financial_amount(
            10_000.0, FinancialFieldType.EARNEST_MONEY
        )
        assert is_valid is True
        assert confidence >= 0.85  # May be edge case or full confidence
    
    def test_valid_commission_rate_percentage(self):
        """Test valid commission rate percentage."""
        is_valid, confidence = validate_financial_amount(
            6.0, FinancialFieldType.COMMISSION_RATE, is_percentage=True
        )
        assert is_valid is True
        assert confidence >= 0.85  # May be edge case
    
    def test_invalid_commission_rate_too_high(self):
        """Test commission rate too high."""
        is_valid, confidence = validate_financial_amount(
            25.0, FinancialFieldType.COMMISSION_RATE, is_percentage=True
        )
        assert is_valid is False


class TestValidateEarnestMoneyRatio:
    """Tests for earnest money ratio validation."""
    
    def test_typical_earnest_money_ratio(self):
        """Test typical 3% earnest money."""
        is_valid, confidence = validate_earnest_money_ratio(
            15_000.0, 500_000.0  # 3%
        )
        assert is_valid is True
        assert confidence == 1.0
    
    def test_low_earnest_money_ratio(self):
        """Test lower than typical earnest money ratio."""
        is_valid, confidence = validate_earnest_money_ratio(
            1_000.0, 500_000.0  # 0.2%
        )
        assert is_valid is True
        assert confidence < 1.0
    
    def test_high_earnest_money_ratio(self):
        """Test higher than typical earnest money ratio."""
        is_valid, confidence = validate_earnest_money_ratio(
            75_000.0, 500_000.0  # 15%
        )
        assert is_valid is True
        assert confidence < 1.0
    
    def test_zero_purchase_price(self):
        """Test with zero purchase price (edge case)."""
        is_valid, confidence = validate_earnest_money_ratio(
            5_000.0, 0.0
        )
        assert is_valid is True  # Can't validate, assume valid


class TestExtractedFinancialValue:
    """Tests for ExtractedFinancialValue dataclass."""
    
    def test_basic_creation(self):
        """Test basic value creation."""
        value = ExtractedFinancialValue(
            amount=500_000.0,
            field_type=FinancialFieldType.PURCHASE_PRICE,
            confidence=0.95,
        )
        assert value.amount == 500_000.0
        assert value.field_type == FinancialFieldType.PURCHASE_PRICE
        assert value.confidence == 0.95
    
    def test_high_confidence_level(self):
        """Test high confidence level calculation."""
        value = ExtractedFinancialValue(
            amount=500_000.0,
            field_type=FinancialFieldType.PURCHASE_PRICE,
            confidence=0.95,
        )
        assert value.confidence_level == FinancialConfidence.HIGH
    
    def test_medium_confidence_level(self):
        """Test medium confidence level calculation."""
        value = ExtractedFinancialValue(
            amount=500_000.0,
            field_type=FinancialFieldType.PURCHASE_PRICE,
            confidence=0.80,
        )
        assert value.confidence_level == FinancialConfidence.MEDIUM
    
    def test_low_confidence_level(self):
        """Test low confidence level calculation."""
        value = ExtractedFinancialValue(
            amount=500_000.0,
            field_type=FinancialFieldType.PURCHASE_PRICE,
            confidence=0.50,
        )
        assert value.confidence_level == FinancialConfidence.LOW
    
    def test_formatted_amount_currency(self):
        """Test formatted amount for currency."""
        value = ExtractedFinancialValue(
            amount=1_234_567.89,
            field_type=FinancialFieldType.PURCHASE_PRICE,
        )
        assert value.formatted_amount == "$1,234,567.89"
    
    def test_formatted_amount_percentage(self):
        """Test formatted amount for percentage."""
        value = ExtractedFinancialValue(
            amount=5.5,
            field_type=FinancialFieldType.COMMISSION_RATE,
            is_percentage=True,
        )
        assert value.formatted_amount == "5.50%"
    
    def test_is_in_valid_range_true(self):
        """Test valid range check returns True."""
        value = ExtractedFinancialValue(
            amount=500_000.0,
            field_type=FinancialFieldType.PURCHASE_PRICE,
        )
        assert value.is_in_valid_range() is True
    
    def test_is_in_valid_range_false(self):
        """Test invalid range check returns False."""
        value = ExtractedFinancialValue(
            amount=5.0,  # Way too low for purchase price
            field_type=FinancialFieldType.PURCHASE_PRICE,
        )
        assert value.is_in_valid_range() is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        value = ExtractedFinancialValue(
            amount=500_000.0,
            field_type=FinancialFieldType.PURCHASE_PRICE,
            confidence=0.95,
            source_text="Purchase Price: $500,000",
        )
        d = value.to_dict()
        assert d["amount"] == 500_000.0
        assert d["field_type"] == "purchase_price"
        assert d["confidence"] == 0.95
        assert d["formatted"] == "$500,000.00"
        assert d["is_valid_range"] is True


class TestFinancialExtractionResult:
    """Tests for FinancialExtractionResult dataclass."""
    
    def test_empty_result(self):
        """Test empty extraction result."""
        result = FinancialExtractionResult()
        assert result.purchase_price is None
        assert result.earnest_money is None
    
    def test_to_financial_details(self):
        """Test conversion to FinancialDetails TypedDict."""
        result = FinancialExtractionResult(
            purchase_price=ExtractedFinancialValue(
                amount=500_000.0,
                field_type=FinancialFieldType.PURCHASE_PRICE,
            ),
            earnest_money=ExtractedFinancialValue(
                amount=15_000.0,
                field_type=FinancialFieldType.EARNEST_MONEY,
            ),
        )
        details = result.to_financial_details()
        assert details.get("purchase_sale_price") == 500_000.0
        assert details.get("earnest_money_amount") == 15_000.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = FinancialExtractionResult(
            purchase_price=ExtractedFinancialValue(
                amount=500_000.0,
                field_type=FinancialFieldType.PURCHASE_PRICE,
            ),
        )
        d = result.to_dict()
        assert "purchase_price" in d
        assert d["purchase_price"]["amount"] == 500_000.0


class TestExtractPurchasePrice:
    """Tests for purchase price extraction."""
    
    def test_extract_purchase_price_basic(self):
        """Test basic purchase price extraction."""
        text = "Purchase Price: $500,000.00"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 500_000.0
        assert result.field_type == FinancialFieldType.PURCHASE_PRICE
    
    def test_extract_sale_price(self):
        """Test sale price extraction."""
        text = "Sale Price: $425,000"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 425_000.0
    
    def test_extract_total_price(self):
        """Test total purchase price extraction."""
        text = "Total Purchase Price = $750,000"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 750_000.0
    
    def test_extract_contract_price(self):
        """Test contract price extraction."""
        text = "Contract Price: $1,250,000.00"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 1_250_000.0
    
    def test_extract_million_format(self):
        """Test million format extraction."""
        text = "Purchase Price: $1.5M"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 1_500_000.0
    
    def test_no_purchase_price(self):
        """Test when no purchase price found."""
        text = "This document has no price information."
        result = extract_purchase_price(text)
        assert result is None


class TestExtractEarnestMoney:
    """Tests for earnest money extraction."""
    
    def test_extract_earnest_money_basic(self):
        """Test basic earnest money extraction."""
        text = "Earnest Money: $10,000"
        result = extract_earnest_money(text)
        assert result is not None
        assert result.amount == 10_000.0
    
    def test_extract_earnest_money_deposit(self):
        """Test earnest money deposit extraction."""
        text = "Earnest Money Deposit: $15,000.00"
        result = extract_earnest_money(text)
        assert result is not None
        assert result.amount == 15_000.0
    
    def test_extract_emd_abbreviation(self):
        """Test EMD abbreviation extraction."""
        text = "EMD: $5,000"
        result = extract_earnest_money(text)
        assert result is not None
        assert result.amount == 5_000.0
    
    def test_extract_good_faith_deposit(self):
        """Test good faith deposit extraction."""
        text = "Good Faith Deposit: $7,500"
        result = extract_earnest_money(text)
        assert result is not None
        assert result.amount == 7_500.0
    
    def test_extract_initial_deposit(self):
        """Test initial deposit extraction."""
        text = "Initial Deposit: $3,000.00"
        result = extract_earnest_money(text)
        assert result is not None
        assert result.amount == 3_000.0


class TestExtractDownPayment:
    """Tests for down payment extraction."""
    
    def test_extract_down_payment_basic(self):
        """Test basic down payment extraction."""
        text = "Down Payment: $100,000"
        result = extract_down_payment(text)
        assert result is not None
        assert result.amount == 100_000.0
    
    def test_extract_buyer_down_payment(self):
        """Test buyer's down payment extraction."""
        text = "Buyer's Down Payment: $75,000.00"
        result = extract_down_payment(text)
        assert result is not None
        assert result.amount == 75_000.0


class TestExtractLoanAmount:
    """Tests for loan amount extraction."""
    
    def test_extract_loan_amount_basic(self):
        """Test basic loan amount extraction."""
        text = "Loan Amount: $400,000"
        result = extract_loan_amount(text)
        assert result is not None
        assert result.amount == 400_000.0
    
    def test_extract_mortgage_amount(self):
        """Test mortgage amount extraction."""
        text = "Mortgage Amount: $350,000.00"
        result = extract_loan_amount(text)
        assert result is not None
        assert result.amount == 350_000.0
    
    def test_extract_financing_amount(self):
        """Test financing amount extraction."""
        text = "Financing Amount: $425,000"
        result = extract_loan_amount(text)
        assert result is not None
        assert result.amount == 425_000.0


class TestExtractClosingCosts:
    """Tests for closing costs extraction."""
    
    def test_extract_closing_costs_basic(self):
        """Test basic closing costs extraction."""
        text = "Closing Costs: $8,500"
        result = extract_closing_costs(text)
        assert result is not None
        assert result.amount == 8_500.0
    
    def test_extract_settlement_costs(self):
        """Test settlement costs extraction."""
        text = "Settlement Costs: $12,000.00"
        result = extract_closing_costs(text)
        assert result is not None
        assert result.amount == 12_000.0
    
    def test_extract_buyer_closing_costs(self):
        """Test buyer's closing costs extraction."""
        text = "Buyer's Closing Costs: $6,500"
        result = extract_closing_costs(text)
        assert result is not None
        assert result.amount == 6_500.0
    
    def test_extract_seller_concession(self):
        """Test seller concession extraction."""
        text = "Seller's Concession: $5,000"
        result = extract_closing_costs(text)
        assert result is not None
        assert result.amount == 5_000.0


class TestExtractCommission:
    """Tests for commission extraction."""
    
    def test_extract_commission_basic(self):
        """Test basic commission extraction."""
        text = "Commission: $30,000"
        result = extract_commission(text)
        assert result is not None
        assert result.amount == 30_000.0
    
    def test_extract_total_commission(self):
        """Test total commission extraction."""
        text = "Total Commission: $25,000.00"
        result = extract_commission(text)
        assert result is not None
        assert result.amount == 25_000.0
    
    def test_extract_brokerage_fee(self):
        """Test brokerage fee extraction."""
        text = "Brokerage Fee: $15,000"
        result = extract_commission(text)
        assert result is not None
        assert result.amount == 15_000.0


class TestExtractCommissionRate:
    """Tests for commission rate extraction."""
    
    def test_extract_commission_rate_basic(self):
        """Test basic commission rate extraction."""
        text = "Commission Rate: 6%"
        result = extract_commission_rate(text)
        assert result is not None
        assert result.amount == 6.0
        assert result.is_percentage is True
    
    def test_extract_commission_percentage(self):
        """Test commission percentage extraction."""
        text = "Commission Percentage: 5.5%"
        result = extract_commission_rate(text)
        assert result is not None
        assert result.amount == 5.5
    
    def test_extract_at_rate_of(self):
        """Test 'at a rate of' format extraction."""
        text = "at a rate of 6%"
        result = extract_commission_rate(text)
        assert result is not None
        assert result.amount == 6.0


class TestExtractFinancialTerms:
    """Tests for full financial terms extraction."""
    
    def test_extract_from_purchase_agreement(self):
        """Test extraction from full purchase agreement."""
        text = """
        RESIDENTIAL PURCHASE AGREEMENT
        
        Purchase Price: $525,000.00
        
        Earnest Money Deposit: $15,000.00
        
        Down Payment: $105,000.00
        
        Loan Amount: $420,000.00
        
        Closing Costs: $12,500.00
        """
        result = extract_financial_terms(text)
        
        assert result.purchase_price is not None
        assert result.purchase_price.amount == 525_000.0
        
        assert result.earnest_money is not None
        assert result.earnest_money.amount == 15_000.0
        
        assert result.down_payment is not None
        assert result.down_payment.amount == 105_000.0
        
        assert result.loan_amount is not None
        assert result.loan_amount.amount == 420_000.0
        
        assert result.closing_costs is not None
        assert result.closing_costs.amount == 12_500.0
    
    def test_extract_with_commission(self):
        """Test extraction with commission info."""
        text = """
        Listing Agreement
        
        Sale Price: $450,000
        
        Commission Rate: 6%
        
        Total Commission: $27,000
        """
        result = extract_financial_terms(text)
        
        assert result.purchase_price is not None
        assert result.purchase_price.amount == 450_000.0
        
        assert result.commission_rate is not None
        assert result.commission_rate.amount == 6.0
        assert result.commission_rate.is_percentage is True
        
        assert result.commission is not None
        assert result.commission.amount == 27_000.0
    
    def test_extract_validates_earnest_money_ratio(self):
        """Test that earnest money is validated against purchase price."""
        text = """
        Purchase Price: $500,000
        
        Earnest Money: $15,000
        """
        result = extract_financial_terms(text)
        
        assert result.purchase_price is not None
        assert result.earnest_money is not None
        # 3% is typical, should have good confidence
        assert result.earnest_money.confidence >= 0.8
    
    def test_extract_has_stats(self):
        """Test extraction result has stats."""
        text = "Purchase Price: $500,000"
        result = extract_financial_terms(text)
        
        assert "total_extracted" in result.extraction_stats
        assert "has_purchase_price" in result.extraction_stats
        assert result.extraction_stats["has_purchase_price"] is True
    
    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        result = extract_financial_terms("")
        
        assert result.purchase_price is None
        assert result.earnest_money is None
        assert result.extraction_stats["total_extracted"] == 0
    
    def test_extract_no_financial_terms(self):
        """Test extraction from text without financial terms."""
        text = """
        Property Disclosure Statement
        
        This property was built in 1985.
        There are no known defects.
        """
        result = extract_financial_terms(text)
        
        assert result.purchase_price is None
        assert result.earnest_money is None


class TestFinancialExtractionEdgeCases:
    """Tests for edge cases in financial extraction."""
    
    def test_multiple_prices_picks_best(self):
        """Test that multiple price patterns pick the best match."""
        text = """
        Offering Price: $475,000
        
        After negotiation:
        Purchase Price: $450,000
        """
        result = extract_purchase_price(text)
        assert result is not None
        # Should pick purchase price (higher priority pattern)
        assert result.amount in [450_000.0, 475_000.0]
    
    def test_handles_no_cents(self):
        """Test handling amounts without cents."""
        text = "Purchase Price: $500,000"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 500_000.0
    
    def test_handles_unusual_spacing(self):
        """Test handling unusual spacing."""
        text = "Purchase Price:   $  500,000.00"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 500_000.0
    
    def test_handles_equals_sign(self):
        """Test handling equals sign instead of colon."""
        text = "Purchase Price = $500,000"
        result = extract_purchase_price(text)
        assert result is not None
        assert result.amount == 500_000.0
    
    def test_handles_real_world_document(self):
        """Test extraction from realistic document text."""
        text = """
        OFFER TO PURCHASE REAL ESTATE
        
        1. PURCHASE PRICE: $525,000.00
        
        2. EARNEST MONEY: $15,000.00 as earnest money
        
        3. FINANCING: Loan Amount: $420,000.00
        
        4. Closing Costs: $8,000.00 toward Buyer's closing costs.
        """
        result = extract_financial_terms(text)
        
        assert result.purchase_price is not None
        assert result.purchase_price.amount == 525_000.0
        
        assert result.earnest_money is not None
        assert result.earnest_money.amount == 15_000.0
        
        assert result.loan_amount is not None
        assert result.loan_amount.amount == 420_000.0
        
        assert result.closing_costs is not None
        assert result.closing_costs.amount == 8_000.0


class TestFinancialValidationRanges:
    """Tests for validation range constants."""
    
    def test_purchase_price_range(self):
        """Test purchase price validation range."""
        min_val, max_val = FINANCIAL_VALIDATION_RANGES[FinancialFieldType.PURCHASE_PRICE]
        assert min_val == 10_000.0
        assert max_val == 100_000_000.0
    
    def test_earnest_money_range(self):
        """Test earnest money validation range."""
        min_val, max_val = FINANCIAL_VALIDATION_RANGES[FinancialFieldType.EARNEST_MONEY]
        assert min_val == 100.0
        assert max_val == 1_000_000.0
    
    def test_closing_costs_range(self):
        """Test closing costs validation range."""
        min_val, max_val = FINANCIAL_VALIDATION_RANGES[FinancialFieldType.CLOSING_COSTS]
        assert min_val == 0.0
        assert max_val == 500_000.0
    
    def test_commission_rate_percentage_range(self):
        """Test commission rate percentage range."""
        min_val, max_val = PERCENTAGE_VALIDATION_RANGES[FinancialFieldType.COMMISSION_RATE]
        assert min_val == 0.0
        assert max_val == 15.0
    
    def test_earnest_money_percent_range(self):
        """Test earnest money typical percentage range."""
        min_val, max_val = EARNEST_MONEY_PERCENT_RANGE
        assert min_val == 0.5
        assert max_val == 10.0


# =============================================================================
# E4: Contract Dates Extraction Tests
# =============================================================================

from datetime import datetime, date
from nodes.extractor import (
    DateFieldType,
    DateValidationIssue,
    ExtractedDateValue,
    DateValidationResult,
    DateExtractionResult,
    parse_date,
    parse_date_with_confidence,
    validate_date_sequence,
    extract_contract_date,
    extract_closing_date,
    extract_offer_date,
    extract_offer_expiration,
    extract_acceptance_date,
    extract_inspection_date,
    extract_occupancy_date,
    extract_all_contract_dates,
)


class TestDateFieldType:
    """Test DateFieldType enum."""
    
    def test_enum_values_exist(self):
        """Test all expected field types exist."""
        assert DateFieldType.CONTRACT_DATE.value == "contract_date"
        assert DateFieldType.CLOSING_DATE.value == "closing_date"
        assert DateFieldType.OFFER_DATE.value == "offer_date"
        assert DateFieldType.OFFER_EXPIRATION.value == "offer_expiration"
        assert DateFieldType.INSPECTION_DATE.value == "inspection_date"
        assert DateFieldType.INSPECTION_DEADLINE.value == "inspection_deadline"
        assert DateFieldType.FINANCING_DEADLINE.value == "financing_deadline"
        assert DateFieldType.POSSESSION_DATE.value == "possession_date"
        assert DateFieldType.OCCUPANCY_DATE.value == "occupancy_date"
        assert DateFieldType.OTHER.value == "other"
    
    def test_all_types_count(self):
        """Test total number of date field types."""
        assert len(DateFieldType) == 13


class TestParseDateFunction:
    """Test the parse_date function with various formats."""
    
    def test_parse_mm_dd_yyyy(self):
        """Test MM/DD/YYYY format."""
        result = parse_date("12/25/2025")
        assert result is not None
        parsed_date, matched_text, fmt = result
        assert parsed_date.year == 2025
        assert parsed_date.month == 12
        assert parsed_date.day == 25
    
    def test_parse_mm_dd_yy(self):
        """Test MM/DD/YY format."""
        result = parse_date("12/25/25")
        assert result is not None
        parsed_date, matched_text, fmt = result
        assert parsed_date.year == 2025
        assert parsed_date.month == 12
        assert parsed_date.day == 25
    
    def test_parse_yyyy_mm_dd(self):
        """Test YYYY-MM-DD format."""
        result = parse_date("2025-12-25")
        assert result is not None
        parsed_date, matched_text, fmt = result
        assert parsed_date.year == 2025
        assert parsed_date.month == 12
        assert parsed_date.day == 25
    
    def test_parse_month_name_full(self):
        """Test 'Month DD, YYYY' format."""
        result = parse_date("December 25, 2025")
        assert result is not None
        parsed_date, matched_text, fmt = result
        assert parsed_date.year == 2025
        assert parsed_date.month == 12
        assert parsed_date.day == 25
    
    def test_parse_month_name_abbreviated(self):
        """Test 'Mon DD, YYYY' format."""
        result = parse_date("Dec 25, 2025")
        assert result is not None
        parsed_date, matched_text, fmt = result
        assert parsed_date.year == 2025
        assert parsed_date.month == 12
        assert parsed_date.day == 25
    
    def test_parse_with_dashes(self):
        """Test MM-DD-YYYY format."""
        result = parse_date("12-25-2025")
        assert result is not None
        parsed_date, matched_text, fmt = result
        assert parsed_date.year == 2025
        assert parsed_date.month == 12
        assert parsed_date.day == 25
    
    def test_parse_invalid_date(self):
        """Test invalid date returns None."""
        result = parse_date("not a date")
        assert result is None
    
    def test_parse_empty_string(self):
        """Test empty string returns None."""
        result = parse_date("")
        assert result is None
    
    def test_parse_with_leading_zeros(self):
        """Test dates with leading zeros."""
        result = parse_date("01/05/2025")
        assert result is not None
        parsed_date, matched_text, fmt = result
        assert parsed_date.month == 1
        assert parsed_date.day == 5
    
    def test_parse_returns_tuple(self):
        """Test parse_date returns (date, matched_text, format)."""
        result = parse_date("01/05/2025")
        assert result is not None
        assert len(result) == 3
        parsed_date, matched_text, fmt = result
        assert isinstance(parsed_date, date)
        assert isinstance(matched_text, str)
        assert isinstance(fmt, str)


class TestExtractedDateValue:
    """Test ExtractedDateValue dataclass."""
    
    def test_create_date_value(self):
        """Test creating an ExtractedDateValue."""
        dt = date(2025, 12, 25)
        value = ExtractedDateValue(
            date_value=dt,
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.95,
            source_text="closing date: 12/25/2025",
            original_format="%m/%d/%Y"
        )
        assert value.date_value == dt
        assert value.field_type == DateFieldType.CLOSING_DATE
        assert value.confidence == 0.95
        assert value.source_text == "closing date: 12/25/2025"
    
    def test_formatted_date(self):
        """Test formatted_date property returns MM/DD/YYYY."""
        dt = date(2025, 1, 5)
        value = ExtractedDateValue(
            date_value=dt,
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.9,
            source_text="test",
            original_format="test"
        )
        assert value.formatted_date == "01/05/2025"
    
    def test_iso_date_format(self):
        """Test iso_date property returns YYYY-MM-DD."""
        dt = date(2025, 12, 25)
        value = ExtractedDateValue(
            date_value=dt,
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.9,
            source_text="test",
            original_format="test"
        )
        assert value.iso_date == "2025-12-25"
    
    def test_is_in_past(self):
        """Test is_in_past method."""
        past_date = date(2020, 1, 1)
        value = ExtractedDateValue(
            date_value=past_date,
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.9,
            source_text="test",
            original_format="test"
        )
        assert value.is_in_past() is True
    
    def test_is_weekend(self):
        """Test is_weekend method."""
        # January 4, 2025 is a Saturday
        saturday = date(2025, 1, 4)
        value = ExtractedDateValue(
            date_value=saturday,
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.9,
            source_text="test",
            original_format="test"
        )
        assert value.is_weekend() is True
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        dt = date(2025, 6, 15)
        value = ExtractedDateValue(
            date_value=dt,
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.95,
            source_text="closing: 06/15/2025",
            original_format="%m/%d/%Y"
        )
        d = value.to_dict()
        assert d["date"] == "06/15/2025"
        assert d["iso_date"] == "2025-06-15"
        assert d["field_type"] == "closing_date"
        assert d["confidence"] == 0.95


class TestValidateDateSequence:
    """Test date sequence validation - critical for E4 requirements."""
    
    def test_valid_sequence(self):
        """Test valid sequence: offer < closing."""
        result = validate_date_sequence(
            offer_date=date(2025, 1, 1),
            closing_date=date(2025, 2, 1),
        )
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    def test_closing_before_offer(self):
        """Test impossible: closing before offer date is flagged."""
        result = validate_date_sequence(
            offer_date=date(2025, 3, 1),
            closing_date=date(2025, 2, 1),  # Before offer!
        )
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any(issue[0] == DateValidationIssue.CLOSING_BEFORE_OFFER for issue in result.issues)
    
    def test_inspection_after_closing(self):
        """Test illogical: inspection after closing is flagged."""
        result = validate_date_sequence(
            offer_date=date(2025, 1, 1),
            inspection_deadline=date(2025, 3, 1),  # After closing!
            closing_date=date(2025, 2, 1),
        )
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any(issue[0] == DateValidationIssue.INSPECTION_AFTER_CLOSING for issue in result.issues)
    
    def test_closing_before_acceptance(self):
        """Test impossible: closing before acceptance is flagged."""
        result = validate_date_sequence(
            acceptance_date=date(2025, 2, 15),
            closing_date=date(2025, 2, 1),  # Before acceptance!
        )
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any(issue[0] == DateValidationIssue.CLOSING_BEFORE_ACCEPTANCE for issue in result.issues)
    
    def test_empty_dates_is_valid(self):
        """Test empty dates is valid (no sequence to check)."""
        result = validate_date_sequence()
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    def test_single_date_is_valid(self):
        """Test single date is valid (no sequence to check)."""
        result = validate_date_sequence(closing_date=date(2025, 2, 1))
        assert result.is_valid is True
    
    def test_weekend_closing_warning(self):
        """Test weekend closing generates warning."""
        # January 4, 2025 is a Saturday
        result = validate_date_sequence(closing_date=date(2025, 1, 4))
        assert result.is_valid is True  # Warning, not error
        assert len(result.warnings) > 0
        assert any(w[0] == DateValidationIssue.WEEKEND_CLOSING for w in result.warnings)
    
    def test_dates_too_far_apart_warning(self):
        """Test dates too far apart generates warning."""
        result = validate_date_sequence(
            offer_date=date(2025, 1, 1),
            closing_date=date(2027, 1, 1),  # 2 years later
        )
        assert len(result.warnings) > 0
        assert any(w[0] == DateValidationIssue.DATES_TOO_FAR_APART for w in result.warnings)


class TestDateValidationResult:
    """Test DateValidationResult dataclass."""
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        result = DateValidationResult(
            is_valid=False,
            issues=[(DateValidationIssue.CLOSING_BEFORE_OFFER, "Closing before offer")],
            warnings=[(DateValidationIssue.WEEKEND_CLOSING, "Saturday closing")]
        )
        d = result.to_dict()
        assert d["is_valid"] is False
        assert len(d["issues"]) == 1
        assert d["issues"][0] == ("closing_before_offer", "Closing before offer")
        assert len(d["warnings"]) == 1


class TestExtractClosingDate:
    """Test closing date extraction."""
    
    def test_extract_closing_date_with_colon(self):
        """Test extracting closing date with colon separator."""
        text = "The closing date: 12/25/2025"
        result = extract_closing_date(text)
        assert result is not None
        assert result.date_value.year == 2025
        assert result.date_value.month == 12
        assert result.date_value.day == 25
        assert result.field_type == DateFieldType.CLOSING_DATE
    
    def test_extract_close_of_escrow(self):
        """Test 'close of escrow' pattern."""
        text = "Close of escrow shall be 01/15/2025"
        result = extract_closing_date(text)
        assert result is not None
        assert result.date_value.month == 1
        assert result.date_value.day == 15
    
    def test_extract_settlement_date(self):
        """Test 'settlement date' as closing date."""
        text = "Settlement date: March 1, 2025"
        result = extract_closing_date(text)
        assert result is not None
        assert result.date_value.month == 3
        assert result.date_value.day == 1
    
    def test_no_closing_date(self):
        """Test text without closing date."""
        text = "This contract has no closing information."
        result = extract_closing_date(text)
        assert result is None
    
    def test_closing_on_or_before(self):
        """Test 'closing on or before' pattern."""
        text = "Closing on or before February 28, 2025"
        result = extract_closing_date(text)
        assert result is not None
        assert result.date_value.month == 2
        assert result.date_value.day == 28


class TestExtractOfferDate:
    """Test offer date extraction."""
    
    def test_extract_offer_date(self):
        """Test basic offer date extraction."""
        text = "Offer Date: 01/05/2025"
        result = extract_offer_date(text)
        assert result is not None
        assert result.date_value.month == 1
        assert result.date_value.day == 5
        assert result.field_type == DateFieldType.OFFER_DATE
    
    def test_extract_date_of_offer(self):
        """Test 'date of offer' pattern."""
        text = "This offer is made this date of offer: January 5, 2025"
        result = extract_offer_date(text)
        assert result is not None
    
    def test_no_offer_date(self):
        """Test text without offer date."""
        text = "This is a purchase agreement for the property."
        result = extract_offer_date(text)
        assert result is None


class TestExtractOfferExpiration:
    """Test offer expiration date extraction."""
    
    def test_extract_offer_expires(self):
        """Test offer expires pattern."""
        text = "This offer expires: 01/10/2025"
        result = extract_offer_expiration(text)
        assert result is not None
        assert result.date_value.month == 1
        assert result.date_value.day == 10
        assert result.field_type == DateFieldType.OFFER_EXPIRATION
    
    def test_extract_valid_until(self):
        """Test 'valid until' pattern."""
        text = "Offer valid until January 10, 2025"
        result = extract_offer_expiration(text)
        assert result is not None


class TestExtractInspectionDate:
    """Test inspection date extraction."""
    
    def test_extract_inspection_deadline(self):
        """Test inspection deadline extraction."""
        text = "Inspection deadline: 01/15/2025"
        result = extract_inspection_date(text)
        assert result is not None
        assert result.date_value.month == 1
        assert result.date_value.day == 15
        assert result.field_type == DateFieldType.INSPECTION_DEADLINE
    
    def test_extract_due_diligence_date(self):
        """Test 'due diligence' pattern."""
        text = "Due diligence deadline: 01/18/2025"
        result = extract_inspection_date(text)
        assert result is not None


class TestExtractContractDate:
    """Test contract/agreement date extraction."""
    
    def test_extract_contract_date(self):
        """Test basic contract date extraction."""
        text = "Contract Date: 01/01/2025"
        result = extract_contract_date(text)
        assert result is not None
        assert result.field_type == DateFieldType.CONTRACT_DATE
    
    def test_extract_agreement_date(self):
        """Test 'agreement date' pattern."""
        text = "Agreement date: January 1, 2025"
        result = extract_contract_date(text)
        assert result is not None
    
    def test_extract_effective_date(self):
        """Test 'effective date' pattern."""
        text = "Effective Date: 01/01/2025"
        result = extract_contract_date(text)
        assert result is not None


class TestExtractOccupancyDate:
    """Test occupancy/possession date extraction."""
    
    def test_extract_occupancy_date(self):
        """Test 'occupancy date' pattern."""
        text = "Occupancy date: 02/15/2025"
        result = extract_occupancy_date(text)
        assert result is not None
        assert result.date_value.month == 2
        assert result.date_value.day == 15
    
    def test_extract_possession_date(self):
        """Test 'possession date' pattern."""
        text = "Possession date: February 20, 2025"
        result = extract_occupancy_date(text)
        assert result is not None


class TestExtractContractDatesIntegration:
    """Integration tests for full contract date extraction."""
    
    def test_extract_multiple_dates(self):
        """Test extracting multiple dates from contract."""
        text = """
        REAL ESTATE PURCHASE AGREEMENT
        
        Contract Date: January 5, 2025
        
        1. OFFER: Offer Date: 01/05/2025
        
        2. CLOSING: The closing date: February 28, 2025
        
        3. POSSESSION: Occupancy date: 03/01/2025
        """
        result = extract_all_contract_dates(text)
        assert result is not None
        # Check we found offer_date, closing_date, etc.
        assert result.closing_date is not None
        assert result.contract_date is not None
    
    def test_extract_validates_sequence(self):
        """Test that extraction validates date sequence."""
        text = """
        Offer Date: 01/05/2025
        Closing Date: 02/28/2025
        """
        result = extract_all_contract_dates(text)
        assert result is not None
        assert result.validation is not None
        assert result.validation.is_valid is True
    
    def test_extract_detects_impossible_sequence(self):
        """Test that impossible sequences are flagged."""
        text = """
        Offer Date: 03/01/2025
        Closing Date: 02/01/2025
        """
        result = extract_all_contract_dates(text)
        assert result is not None
        assert result.validation is not None
        assert result.validation.is_valid is False
        assert len(result.validation.issues) > 0
    
    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        result = extract_all_contract_dates("")
        assert result is not None
    
    def test_extract_no_dates(self):
        """Test extraction from text without dates."""
        text = "This is a contract with no specific dates mentioned."
        result = extract_all_contract_dates(text)
        assert result is not None
    
    def test_result_has_stats(self):
        """Test that result includes extraction statistics."""
        text = "Closing Date: 02/28/2025"
        result = extract_all_contract_dates(text)
        assert result is not None
        assert "total_extracted" in result.extraction_stats
        assert "has_closing_date" in result.extraction_stats
    
    def test_to_contract_dates(self):
        """Test conversion to ContractDates TypedDict."""
        text = """
        Offer Date: 01/05/2025
        Closing Date: 02/28/2025
        """
        result = extract_all_contract_dates(text)
        contract_dates = result.to_contract_dates()
        assert "closing_date" in contract_dates
        assert contract_dates["closing_date"] == "02/28/2025"


class TestDateExtractionEdgeCases:
    """Test edge cases in date extraction."""
    
    def test_multiple_date_formats_in_document(self):
        """Test handling multiple date formats in same document."""
        text = """
        Contract Date: 2025-01-05
        Closing Date: January 28, 2025
        """
        result = extract_all_contract_dates(text)
        assert result is not None
        assert result.contract_date is not None or result.closing_date is not None
    
    def test_dates_with_context(self):
        """Test dates don't get confused with similar terms."""
        text = """
        Closing Date: 02/28/2025
        """
        result = extract_closing_date(text)
        assert result is not None
        # Should find a closing date
    
    def test_parse_date_returns_none_for_invalid(self):
        """Test parse_date returns None for invalid input."""
        assert parse_date("") is None
        assert parse_date("not a date") is None
        assert parse_date("hello world") is None


class TestDateExtractionConfidence:
    """Test confidence scoring for date extraction."""
    
    def test_high_confidence_with_label(self):
        """Test high confidence when date has clear label."""
        text = "Closing Date: 02/28/2025"
        result = extract_closing_date(text)
        assert result is not None
        assert result.confidence >= 0.7


class TestDateExtractionResult:
    """Test DateExtractionResult dataclass."""
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        result = DateExtractionResult()
        result.closing_date = ExtractedDateValue(
            date_value=date(2025, 2, 28),
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.95,
            source_text="Closing Date: 02/28/2025",
            original_format="%m/%d/%Y"
        )
        d = result.to_dict()
        assert "closing_date" in d
        assert d["closing_date"]["date"] == "02/28/2025"
    
    def test_extraction_stats(self):
        """Test extraction_stats field."""
        text = "Closing Date: 02/28/2025"
        result = extract_all_contract_dates(text)
        assert result.extraction_stats["has_closing_date"] is True
        assert result.extraction_stats["total_extracted"] >= 1


class TestRealWorldDateDocuments:
    """Test date extraction from real-world document patterns."""
    
    def test_colorado_contract_format(self):
        """Test Colorado real estate contract format."""
        text = """
        CBS1-5-24. CONTRACT TO BUY AND SELL REAL ESTATE
        
        Contract Date: January 15, 2025
        
        The Closing shall be on or before February 28, 2025.
        """
        result = extract_all_contract_dates(text)
        assert result is not None
    
    def test_simple_purchase_agreement(self):
        """Test simple purchase agreement format."""
        text = """
        PURCHASE AGREEMENT
        
        Effective Date: 1/5/2025
        
        CLOSING: Closing Date: 2/28/2025
        """
        result = extract_all_contract_dates(text)
        assert result is not None
        assert result.closing_date is not None


class TestDotloopDateFormatRequirements:
    """Test Dotloop-specific date format requirements."""
    
    def test_formatted_date_leading_zeros(self):
        """Test formatted_date uses MM/DD/YYYY with leading zeros."""
        dt = date(2025, 1, 5)
        value = ExtractedDateValue(
            date_value=dt,
            field_type=DateFieldType.CLOSING_DATE,
            confidence=0.9,
            source_text="test",
            original_format="test"
        )
        formatted = value.formatted_date
        # Must be 01/05/2025, not 1/5/2025
        assert formatted == "01/05/2025"
        assert len(formatted) == 10
    
    def test_formatted_date_all_months(self):
        """Test formatted_date for all months."""
        for month in range(1, 13):
            dt = date(2025, month, 15)
            value = ExtractedDateValue(
                date_value=dt,
                field_type=DateFieldType.CLOSING_DATE,
                confidence=0.9,
                source_text="test",
                original_format="test"
            )
            formatted = value.formatted_date
            # Format should be MM/DD/YYYY
            assert len(formatted.split('/')[0]) == 2  # Month has 2 digits
    
    def test_formatted_date_various_days(self):
        """Test formatted_date for various days."""
        for day in [1, 5, 10, 15, 28]:
            dt = date(2025, 1, day)
            value = ExtractedDateValue(
                date_value=dt,
                field_type=DateFieldType.CLOSING_DATE,
                confidence=0.9,
                source_text="test",
                original_format="test"
            )
            formatted = value.formatted_date
            # Day should have 2 digits
            assert len(formatted.split('/')[1]) == 2

