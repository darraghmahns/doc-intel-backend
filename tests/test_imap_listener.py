"""
Unit Tests for IMAP Listener Node

Tests the email detection and PDF attachment extraction logic
without requiring real IMAP credentials.

Run with: pytest tests/test_imap_listener.py -v
"""

import os
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pytest

# Import the module under test
from nodes.imap_listener import (
    IMAPConfig,
    IMAPAccountConfig,
    is_pdf_attachment,
    extract_pdf_attachments,
    build_email_metadata,
    fetch_unprocessed_emails,
    fetch_emails_from_all_accounts,
    load_imap_accounts,
    imap_listener_node,
    extract_domain,
    is_sender_allowed,
    is_sender_allowed_for_account,
    with_retry,
    RetryResult,
    _fetch_emails_from_mailbox,
    _fetch_emails_from_account,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def imap_config():
    """Create a test IMAP configuration."""
    config = IMAPConfig()
    config.host = "imap.test.com"
    config.username = "test@test.com"
    config.password = "testpassword"
    config.attachment_dir = tempfile.gettempdir()
    return config


@pytest.fixture
def mock_pdf_attachment():
    """Create a mock PDF attachment."""
    attachment = Mock()
    attachment.filename = "contract.pdf"
    attachment.payload = b"%PDF-1.4 fake pdf content" + b"x" * 2000  # > 1KB
    return attachment


@pytest.fixture
def mock_non_pdf_attachment():
    """Create a mock non-PDF attachment."""
    attachment = Mock()
    attachment.filename = "image.jpg"
    attachment.payload = b"fake image content" * 100
    return attachment


@pytest.fixture
def mock_email_message(mock_pdf_attachment):
    """Create a mock email message with PDF attachment."""
    msg = Mock()
    msg.uid = "12345"
    msg.subject = "Contract for 123 Main St"
    msg.from_ = "agent@realty.com"
    msg.to = ["buyer@email.com"]
    msg.date = datetime(2024, 12, 1, 10, 30, 0)
    msg.headers = {"message-id": ["<abc123@mail.com>"]}
    msg.attachments = [mock_pdf_attachment]
    return msg


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


# ============================================================================
# Test: is_pdf_attachment
# ============================================================================

class TestIsPdfAttachment:
    """Tests for the is_pdf_attachment function."""
    
    def test_pdf_extension_lowercase(self):
        """Should return True for .pdf extension."""
        assert is_pdf_attachment("document.pdf") is True
    
    def test_pdf_extension_uppercase(self):
        """Should return True for .PDF extension."""
        assert is_pdf_attachment("DOCUMENT.PDF") is True
    
    def test_pdf_extension_mixed_case(self):
        """Should return True for .Pdf extension."""
        assert is_pdf_attachment("Document.Pdf") is True
    
    def test_non_pdf_extension(self):
        """Should return False for non-PDF extensions."""
        assert is_pdf_attachment("image.jpg") is False
        assert is_pdf_attachment("document.docx") is False
        assert is_pdf_attachment("spreadsheet.xlsx") is False
    
    def test_empty_filename(self):
        """Should return False for empty filename."""
        assert is_pdf_attachment("") is False
    
    def test_none_filename(self):
        """Should return False for None filename."""
        assert is_pdf_attachment(None) is False
    
    def test_no_extension(self):
        """Should return False for filename without extension."""
        assert is_pdf_attachment("document") is False
    
    def test_pdf_in_name_but_wrong_extension(self):
        """Should return False when pdf is in name but not extension."""
        assert is_pdf_attachment("pdf_document.txt") is False


# ============================================================================
# Test: IMAPConfig
# ============================================================================

class TestIMAPConfig:
    """Tests for the IMAPConfig class."""
    
    def test_default_values(self):
        """Should have sensible defaults."""
        config = IMAPConfig()
        assert config.host == "imap.gmail.com"
        assert config.port == 993
        assert config.inbox_folder == "INBOX"
    
    def test_is_valid_with_credentials(self):
        """Should return True when credentials are set."""
        config = IMAPConfig()
        config.username = "test@test.com"
        config.password = "password123"
        config.host = "imap.test.com"
        assert config.is_valid() is True
    
    def test_is_valid_without_username(self):
        """Should return False when username is missing."""
        config = IMAPConfig()
        config.password = "password123"
        assert config.is_valid() is False
    
    def test_is_valid_without_password(self):
        """Should return False when password is missing."""
        config = IMAPConfig()
        config.username = "test@test.com"
        assert config.is_valid() is False
    
    @patch.dict(os.environ, {
        "IMAP_HOST": "custom.imap.com",
        "IMAP_PORT": "995",
        "IMAP_USERNAME": "user@custom.com",
        "IMAP_PASSWORD": "secret",
    })
    def test_loads_from_environment(self):
        """Should load configuration from environment variables."""
        config = IMAPConfig()
        assert config.host == "custom.imap.com"
        assert config.port == 995
        assert config.username == "user@custom.com"
        assert config.password == "secret"


# ============================================================================
# Test: extract_pdf_attachments
# ============================================================================

class TestExtractPdfAttachments:
    """Tests for the extract_pdf_attachments function."""
    
    def test_extracts_pdf_attachment(self, mock_email_message, imap_config):
        """Should extract and save PDF attachments."""
        result = extract_pdf_attachments(mock_email_message, imap_config)
        
        assert len(result) == 1
        assert result[0]["filename"] == "contract.pdf"
        assert os.path.exists(result[0]["path"])
        assert result[0]["size"] > 0
        
        # Cleanup
        os.remove(result[0]["path"])
    
    def test_skips_non_pdf_attachments(self, mock_non_pdf_attachment, imap_config):
        """Should skip non-PDF attachments."""
        msg = Mock()
        msg.uid = "12345"
        msg.attachments = [mock_non_pdf_attachment]
        
        result = extract_pdf_attachments(msg, imap_config)
        
        assert len(result) == 0
    
    def test_skips_small_attachments(self, imap_config):
        """Should skip attachments smaller than minimum size."""
        small_attachment = Mock()
        small_attachment.filename = "tiny.pdf"
        small_attachment.payload = b"small"  # < 1KB
        
        msg = Mock()
        msg.uid = "12345"
        msg.attachments = [small_attachment]
        
        result = extract_pdf_attachments(msg, imap_config)
        
        assert len(result) == 0
    
    def test_skips_large_attachments(self, imap_config):
        """Should skip attachments larger than maximum size."""
        large_attachment = Mock()
        large_attachment.filename = "huge.pdf"
        large_attachment.payload = b"x" * (51 * 1024 * 1024)  # > 50MB
        
        msg = Mock()
        msg.uid = "12345"
        msg.attachments = [large_attachment]
        
        result = extract_pdf_attachments(msg, imap_config)
        
        assert len(result) == 0
    
    def test_handles_multiple_pdfs(self, imap_config):
        """Should extract all valid PDF attachments."""
        pdf1 = Mock()
        pdf1.filename = "contract1.pdf"
        pdf1.payload = b"%PDF" + b"x" * 2000
        
        pdf2 = Mock()
        pdf2.filename = "contract2.pdf"
        pdf2.payload = b"%PDF" + b"y" * 2000
        
        msg = Mock()
        msg.uid = "12345"
        msg.attachments = [pdf1, pdf2]
        
        result = extract_pdf_attachments(msg, imap_config)
        
        assert len(result) == 2
        
        # Cleanup
        for r in result:
            os.remove(r["path"])


# ============================================================================
# Test: build_email_metadata
# ============================================================================

class TestBuildEmailMetadata:
    """Tests for the build_email_metadata function."""
    
    def test_extracts_all_fields(self, mock_email_message):
        """Should extract all metadata fields."""
        result = build_email_metadata(mock_email_message)
        
        assert result["msg_id"] == "12345"
        assert result["subject"] == "Contract for 123 Main St"
        assert result["sender"] == "agent@realty.com"
        assert "2024-12-01" in result["date"]
    
    def test_handles_missing_subject(self):
        """Should handle None subject."""
        msg = Mock()
        msg.uid = "123"
        msg.subject = None
        msg.from_ = "test@test.com"
        msg.to = []
        msg.date = None
        msg.headers = {}
        
        result = build_email_metadata(msg)
        
        assert result["subject"] == "(no subject)"
    
    def test_handles_missing_date(self):
        """Should handle None date."""
        msg = Mock()
        msg.uid = "123"
        msg.subject = "Test"
        msg.from_ = "test@test.com"
        msg.to = []
        msg.date = None
        msg.headers = {}
        
        result = build_email_metadata(msg)
        
        assert result["date"] == ""


# ============================================================================
# Test: imap_listener_node (Integration)
# ============================================================================

class TestImapListenerNode:
    """Tests for the main imap_listener_node function."""
    
    @patch.dict(os.environ, {"USE_MOCK_IMAP": "true"})
    def test_mock_mode_returns_mock_data(self, initial_state):
        """Should return mock data when USE_MOCK_IMAP is true."""
        result = imap_listener_node(initial_state)
        
        assert result["status"] == "Processing"
        assert "email_metadata" in result
        assert result["email_metadata"]["subject"] == "Contract for 324 Muir Ln"
        assert "raw_pdf_path" in result
    
    @patch.dict(os.environ, {"USE_MOCK_IMAP": "false"}, clear=False)
    def test_returns_empty_when_no_credentials(self, initial_state):
        """Should return empty when no accounts are configured (L5 behavior)."""
        # Clear any existing credentials
        with patch.dict(os.environ, {
            "IMAP_USERNAME": "",
            "IMAP_PASSWORD": "",
            "USE_MOCK_IMAP": "false"
        }, clear=True):
            result = imap_listener_node(initial_state)
            
            # L5: Returns empty dict when no accounts configured
            assert result == {}
    
    @patch("nodes.imap_listener.fetch_unprocessed_emails")
    @patch("nodes.imap_listener.load_imap_accounts")
    @patch.dict(os.environ, {
        "USE_MOCK_IMAP": "false",
        "IMAP_USERNAME": "test@test.com",
        "IMAP_PASSWORD": "password",
    })
    def test_processes_real_email(self, mock_load_accounts, mock_fetch, initial_state):
        """Should process emails from fetch_unprocessed_emails."""
        # L5: Mock the accounts to ensure single-account path is used
        mock_load_accounts.return_value = [
            IMAPAccountConfig(
                account_id="primary",
                host="imap.test.com",
                port=993,
                username="test@test.com",
                password="password",
            )
        ]
        mock_fetch.return_value = [{
            "metadata": {
                "msg_id": "999",
                "subject": "Real Contract",
                "sender": "real@agent.com",
                "date": "2024-12-01",
                "account_id": "primary",
            },
            "attachments": [{"filename": "real.pdf", "path": "/tmp/real.pdf"}],
            "primary_pdf": "/tmp/real.pdf",
        }]
        
        result = imap_listener_node(initial_state)
        
        assert result["email_metadata"]["msg_id"] == "999"
        assert result["email_metadata"]["subject"] == "Real Contract"
        assert result["raw_pdf_path"] == "/tmp/real.pdf"
    
    @patch("nodes.imap_listener.fetch_unprocessed_emails")
    @patch.dict(os.environ, {
        "USE_MOCK_IMAP": "false",
        "IMAP_USERNAME": "test@test.com",
        "IMAP_PASSWORD": "password",
    })
    def test_returns_empty_when_no_emails(self, mock_fetch, initial_state):
        """Should return empty dict when no emails found."""
        mock_fetch.return_value = []
        
        result = imap_listener_node(initial_state)
        
        assert result == {}


# ============================================================================
# Test: fetch_unprocessed_emails (with mocked MailBox)
# ============================================================================

class TestFetchUnprocessedEmails:
    """Tests for the fetch_unprocessed_emails function."""
    
    @patch("nodes.imap_listener.MailBox")
    def test_connects_to_mailbox(self, mock_mailbox_class, imap_config):
        """Should connect to the IMAP server."""
        mock_mailbox = MagicMock()
        mock_mailbox_class.return_value.login.return_value.__enter__.return_value = mock_mailbox
        mock_mailbox.fetch.return_value = []
        
        fetch_unprocessed_emails(imap_config, sleep_func=lambda x: None)
        
        mock_mailbox_class.assert_called_once_with(imap_config.host, imap_config.port)
    
    @patch("nodes.imap_listener.MailBox")
    def test_fetches_unseen_emails(self, mock_mailbox_class, imap_config, mock_email_message):
        """Should fetch and process unseen emails with PDFs."""
        mock_mailbox = MagicMock()
        mock_mailbox_class.return_value.login.return_value.__enter__.return_value = mock_mailbox
        mock_mailbox.fetch.return_value = [mock_email_message]
        
        result = fetch_unprocessed_emails(imap_config, sleep_func=lambda x: None)
        
        assert len(result) == 1
        assert result[0]["metadata"]["subject"] == "Contract for 123 Main St"
    
    def test_returns_empty_without_credentials(self):
        """Should return empty list when credentials are invalid."""
        config = IMAPConfig()
        config.username = ""
        config.password = ""
        
        result = fetch_unprocessed_emails(config, sleep_func=lambda x: None)
        
        assert result == []
    
    @patch("nodes.imap_listener.MailBox")
    def test_handles_connection_error(self, mock_mailbox_class, imap_config):
        """Should handle connection errors gracefully."""
        imap_config.retry_delays = [0, 0, 0]  # No delays in tests
        mock_mailbox_class.return_value.login.side_effect = Exception("Connection failed")
        
        result = fetch_unprocessed_emails(imap_config, sleep_func=lambda x: None)
        
        assert result == []


# ============================================================================
# Test: extract_domain (L2)
# ============================================================================

class TestExtractDomain:
    """Tests for the extract_domain function."""
    
    def test_simple_email(self):
        """Should extract domain from simple email address."""
        assert extract_domain("agent@titlecompany.com") == "titlecompany.com"
    
    def test_email_with_name(self):
        """Should extract domain from 'Name <email>' format."""
        assert extract_domain("John Doe <john@escrow.com>") == "escrow.com"
    
    def test_email_with_quoted_name(self):
        """Should extract domain when name contains special chars."""
        assert extract_domain('"Doe, John" <john@escrow.com>') == "escrow.com"
    
    def test_uppercase_domain(self):
        """Should normalize domain to lowercase."""
        assert extract_domain("agent@TitleCompany.COM") == "titlecompany.com"
    
    def test_subdomain(self):
        """Should return full domain including subdomains."""
        assert extract_domain("agent@mail.titlecompany.com") == "mail.titlecompany.com"
    
    def test_empty_string(self):
        """Should return empty string for empty input."""
        assert extract_domain("") == ""
    
    def test_none_input(self):
        """Should return empty string for None input."""
        assert extract_domain(None) == ""
    
    def test_no_at_symbol(self):
        """Should return empty string if no @ symbol."""
        assert extract_domain("not-an-email") == ""


# ============================================================================
# Test: is_sender_allowed (L2)
# ============================================================================

class TestIsSenderAllowed:
    """Tests for the is_sender_allowed function with allowlist/blocklist."""
    
    def test_no_filtering_allows_all(self, imap_config):
        """Should allow all senders when no allowlist/blocklist configured."""
        imap_config.sender_allowlist = []
        imap_config.sender_blocklist = []
        
        assert is_sender_allowed("anyone@anydomain.com", imap_config) is True
        assert is_sender_allowed("spam@badsite.com", imap_config) is True
    
    def test_allowlist_allows_matching_domain(self, imap_config):
        """Should allow senders from allowlisted domains."""
        imap_config.sender_allowlist = ["titlecompany.com", "escrow.com"]
        imap_config.sender_blocklist = []
        
        assert is_sender_allowed("agent@titlecompany.com", imap_config) is True
        assert is_sender_allowed("support@escrow.com", imap_config) is True
    
    def test_allowlist_blocks_non_matching_domain(self, imap_config):
        """Should block senders not in allowlist."""
        imap_config.sender_allowlist = ["titlecompany.com"]
        imap_config.sender_blocklist = []
        
        assert is_sender_allowed("spam@random.com", imap_config) is False
        assert is_sender_allowed("agent@other-title.com", imap_config) is False
    
    def test_blocklist_blocks_matching_domain(self, imap_config):
        """Should block senders from blocklisted domains."""
        imap_config.sender_allowlist = []
        imap_config.sender_blocklist = ["spam.com", "marketing.com"]
        
        assert is_sender_allowed("promo@spam.com", imap_config) is False
        assert is_sender_allowed("newsletter@marketing.com", imap_config) is False
    
    def test_blocklist_allows_non_matching_domain(self, imap_config):
        """Should allow senders not in blocklist."""
        imap_config.sender_allowlist = []
        imap_config.sender_blocklist = ["spam.com"]
        
        assert is_sender_allowed("agent@titlecompany.com", imap_config) is True
        assert is_sender_allowed("client@gmail.com", imap_config) is True
    
    def test_allowlist_takes_precedence_over_blocklist(self, imap_config):
        """Allowlist should take precedence when both are set."""
        imap_config.sender_allowlist = ["titlecompany.com"]
        imap_config.sender_blocklist = ["spam.com"]
        
        # Only allowlist is checked when it's set
        assert is_sender_allowed("agent@titlecompany.com", imap_config) is True
        assert is_sender_allowed("promo@spam.com", imap_config) is False  # Not in allowlist
        assert is_sender_allowed("other@random.com", imap_config) is False  # Not in allowlist
    
    def test_handles_name_email_format(self, imap_config):
        """Should handle 'Name <email>' format correctly."""
        imap_config.sender_allowlist = ["titlecompany.com"]
        imap_config.sender_blocklist = []
        
        assert is_sender_allowed("John Agent <john@titlecompany.com>", imap_config) is True
        assert is_sender_allowed("Spam Bot <spam@random.com>", imap_config) is False
    
    def test_case_insensitive_matching(self, imap_config):
        """Should match domains case-insensitively."""
        imap_config.sender_allowlist = ["titlecompany.com"]
        imap_config.sender_blocklist = []
        
        assert is_sender_allowed("agent@TitleCompany.COM", imap_config) is True
        assert is_sender_allowed("agent@TITLECOMPANY.com", imap_config) is True
    
    def test_empty_sender_allowed_by_default(self, imap_config):
        """Should allow empty/None sender by default."""
        imap_config.sender_allowlist = ["titlecompany.com"]
        imap_config.sender_blocklist = []
        
        # Can't determine domain, allow by default
        assert is_sender_allowed("", imap_config) is True
        assert is_sender_allowed(None, imap_config) is True


# ============================================================================
# Test: IMAPConfig with sender filtering (L2)
# ============================================================================

class TestIMAPConfigSenderFiltering:
    """Tests for IMAPConfig sender allowlist/blocklist loading."""
    
    def test_loads_allowlist_from_env(self):
        """Should load sender allowlist from environment variable."""
        with patch.dict(os.environ, {"IMAP_SENDER_ALLOWLIST": "titleco.com, escrow.com, realty.com"}):
            config = IMAPConfig()
            assert config.sender_allowlist == ["titleco.com", "escrow.com", "realty.com"]
    
    def test_loads_blocklist_from_env(self):
        """Should load sender blocklist from environment variable."""
        with patch.dict(os.environ, {"IMAP_SENDER_BLOCKLIST": "spam.com, marketing.com"}):
            config = IMAPConfig()
            assert config.sender_blocklist == ["spam.com", "marketing.com"]
    
    def test_empty_env_creates_empty_lists(self):
        """Should create empty lists when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = IMAPConfig()
            assert config.sender_allowlist == []
            assert config.sender_blocklist == []
    
    def test_normalizes_domains_to_lowercase(self):
        """Should normalize domains to lowercase."""
        with patch.dict(os.environ, {"IMAP_SENDER_ALLOWLIST": "TitleCo.COM, ESCROW.Com"}):
            config = IMAPConfig()
            assert config.sender_allowlist == ["titleco.com", "escrow.com"]
    
    def test_handles_whitespace_in_list(self):
        """Should handle extra whitespace in domain lists."""
        with patch.dict(os.environ, {"IMAP_SENDER_ALLOWLIST": "  titleco.com ,  escrow.com  , realty.com  "}):
            config = IMAPConfig()
            assert config.sender_allowlist == ["titleco.com", "escrow.com", "realty.com"]


# ============================================================================
# Test: with_retry (L4 - Exponential Backoff)
# ============================================================================

class TestWithRetry:
    """Tests for the with_retry function implementing exponential backoff."""
    
    @pytest.fixture
    def retry_config(self):
        """Create a config with fast retry delays for testing."""
        config = IMAPConfig()
        config.retry_delays = [0, 0, 0]  # No delays in tests
        config.max_retries = 3
        return config
    
    def test_succeeds_on_first_attempt(self, retry_config):
        """Should return success on first attempt without retries."""
        operation = Mock(return_value="success")
        
        result = with_retry(operation, retry_config, "test op", sleep_func=lambda x: None)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1
        assert result.last_error is None
        operation.assert_called_once()
    
    def test_retries_on_failure_then_succeeds(self, retry_config):
        """Should retry on failure and return success when operation eventually succeeds."""
        operation = Mock(side_effect=[Exception("fail 1"), Exception("fail 2"), "success"])
        
        result = with_retry(operation, retry_config, "test op", sleep_func=lambda x: None)
        
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 3
        assert operation.call_count == 3
    
    def test_fails_after_max_retries(self, retry_config):
        """Should fail permanently after exhausting all retries."""
        operation = Mock(side_effect=Exception("always fails"))
        
        result = with_retry(operation, retry_config, "test op", sleep_func=lambda x: None)
        
        assert result.success is False
        assert result.result is None
        assert result.attempts == 4  # 1 initial + 3 retries
        assert result.last_error == "always fails"
        assert operation.call_count == 4
    
    def test_uses_configured_delays(self, retry_config):
        """Should sleep for configured delay times between retries."""
        retry_config.retry_delays = [30, 60, 300]
        operation = Mock(side_effect=[Exception("fail"), Exception("fail"), Exception("fail"), "success"])
        sleep_calls = []
        
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
        
        result = with_retry(operation, retry_config, "test op", sleep_func=mock_sleep)
        
        assert result.success is True
        assert sleep_calls == [30, 60, 300]
    
    def test_does_not_sleep_on_first_attempt(self, retry_config):
        """Should not sleep before the first attempt."""
        operation = Mock(return_value="success")
        sleep_calls = []
        
        with_retry(operation, retry_config, "test op", sleep_func=lambda x: sleep_calls.append(x))
        
        assert sleep_calls == []
    
    def test_captures_last_error_message(self, retry_config):
        """Should capture the error message from the last failed attempt."""
        retry_config.retry_delays = [0]
        retry_config.max_retries = 1
        operation = Mock(side_effect=[Exception("first error"), Exception("second error")])
        
        result = with_retry(operation, retry_config, "test op", sleep_func=lambda x: None)
        
        assert result.last_error == "second error"


class TestRetryResult:
    """Tests for the RetryResult dataclass."""
    
    def test_success_result(self):
        """Should create a success result."""
        result = RetryResult(success=True, result="data", attempts=1)
        
        assert result.success is True
        assert result.result == "data"
        assert result.attempts == 1
        assert result.last_error is None
    
    def test_failure_result(self):
        """Should create a failure result with error."""
        result = RetryResult(success=False, result=None, attempts=4, last_error="Connection timeout")
        
        assert result.success is False
        assert result.result is None
        assert result.attempts == 4
        assert result.last_error == "Connection timeout"


class TestIMAPConfigRetry:
    """Tests for IMAPConfig retry configuration (L4)."""
    
    def test_default_retry_delays(self):
        """Should have default retry delays of 30s, 60s, 300s."""
        config = IMAPConfig()
        assert config.retry_delays == [30, 60, 300]
        assert config.max_retries == 3
    
    def test_loads_retry_delays_from_env(self):
        """Should load retry delays from environment variable."""
        with patch.dict(os.environ, {"IMAP_RETRY_DELAYS": "10,20,30,40"}):
            config = IMAPConfig()
            assert config.retry_delays == [10, 20, 30, 40]
            assert config.max_retries == 4
    
    def test_single_retry_delay(self):
        """Should handle single retry delay."""
        with patch.dict(os.environ, {"IMAP_RETRY_DELAYS": "60"}):
            config = IMAPConfig()
            assert config.retry_delays == [60]
            assert config.max_retries == 1


class TestFetchUnprocessedEmailsWithRetry:
    """Tests for fetch_unprocessed_emails with retry logic (L4)."""
    
    @patch("nodes.imap_listener.MailBox")
    def test_retries_on_connection_failure(self, mock_mailbox_class, imap_config):
        """Should retry when IMAP connection fails."""
        imap_config.retry_delays = [0, 0]  # Fast retries for testing
        imap_config.max_retries = 2
        
        # Fail twice, then succeed
        mock_mailbox = MagicMock()
        mock_mailbox.fetch.return_value = []
        mock_mailbox_class.return_value.login.side_effect = [
            Exception("Connection refused"),
            Exception("Timeout"),
            MagicMock(__enter__=Mock(return_value=mock_mailbox), __exit__=Mock(return_value=False)),
        ]
        
        result = fetch_unprocessed_emails(imap_config, sleep_func=lambda x: None)
        
        # Should have retried and eventually succeeded
        assert mock_mailbox_class.return_value.login.call_count == 3
    
    @patch("nodes.imap_listener.MailBox")
    def test_returns_empty_after_all_retries_fail(self, mock_mailbox_class, imap_config):
        """Should return empty list after all retries are exhausted."""
        imap_config.retry_delays = [0]
        imap_config.max_retries = 1
        
        mock_mailbox_class.return_value.login.side_effect = Exception("Permanent failure")
        
        result = fetch_unprocessed_emails(imap_config, sleep_func=lambda x: None)
        
        assert result == []
        assert mock_mailbox_class.return_value.login.call_count == 2  # 1 initial + 1 retry


# ============================================================================
# Test: IMAPAccountConfig (L5 - Multi-Account)
# ============================================================================

class TestIMAPAccountConfig:
    """Tests for the IMAPAccountConfig dataclass (L5)."""
    
    def test_create_from_dict(self):
        """Should create account config from dictionary."""
        data = {
            "account_id": "office-main",
            "host": "imap.office.com",
            "port": 993,
            "username": "office@company.com",
            "password": "secret123",
            "inbox_folder": "Contracts",
            "sender_allowlist": "titleco.com,escrow.com",
        }
        
        account = IMAPAccountConfig.from_dict(data)
        
        assert account.account_id == "office-main"
        assert account.host == "imap.office.com"
        assert account.port == 993
        assert account.username == "office@company.com"
        assert account.password == "secret123"
        assert account.inbox_folder == "Contracts"
        assert account.sender_allowlist == ["titleco.com", "escrow.com"]
    
    def test_create_from_dict_with_defaults(self):
        """Should use defaults for missing optional fields."""
        data = {
            "username": "test@test.com",
            "password": "pass",
        }
        
        account = IMAPAccountConfig.from_dict(data)
        
        assert account.account_id == "test@test.com"  # Defaults to username
        assert account.host == "imap.gmail.com"
        assert account.port == 993
        assert account.inbox_folder == "INBOX"
        assert account.enabled is True
    
    def test_is_valid_with_credentials(self):
        """Should return True when account has valid credentials."""
        account = IMAPAccountConfig(
            account_id="test",
            host="imap.test.com",
            port=993,
            username="user@test.com",
            password="pass",
        )
        
        assert account.is_valid() is True
    
    def test_is_valid_when_disabled(self):
        """Should return False when account is disabled."""
        account = IMAPAccountConfig(
            account_id="test",
            host="imap.test.com",
            port=993,
            username="user@test.com",
            password="pass",
            enabled=False,
        )
        
        assert account.is_valid() is False
    
    def test_is_valid_without_password(self):
        """Should return False when password is missing."""
        account = IMAPAccountConfig(
            account_id="test",
            host="imap.test.com",
            port=993,
            username="user@test.com",
            password="",
        )
        
        assert account.is_valid() is False
    
    def test_sender_allowlist_as_list(self):
        """Should handle sender_allowlist provided as list."""
        data = {
            "username": "test@test.com",
            "password": "pass",
            "sender_allowlist": ["domain1.com", "domain2.com"],
        }
        
        account = IMAPAccountConfig.from_dict(data)
        
        assert account.sender_allowlist == ["domain1.com", "domain2.com"]


class TestLoadIMAPAccounts:
    """Tests for load_imap_accounts function (L5)."""
    
    def test_loads_from_json_env_var(self):
        """Should load accounts from IMAP_ACCOUNTS_JSON environment variable."""
        accounts_json = json.dumps([
            {
                "account_id": "account1",
                "host": "imap.test1.com",
                "username": "user1@test.com",
                "password": "pass1",
            },
            {
                "account_id": "account2",
                "host": "imap.test2.com",
                "username": "user2@test.com",
                "password": "pass2",
            },
        ])
        
        with patch.dict(os.environ, {"IMAP_ACCOUNTS_JSON": accounts_json}, clear=True):
            accounts = load_imap_accounts()
        
        assert len(accounts) == 2
        assert accounts[0].account_id == "account1"
        assert accounts[1].account_id == "account2"
    
    def test_loads_from_json_file(self, tmp_path):
        """Should load accounts from IMAP_ACCOUNTS_FILE."""
        accounts_data = [
            {
                "account_id": "file-account",
                "host": "imap.file.com",
                "username": "file@test.com",
                "password": "filepass",
            }
        ]
        
        accounts_file = tmp_path / "accounts.json"
        accounts_file.write_text(json.dumps(accounts_data))
        
        with patch.dict(os.environ, {"IMAP_ACCOUNTS_FILE": str(accounts_file)}, clear=True):
            accounts = load_imap_accounts()
        
        assert len(accounts) == 1
        assert accounts[0].account_id == "file-account"
    
    def test_falls_back_to_legacy_env_vars(self):
        """Should use legacy IMAP_* env vars when no accounts configured."""
        legacy_env = {
            "IMAP_HOST": "imap.legacy.com",
            "IMAP_USERNAME": "legacy@test.com",
            "IMAP_PASSWORD": "legacypass",
        }
        
        with patch.dict(os.environ, legacy_env, clear=True):
            accounts = load_imap_accounts()
        
        assert len(accounts) == 1
        assert accounts[0].account_id == "primary"
        assert accounts[0].username == "legacy@test.com"
    
    def test_skips_invalid_accounts(self):
        """Should skip accounts with missing credentials."""
        accounts_json = json.dumps([
            {
                "account_id": "valid",
                "host": "imap.test.com",
                "username": "valid@test.com",
                "password": "pass",
            },
            {
                "account_id": "invalid",
                "host": "imap.test.com",
                "username": "invalid@test.com",
                "password": "",  # Missing password
            },
        ])
        
        with patch.dict(os.environ, {"IMAP_ACCOUNTS_JSON": accounts_json}, clear=True):
            accounts = load_imap_accounts()
        
        assert len(accounts) == 1
        assert accounts[0].account_id == "valid"
    
    def test_returns_empty_when_no_config(self):
        """Should return empty list when no configuration exists."""
        with patch.dict(os.environ, {}, clear=True):
            accounts = load_imap_accounts()
        
        assert accounts == []


class TestIsSenderAllowedForAccount:
    """Tests for is_sender_allowed_for_account function (L5)."""
    
    def test_uses_account_specific_allowlist(self):
        """Should use the account's own allowlist."""
        account = IMAPAccountConfig(
            account_id="test",
            host="imap.test.com",
            port=993,
            username="test@test.com",
            password="pass",
            sender_allowlist=["trusted.com"],
        )
        
        assert is_sender_allowed_for_account("agent@trusted.com", account) is True
        assert is_sender_allowed_for_account("spam@other.com", account) is False
    
    def test_uses_account_specific_blocklist(self):
        """Should use the account's own blocklist."""
        account = IMAPAccountConfig(
            account_id="test",
            host="imap.test.com",
            port=993,
            username="test@test.com",
            password="pass",
            sender_blocklist=["spam.com"],
        )
        
        assert is_sender_allowed_for_account("promo@spam.com", account) is False
        assert is_sender_allowed_for_account("agent@trusted.com", account) is True


class TestBuildEmailMetadataWithAccountId:
    """Tests for build_email_metadata with account_id (L5)."""
    
    def test_includes_account_id(self, mock_email_message):
        """Should include account_id in metadata."""
        metadata = build_email_metadata(mock_email_message, "office-account")
        
        assert metadata["account_id"] == "office-account"
    
    def test_default_account_id(self, mock_email_message):
        """Should use 'primary' as default account_id."""
        metadata = build_email_metadata(mock_email_message)
        
        assert metadata["account_id"] == "primary"


class TestFetchEmailsFromAllAccounts:
    """Tests for fetch_emails_from_all_accounts function (L5)."""
    
    @patch("nodes.imap_listener.load_imap_accounts")
    @patch("nodes.imap_listener.MailBox")
    def test_fetches_from_multiple_accounts(self, mock_mailbox_class, mock_load_accounts, mock_email_message):
        """Should fetch emails from all configured accounts."""
        # Setup two accounts
        mock_load_accounts.return_value = [
            IMAPAccountConfig(
                account_id="account1",
                host="imap.test1.com",
                port=993,
                username="user1@test.com",
                password="pass1",
            ),
            IMAPAccountConfig(
                account_id="account2",
                host="imap.test2.com",
                port=993,
                username="user2@test.com",
                password="pass2",
            ),
        ]
        
        # Mock successful connections
        mock_mailbox = MagicMock()
        mock_mailbox_class.return_value.login.return_value.__enter__.return_value = mock_mailbox
        mock_mailbox.fetch.return_value = [mock_email_message]
        
        result = fetch_emails_from_all_accounts(sleep_func=lambda x: None)
        
        # Should have fetched from both accounts
        assert mock_mailbox_class.call_count == 2
    
    @patch("nodes.imap_listener.load_imap_accounts")
    def test_returns_empty_when_no_accounts(self, mock_load_accounts):
        """Should return empty list when no accounts configured."""
        mock_load_accounts.return_value = []
        
        result = fetch_emails_from_all_accounts(sleep_func=lambda x: None)
        
        assert result == []


class TestIMAPConfigToAccountConfig:
    """Tests for IMAPConfig.to_account_config() method (L5)."""
    
    def test_converts_to_account_config(self):
        """Should convert legacy IMAPConfig to IMAPAccountConfig."""
        with patch.dict(os.environ, {
            "IMAP_HOST": "imap.test.com",
            "IMAP_USERNAME": "user@test.com",
            "IMAP_PASSWORD": "pass",
            "IMAP_SENDER_ALLOWLIST": "trusted.com,other.com",
        }):
            config = IMAPConfig()
            account = config.to_account_config("my-account")
        
        assert account.account_id == "my-account"
        assert account.host == "imap.test.com"
        assert account.username == "user@test.com"
        assert account.password == "pass"
        assert account.sender_allowlist == ["trusted.com", "other.com"]


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
