"""
IMAP Listener Node - Email Monitoring & Attachment Extraction

User Story L1:
As a real estate agent, I want the system to automatically detect new emails 
with contract attachments so I don't have to manually upload documents.

Acceptance Criteria:
- System connects to IMAP server
- Filters emails by attachment type (PDF)
- Downloads PDF attachments to temp storage
- Marks emails as processing (flags/moves to folder)

User Story L2:
As a broker, I want emails filtered by sender domain (e.g., only from trusted 
title companies) so spam doesn't enter the pipeline.

Acceptance Criteria:
- Configurable allowlist for trusted sender domains
- Configurable blocklist for spam/unwanted domains
- Allowlist takes precedence (if set, only those domains allowed)
- Blocklist filters out specific domains when allowlist is empty

User Story L4:
As an admin, I want failed email fetches to retry with exponential backoff 
so temporary IMAP issues don't break the system.

Acceptance Criteria:
- 3 retries with 30s/1m/5m delays before marking failed
- Configurable retry delays
- Clear error logging for each retry attempt

User Story L5:
As a user, I want to connect multiple email accounts so contracts from 
different inboxes are processed.

Acceptance Criteria:
- Support for multiple IMAP credentials per user/organization
- Each account can have its own allowlist/blocklist
- Emails from all accounts are aggregated for processing
- Account identifier included in email metadata
"""

import os
import time
import json
import tempfile
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from imap_tools.mailbox import MailBox
from imap_tools.query import AND
from imap_tools.message import MailMessage
from state import DealState


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IMAPAccountConfig:
    """
    Configuration for a single IMAP account (L5).
    
    Can be created from:
    - Direct instantiation with parameters
    - JSON configuration
    - Environment variables (legacy single-account mode)
    """
    account_id: str
    host: str
    port: int
    username: str
    password: str
    inbox_folder: str = "INBOX"
    processed_folder: str = "Processed"
    sender_allowlist: List[str] = field(default_factory=list)
    sender_blocklist: List[str] = field(default_factory=list)
    enabled: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IMAPAccountConfig":
        """Create an account config from a dictionary."""
        return cls(
            account_id=data.get("account_id", data.get("username", "unknown")),
            host=data.get("host", "imap.gmail.com"),
            port=int(data.get("port", 993)),
            username=data.get("username", ""),
            password=data.get("password", ""),
            inbox_folder=data.get("inbox_folder", "INBOX"),
            processed_folder=data.get("processed_folder", "Processed"),
            sender_allowlist=[
                d.strip().lower() 
                for d in data.get("sender_allowlist", "").split(",") 
                if d.strip()
            ] if isinstance(data.get("sender_allowlist"), str) else data.get("sender_allowlist", []),
            sender_blocklist=[
                d.strip().lower() 
                for d in data.get("sender_blocklist", "").split(",") 
                if d.strip()
            ] if isinstance(data.get("sender_blocklist"), str) else data.get("sender_blocklist", []),
            enabled=data.get("enabled", True),
        )
    
    def is_valid(self) -> bool:
        """Check if account has required credentials."""
        return bool(self.username and self.password and self.host and self.enabled)


class IMAPConfig:
    """IMAP connection configuration from environment variables."""
    
    def __init__(self):
        self.host = os.getenv("IMAP_HOST", "imap.gmail.com")
        self.port = int(os.getenv("IMAP_PORT", "993"))
        self.username = os.getenv("IMAP_USERNAME", "")
        self.password = os.getenv("IMAP_PASSWORD", "")  # App password for Gmail
        self.inbox_folder = os.getenv("IMAP_INBOX", "INBOX")
        self.processed_folder = os.getenv("IMAP_PROCESSED_FOLDER", "Processed")
        self.attachment_dir = os.getenv("ATTACHMENT_DIR", tempfile.gettempdir())
        
        # Filtering options
        self.allowed_extensions = [".pdf"]
        self.min_attachment_size = 1024  # 1KB minimum (skip tiny/empty files)
        self.max_attachment_size = 50 * 1024 * 1024  # 50MB maximum
        
        # Sender domain filtering (L2)
        # Comma-separated list of allowed domains (if set, ONLY these are allowed)
        # Example: "titlecompany.com,escrow.com,realtor.com"
        allowlist_str = os.getenv("IMAP_SENDER_ALLOWLIST", "")
        self.sender_allowlist: List[str] = [
            d.strip().lower() for d in allowlist_str.split(",") if d.strip()
        ]
        
        # Comma-separated list of blocked domains (ignored if allowlist is set)
        # Example: "spam.com,marketing.com,newsletter.com"
        blocklist_str = os.getenv("IMAP_SENDER_BLOCKLIST", "")
        self.sender_blocklist: List[str] = [
            d.strip().lower() for d in blocklist_str.split(",") if d.strip()
        ]
        
        # Retry configuration (L4)
        # Comma-separated delays in seconds: "30,60,300" = 30s, 1m, 5m
        retry_delays_str = os.getenv("IMAP_RETRY_DELAYS", "30,60,300")
        self.retry_delays: List[int] = [
            int(d.strip()) for d in retry_delays_str.split(",") if d.strip()
        ]
        self.max_retries = len(self.retry_delays)
    
    def is_valid(self) -> bool:
        """Check if required credentials are configured."""
        return bool(self.username and self.password and self.host)
    
    def to_account_config(self, account_id: str = "primary") -> IMAPAccountConfig:
        """Convert legacy IMAPConfig to IMAPAccountConfig."""
        return IMAPAccountConfig(
            account_id=account_id,
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            inbox_folder=self.inbox_folder,
            processed_folder=self.processed_folder,
            sender_allowlist=self.sender_allowlist,
            sender_blocklist=self.sender_blocklist,
            enabled=True,
        )


def load_imap_accounts() -> List[IMAPAccountConfig]:
    """
    Load IMAP account configurations (L5).
    
    Supports multiple configuration methods:
    1. IMAP_ACCOUNTS_JSON env var: JSON array of account configs
    2. IMAP_ACCOUNTS_FILE env var: Path to JSON file with accounts
    3. Legacy single-account env vars (IMAP_HOST, IMAP_USERNAME, etc.)
    
    Returns:
        List of valid, enabled IMAPAccountConfig objects
    
    Example JSON format:
    [
        {
            "account_id": "office-main",
            "host": "imap.gmail.com",
            "port": 993,
            "username": "office@company.com",
            "password": "app-password-1",
            "sender_allowlist": "titleco.com,escrow.com"
        },
        {
            "account_id": "agent-personal",
            "host": "imap.gmail.com",
            "port": 993,
            "username": "agent@gmail.com",
            "password": "app-password-2",
            "sender_blocklist": "spam.com,marketing.com"
        }
    ]
    """
    accounts: List[IMAPAccountConfig] = []
    
    # Method 1: JSON string in environment variable
    accounts_json = os.getenv("IMAP_ACCOUNTS_JSON", "")
    if accounts_json:
        try:
            accounts_data = json.loads(accounts_json)
            if isinstance(accounts_data, list):
                for acc_data in accounts_data:
                    account = IMAPAccountConfig.from_dict(acc_data)
                    if account.is_valid():
                        accounts.append(account)
                        print(f"   Loaded account: {account.account_id} ({account.username})")
                    else:
                        print(f"   Skipped invalid account: {acc_data.get('account_id', 'unknown')}")
        except json.JSONDecodeError as e:
            print(f"   ERROR: Failed to parse IMAP_ACCOUNTS_JSON: {e}")
    
    # Method 2: JSON file path
    accounts_file = os.getenv("IMAP_ACCOUNTS_FILE", "")
    if accounts_file and os.path.exists(accounts_file):
        try:
            with open(accounts_file, "r") as f:
                accounts_data = json.load(f)
            if isinstance(accounts_data, list):
                for acc_data in accounts_data:
                    account = IMAPAccountConfig.from_dict(acc_data)
                    if account.is_valid():
                        accounts.append(account)
                        print(f"   Loaded account from file: {account.account_id}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"   ERROR: Failed to load accounts from {accounts_file}: {e}")
    
    # Method 3: Legacy single-account environment variables
    if not accounts:
        legacy_config = IMAPConfig()
        if legacy_config.is_valid():
            accounts.append(legacy_config.to_account_config("primary"))
            print(f"   Loaded legacy account: {legacy_config.username}")
    
    return accounts


def get_account_count() -> int:
    """Get the number of configured IMAP accounts."""
    return len(load_imap_accounts())


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any
    attempts: int
    last_error: Optional[str] = None


def with_retry(
    operation: Callable[[], Any],
    config: IMAPConfig,
    operation_name: str = "operation",
    sleep_func: Callable[[int], None] = time.sleep,
) -> RetryResult:
    """
    Execute an operation with exponential backoff retry.
    
    L4: Implements 3 retries with 30s/1m/5m delays before marking failed.
    
    Args:
        operation: Callable that performs the operation (should raise on failure)
        config: IMAPConfig with retry_delays configured
        operation_name: Name for logging purposes
        sleep_func: Sleep function (injectable for testing)
    
    Returns:
        RetryResult with success status, result, attempts count, and last error
    """
    last_error: Optional[str] = None
    
    for attempt in range(config.max_retries + 1):
        try:
            result = operation()
            if attempt > 0:
                print(f"   {operation_name} succeeded on attempt {attempt + 1}")
            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1,
                last_error=None
            )
        except Exception as e:
            last_error = str(e)
            
            if attempt < config.max_retries:
                delay = config.retry_delays[attempt]
                print(f"   {operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}")
                print(f"   Retrying in {delay} seconds...")
                sleep_func(delay)
            else:
                print(f"   {operation_name} failed after {attempt + 1} attempts: {e}")
    
    return RetryResult(
        success=False,
        result=None,
        attempts=config.max_retries + 1,
        last_error=last_error
    )


# ============================================================================
# Email Processing Functions
# ============================================================================

def extract_domain(email_address: Optional[str]) -> str:
    """
    Extract the domain from an email address.
    
    Examples:
        "john@titlecompany.com" -> "titlecompany.com"
        "Jane Doe <jane@escrow.com>" -> "escrow.com"
    """
    if not email_address:
        return ""
    
    # Handle "Name <email@domain.com>" format
    if "<" in email_address and ">" in email_address:
        start = email_address.rfind("<") + 1
        end = email_address.rfind(">")
        email_address = email_address[start:end]
    
    # Extract domain part
    if "@" in email_address:
        return email_address.split("@")[-1].strip().lower()
    
    return ""


def is_sender_allowed(sender: Optional[str], config: IMAPConfig) -> bool:
    """
    Check if a sender's domain is allowed based on allowlist/blocklist config.
    
    Logic:
    1. If allowlist is set, sender domain MUST be in allowlist
    2. If allowlist is empty, sender domain must NOT be in blocklist
    3. If both are empty, all senders are allowed
    
    Returns:
        True if sender is allowed, False if filtered out
    """
    domain = extract_domain(sender)
    
    if not domain:
        # Can't determine domain - allow by default (could be made configurable)
        return True
    
    # Allowlist takes precedence
    if config.sender_allowlist:
        is_allowed = domain in config.sender_allowlist
        if not is_allowed:
            print(f"   Filtered out (not in allowlist): {sender}")
        return is_allowed
    
    # Check blocklist
    if config.sender_blocklist:
        is_blocked = domain in config.sender_blocklist
        if is_blocked:
            print(f"   Filtered out (in blocklist): {sender}")
        return not is_blocked
    
    # No filtering configured - allow all
    return True


def is_sender_allowed_for_account(sender: Optional[str], account: IMAPAccountConfig) -> bool:
    """
    Check if sender is allowed for a specific account (L5).
    
    Uses account-specific allowlist/blocklist instead of global config.
    """
    domain = extract_domain(sender)
    
    if not domain:
        return True
    
    # Allowlist takes precedence
    if account.sender_allowlist:
        is_allowed = domain in account.sender_allowlist
        if not is_allowed:
            print(f"   [{account.account_id}] Filtered out (not in allowlist): {sender}")
        return is_allowed
    
    # Check blocklist
    if account.sender_blocklist:
        is_blocked = domain in account.sender_blocklist
        if is_blocked:
            print(f"   [{account.account_id}] Filtered out (in blocklist): {sender}")
        return not is_blocked
    
    return True


def is_pdf_attachment(filename: Optional[str]) -> bool:
    """Check if a filename is a PDF."""
    if not filename:
        return False
    return Path(filename).suffix.lower() == ".pdf"


def extract_pdf_attachments(
    msg: MailMessage, 
    config: IMAPConfig
) -> List[Dict[str, Any]]:
    """
    Extract PDF attachments from an email message.
    
    Returns list of dicts with:
    - filename: Original filename
    - path: Local path where file was saved
    - size: File size in bytes
    """
    extracted = []
    
    for att in msg.attachments:
        # Check if it's a PDF
        if not is_pdf_attachment(att.filename):
            continue
        
        # Check size constraints
        if len(att.payload) < config.min_attachment_size:
            print(f"   Skipping {att.filename}: too small ({len(att.payload)} bytes)")
            continue
        
        if len(att.payload) > config.max_attachment_size:
            print(f"   Skipping {att.filename}: too large ({len(att.payload)} bytes)")
            continue
        
        # Create safe filename
        safe_filename = att.filename.replace("/", "_").replace("\\", "_")
        
        # Add unique prefix to avoid collisions
        unique_filename = f"{msg.uid}_{safe_filename}"
        file_path = os.path.join(config.attachment_dir, unique_filename)
        
        # Save the attachment
        with open(file_path, "wb") as f:
            f.write(att.payload)
        
        extracted.append({
            "filename": att.filename,
            "path": file_path,
            "size": len(att.payload),
        })
        
        print(f"   Saved attachment: {att.filename} ({len(att.payload):,} bytes)")
    
    return extracted


def build_email_metadata(msg: MailMessage, account_id: str = "primary") -> Dict[str, str]:
    """
    Extract relevant metadata from an email message.
    
    L5: Includes account_id to identify which account the email came from.
    """
    return {
        "msg_id": str(msg.uid),
        "message_id": msg.headers.get("message-id", [""])[0],
        "subject": msg.subject or "(no subject)",
        "sender": msg.from_ or "",
        "date": msg.date.isoformat() if msg.date else "",
        "to": ", ".join(msg.to) if msg.to else "",
        "account_id": account_id,  # L5: Track source account
    }


def _fetch_emails_from_mailbox(config: IMAPConfig) -> List[Dict[str, Any]]:
    """
    Core IMAP fetch operation (without retry logic).
    
    This is the inner function that actually connects to IMAP and fetches emails.
    Raises exceptions on connection/auth failures for retry handling.
    """
    emails_to_process = []
    
    # Connect to mailbox - this can raise exceptions
    with MailBox(config.host, config.port).login(
        config.username, 
        config.password,
        initial_folder=config.inbox_folder
    ) as mailbox:
        
        print(f"   Connected to {config.host}")
        print(f"   Checking folder: {config.inbox_folder}")
        
        # Fetch unseen emails (not yet read)
        for msg in mailbox.fetch(AND(seen=False), limit=10):
            
            # L2: Filter by sender domain (allowlist/blocklist)
            if not is_sender_allowed(msg.from_, config):
                continue
            
            # Check if email has PDF attachments
            pdf_attachments = [
                att for att in msg.attachments 
                if is_pdf_attachment(att.filename)
            ]
            
            if not pdf_attachments:
                continue
            
            print(f"   Found email: '{msg.subject}' from {msg.from_} with {len(pdf_attachments)} PDF(s)")
            
            # Extract attachments
            saved_attachments = extract_pdf_attachments(msg, config)
            
            if saved_attachments:
                email_data = {
                    "metadata": build_email_metadata(msg),
                    "attachments": saved_attachments,
                    "primary_pdf": saved_attachments[0]["path"],
                }
                emails_to_process.append(email_data)
                
                # Mark as seen (processing started)
                if msg.uid:
                    mailbox.flag(msg.uid, ["\\Seen"], True)
                
                # Only process one email per run for now
                break
        
        if not emails_to_process:
            print("   No new emails with PDF attachments found")
    
    return emails_to_process


def fetch_unprocessed_emails(
    config: IMAPConfig,
    sleep_func: Callable[[int], None] = time.sleep,
) -> List[Dict[str, Any]]:
    """
    Connect to IMAP and fetch unprocessed emails with PDF attachments.
    
    L4: Implements exponential backoff retry (30s/1m/5m by default).
    
    Args:
        config: IMAP configuration
        sleep_func: Sleep function (injectable for testing)
    
    Returns:
        List of email data dicts ready for processing.
    """
    if not config.is_valid():
        print("   ERROR: IMAP credentials not configured")
        print("   Set IMAP_HOST, IMAP_USERNAME, IMAP_PASSWORD environment variables")
        return []
    
    # L4: Use retry wrapper for resilient IMAP connection
    retry_result = with_retry(
        operation=lambda: _fetch_emails_from_mailbox(config),
        config=config,
        operation_name="IMAP fetch",
        sleep_func=sleep_func,
    )
    
    if retry_result.success:
        return retry_result.result
    else:
        print(f"   IMAP fetch failed permanently after {retry_result.attempts} attempts")
        print(f"   Last error: {retry_result.last_error}")
        return []


def _fetch_emails_from_account(
    account: IMAPAccountConfig, 
    global_config: IMAPConfig
) -> List[Dict[str, Any]]:
    """
    Fetch emails from a specific IMAP account (L5).
    
    This function connects to a specific account and fetches emails,
    using account-specific sender filtering.
    """
    emails_to_process = []
    
    with MailBox(account.host, account.port).login(
        account.username,
        account.password,
        initial_folder=account.inbox_folder
    ) as mailbox:
        
        print(f"   [{account.account_id}] Connected to {account.host}")
        print(f"   [{account.account_id}] Checking folder: {account.inbox_folder}")
        
        for msg in mailbox.fetch(AND(seen=False), limit=10):
            
            # Use account-specific sender filtering
            if not is_sender_allowed_for_account(msg.from_, account):
                continue
            
            pdf_attachments = [
                att for att in msg.attachments
                if is_pdf_attachment(att.filename)
            ]
            
            if not pdf_attachments:
                continue
            
            print(f"   [{account.account_id}] Found email: '{msg.subject}' with {len(pdf_attachments)} PDF(s)")
            
            # Create a temporary config for attachment extraction
            temp_config = IMAPConfig()
            temp_config.min_attachment_size = global_config.min_attachment_size
            temp_config.max_attachment_size = global_config.max_attachment_size
            temp_config.attachment_dir = global_config.attachment_dir
            
            saved_attachments = extract_pdf_attachments(msg, temp_config)
            
            if saved_attachments:
                email_data = {
                    "metadata": build_email_metadata(msg, account.account_id),
                    "attachments": saved_attachments,
                    "primary_pdf": saved_attachments[0]["path"],
                    "account_id": account.account_id,
                }
                emails_to_process.append(email_data)
                
                if msg.uid:
                    mailbox.flag(msg.uid, ["\\Seen"], True)
                
                # Process one email per account per run
                break
        
        if not emails_to_process:
            print(f"   [{account.account_id}] No new emails with PDF attachments")
    
    return emails_to_process


def fetch_emails_from_all_accounts(
    sleep_func: Callable[[int], None] = time.sleep,
) -> List[Dict[str, Any]]:
    """
    Fetch emails from all configured IMAP accounts (L5).
    
    Iterates through all configured accounts and aggregates emails.
    Each account is processed with its own retry logic.
    
    Returns:
        List of email data dicts from all accounts, with account_id included.
    """
    accounts = load_imap_accounts()
    global_config = IMAPConfig()
    
    if not accounts:
        print("   No IMAP accounts configured")
        return []
    
    print(f"   Processing {len(accounts)} IMAP account(s)")
    
    all_emails: List[Dict[str, Any]] = []
    
    for account in accounts:
        if not account.is_valid():
            print(f"   [{account.account_id}] Skipped: invalid configuration")
            continue
        
        # Use retry wrapper for each account
        retry_result = with_retry(
            operation=lambda acc=account: _fetch_emails_from_account(acc, global_config),
            config=global_config,
            operation_name=f"IMAP fetch [{account.account_id}]",
            sleep_func=sleep_func,
        )
        
        if retry_result.success and retry_result.result:
            all_emails.extend(retry_result.result)
        elif not retry_result.success:
            print(f"   [{account.account_id}] Failed after {retry_result.attempts} attempts")
    
    print(f"   Total emails found across all accounts: {len(all_emails)}")
    return all_emails


# ============================================================================
# Main Node Function
# ============================================================================

def imap_listener_node(state: DealState) -> dict:
    """
    Node A: IMAP Email Listener
    
    Connects to IMAP server(s), finds new emails with PDF attachments,
    downloads PDFs, and returns email metadata for processing.
    
    User Story L1: Automatically detect new emails with contract attachments
    User Story L5: Support multiple IMAP accounts
    """
    print("--- NODE: IMAP Listener ---")
    
    # Check if we should use mock mode (for testing without IMAP)
    use_mock = os.getenv("USE_MOCK_IMAP", "true").lower() == "true"
    
    if use_mock:
        # Mock mode for development/testing
        print("   Running in MOCK mode (set USE_MOCK_IMAP=false for real IMAP)")
        return {
            "email_metadata": {
                "msg_id": "mock-001",
                "subject": "Contract for 324 Muir Ln",
                "sender": "peter@example.com",
                "date": "2024-12-01T10:30:00",
                "account_id": "mock-account",
            },
            "raw_pdf_path": "/tmp/2037_auction-5-Buy-Sell-Agreement.pdf",
            "status": "Processing"
        }
    
    # L5: Check for multi-account configuration
    accounts = load_imap_accounts()
    
    if not accounts:
        print("   No IMAP accounts configured")
        print("   Set IMAP_ACCOUNTS_JSON, IMAP_ACCOUNTS_FILE, or legacy IMAP_* env vars")
        return {}
    
    # L5: Fetch from all configured accounts
    if len(accounts) > 1:
        print(f"   Multi-account mode: {len(accounts)} accounts configured")
        emails = fetch_emails_from_all_accounts()
    else:
        # Single account - use legacy path for backward compatibility
        config = IMAPConfig()
        emails = fetch_unprocessed_emails(config)
    
    if not emails:
        print("   No new emails to process")
        return {}
    
    # Process the first email found
    email_data = emails[0]
    
    return {
        "email_metadata": email_data["metadata"],
        "raw_pdf_path": email_data["primary_pdf"],
        "status": "Processing"
    }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test the IMAP listener standalone.
    
    Usage:
        export IMAP_HOST=imap.gmail.com
        export IMAP_USERNAME=your-email@gmail.com
        export IMAP_PASSWORD=your-app-password
        export USE_MOCK_IMAP=false
        python nodes/imap_listener.py
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create minimal test state
    test_state: DealState = {
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
    
    result = imap_listener_node(test_state)
    
    print("\n--- Result ---")
    for key, value in result.items():
        print(f"  {key}: {value}")