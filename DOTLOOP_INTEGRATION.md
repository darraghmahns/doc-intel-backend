# Enhanced Dotloop Integration - Setup Guide

## What Was Built

I've created a **comprehensive Dotloop API v2 integration** for your Doc Intel project with three new files:

### 1. `dotloop_client_enhanced.py` 
**Full-featured Dotloop API client** with:
- ✅ Profile management
- ✅ Loop creation, update, find
- ✅ Loop details (property, financials, contract dates)
- ✅ Participant management (add, update, delete)
- ✅ Folder management (create, find, list)
- ✅ Document upload
- ✅ Template support
- ✅ Rate limit tracking
- ✅ Comprehensive error handling

### 2. `dotloop_mapping.py`
**Smart mapping utilities** that convert your DealState to Dotloop format:
- Loop naming (property address + buyer name)
- Transaction type determination
- Property details formatting
- Financial details conversion
- Contract dates mapping
- Participant role normalization
- Document-to-folder mapping

### 3. `nodes/executor_enhanced.py`
**Complete integration workflow** that:
1. Creates or finds loops automatically
2. Updates all loop details
3. Adds participants (avoids duplicates)
4. Creates folders intelligently
5. Uploads documents to correct folders
6. Handles errors gracefully

---

## Quick Start

### Step 1: Add Profile ID to .env

Add this line to your `.env` file:

```bash
# Get your profile_id by running the setup script below
DOTLOOP_PROFILE_ID=your_profile_id_here
```

### Step 2: Run Setup Script

```bash
cd /Users/darraghmahns/Desktop/Projects/Active/real_estate/doc_intel/backend
python3 -c "
from dotloop_client_enhanced import DotloopClient
import os

client = DotloopClient()
profiles = client.list_profiles()

print('\n=== Your Dotloop Profiles ===')
for p in profiles:
    default = ' (DEFAULT)' if p.get('default') else ''
    print(f\"ID: {p['id']:<10} Name: {p['name']}{default}\")
    print(f\"   Type: {p.get('type')}, Company: {p.get('company', 'N/A')}\")
    print()

print('Copy the ID of your preferred profile to DOTLOOP_PROFILE_ID in .env')
"
```

### Step 3: Update Your main.py

Replace the import in `main.py`:

```python
# OLD
from nodes.executor import api_executor_node

# NEW
from nodes.executor_enhanced import api_executor_node
```

### Step 4: Test It!

```bash
python main.py
```

---

## Features Breakdown

### Automatic Loop Management

The system will:
- **Create new loops** if they don't exist
- **Find existing loops** by name to avoid duplicates
- **Update loops** with extracted data

### Smart Participant Handling

- Detects existing participants to avoid duplicates
- Maps generic roles to Dotloop's specific roles:
  - "Buyer" → `BUYER`
  - "Seller" → `SELLER` 
  - "Buyer's Agent" → `BUYING_AGENT`
  - "Listing Agent" → `LISTING_AGENT`
  - "Title Company" → `ESCROW_TITLE_REP`
  - etc.

### Intelligent Folder Organization

Documents automatically go to the right folder:
- **Buy-Sell Agreements** → "Contracts" folder
- **Disclosures** → "Disclosures" folder
- **Counter Offers/Addenda** → "Addenda" folder
- **Inspections** → "Inspections" folder
- **Title Documents** → "Title" folder
- **Other** → "Documents" folder

### Error Recovery

- Tracks rate limits (100 requests/minute)
- Continues processing even if some operations fail
- Returns detailed error messages
- Partial success tracking

---

## Configuration Options

### Environment Variables

```bash
# Required
DOTLOOP_API_TOKEN=your_token_here

# Required (get from setup script)
DOTLOOP_PROFILE_ID=your_profile_id

# Optional - Template ID if your profile requires templates
DOTLOOP_TEMPLATE_ID=template_id_here
```

---

## Advanced Usage

### Using Loop-It API (Simplified Creation)

If you want to create a loop with everything in one call:

```python
from dotloop_client_enhanced import DotloopClient
from dotloop_mapping import map_participants_to_dotloop

with DotloopClient() as client:
    result = client.loop_it(
        profile_id=12345,
        name="Brian Erwin, 123 Main St, Chicago, IL",
        transaction_type="PURCHASE_OFFER",
        status="PRE_OFFER",
        street_number="123",
        street_name="Main St",
        city="Chicago",
        state="IL",
        zip_code="60614",
        participants=[
            {
                "fullName": "Brian Erwin",
                "email": "brian@example.com",
                "role": "BUYER"
            }
        ]
    )
    
    print(f"Loop created: {result['loopUrl']}")
```

### Updating Specific Loop Details

```python
from dotloop_client_enhanced import DotloopClient

with DotloopClient() as client:
    client.update_loop_details(
        profile_id=12345,
        loop_id=67890,
        details={
            "Financials": {
                "Purchase/Sale Price": "500000",
                "Earnest Money Amount": "10000",
            },
            "Contract Dates": {
                "Closing Date": "12/31/2024"
            }
        }
    )
```

### Uploading a Document

```python
from dotloop_client_enhanced import DotloopClient

with DotloopClient() as client:
    # Find or create folder
    folder = client.find_or_create_folder(
        profile_id=12345,
        loop_id=67890,
        folder_name="Contracts"
    )
    
    # Upload document
    doc = client.upload_document(
        profile_id=12345,
        loop_id=67890,
        folder_id=folder['id'],
        file_path="/path/to/contract.pdf",
        file_name="Buy-Sell Agreement.pdf"
    )
    
    print(f"Uploaded: {doc['name']}")
```

---

## Troubleshooting

### "No profiles found"
- Make sure your API token is correct
- Check that your Dotloop account has at least one profile

### "Rate limit exceeded"
- Wait 60 seconds and retry
- The client tracks rate limits automatically
- You get 100 requests per minute

### "Template required"
- Some Dotloop profiles require templates
- Get template ID: `client.list_templates(profile_id)`
- Add to .env: `DOTLOOP_TEMPLATE_ID=123`

### "Participant already exists"
- The enhanced executor checks for duplicates automatically
- This is just informational, not an error

---

## Next Steps

### Phase 2 Enhancements (Optional)

1. **OAuth2 Authentication** 
   - For multi-user access
   - More secure than static tokens
   - See Dotloop docs: https://dotloop.github.io/public-api/#authentication

2. **Webhooks**
   - Get notified when loops change
   - Track signature completion
   - Sync status updates back to your system

3. **Retry Logic with Exponential Backoff**
   - Auto-retry failed operations
   - Queue for later processing

4. **Advanced Document Processing**
   - Upload individual split PDFs (not just original)
   - Extract signatures and map to participants
   - Prepare for e-signature workflow

---

## API Reference

Full Dotloop API documentation:
https://dotloop.github.io/public-api/

Your integration plan was based on the official API docs with Montana real estate workflows in mind.

---

## Questions?

The code is extensively commented. Check:
- `dotloop_client_enhanced.py` - All API methods documented
- `dotloop_mapping.py` - Mapping logic with examples
- `nodes/executor_enhanced.py` - Complete workflow walkthrough

Happy automating! 🚀
