# Dotloop Integration - Quick Reference

## Files Created

```
backend/
├── dotloop_client_enhanced.py    # Full Dotloop API client
├── dotloop_mapping.py            # Data conversion utilities  
├── nodes/executor_enhanced.py    # Integration workflow
├── test_dotloop.py              # Connection test script
├── DOTLOOP_INTEGRATION.md       # Complete setup guide
└── QUICK_REFERENCE.md           # This file
```

## Setup (3 Steps)

```bash
# 1. Test connection
cd backend
python3 test_dotloop.py

# 2. Add profile ID to .env (get from test output)
echo "DOTLOOP_PROFILE_ID=12345" >> .env

# 3. Update main.py
# Change: from nodes.executor import api_executor_node
# To:     from nodes.executor_enhanced import api_executor_node
```

## API Methods Cheat Sheet

### Profiles
```python
from dotloop_client_enhanced import DotloopClient

with DotloopClient() as client:
    profiles = client.list_profiles()
    profile = client.get_profile(profile_id)
```

### Loops
```python
# Create
loop = client.create_loop(profile_id, "Property Address", "PURCHASE_OFFER", "PRE_OFFER")

# Find existing
loop = client.find_existing_loop(profile_id, "Property Address")

# Update
loop = client.update_loop(profile_id, loop_id, status="UNDER_CONTRACT")
```

### Loop Details
```python
# Get all details
details = client.get_loop_details(profile_id, loop_id)

# Update details (partial updates supported)
client.update_loop_details(
    profile_id, 
    loop_id,
    {
        "Property Address": {"City": "Chicago"},
        "Financials": {"Purchase/Sale Price": "500000"}
    }
)
```

### Participants
```python
# List
participants = client.list_participants(profile_id, loop_id)

# Add
participant = client.add_participant(
    profile_id, loop_id,
    full_name="John Doe",
    email="john@example.com", 
    role="BUYER"
)

# Update
client.update_participant(profile_id, loop_id, participant_id, email="newemail@example.com")
```

### Folders & Documents
```python
# Find or create folder
folder = client.find_or_create_folder(profile_id, loop_id, "Contracts")

# Upload document
doc = client.upload_document(
    profile_id, loop_id, folder['id'],
    file_path="/path/to/file.pdf",
    file_name="Buy-Sell Agreement.pdf"
)
```

## Dotloop Transaction Types

```
PURCHASE_OFFER
LISTING_FOR_SALE
LISTING_FOR_LEASE
LEASE_OFFER
REAL_ESTATE_OTHER
OTHER
```

## Dotloop Statuses

**PURCHASE_OFFER:**
- PRE_OFFER
- UNDER_CONTRACT
- SOLD
- ARCHIVED

**LISTING_FOR_SALE:**
- PRE_LISTING
- PRIVATE_LISTING
- ACTIVE_LISTING
- UNDER_CONTRACT
- SOLD
- ARCHIVED

## Participant Roles

```
BUYER, SELLER
LISTING_AGENT, BUYING_AGENT
LISTING_BROKER, BUYING_BROKER
ESCROW_TITLE_REP
LOAN_OFFICER, LOAN_PROCESSOR
APPRAISER, INSPECTOR
TRANSACTION_COORDINATOR
BUYER_ATTORNEY, SELLER_ATTORNEY
ADMIN, MANAGING_BROKER
PROPERTY_MANAGER, TENANT, LANDLORD
OTHER
```

## Rate Limits

- **100 requests per minute**
- Client tracks: `client.rate_limit_remaining`
- Reset time: `client.rate_limit_reset` (milliseconds)

## Error Handling

```python
from dotloop_client_enhanced import DotloopAPIError

try:
    loop = client.create_loop(...)
except DotloopAPIError as e:
    print(f"Status: {e.status_code}")
    print(f"Message: {e.message}")
    print(f"Response: {e.response}")
```

## Common Mappings

### Loop Details Fields

**Property Address:**
- Country, Street Number, Street Name, Unit Number
- City, State/Prov, Zip/Postal Code, County
- MLS Number, Parcel/Tax ID

**Financials:**
- Purchase/Sale Price
- Earnest Money Amount, Earnest Money Held By
- Sale Commission Rate, Sale Commission Total
- Sale Commission Split % - Buy Side
- Sale Commission Split % - Sell Side

**Contract Dates:**
- Contract Agreement Date, Closing Date

**Offer Dates:**
- Offer Date, Offer Expiration Date
- Inspection Date, Occupancy Date

### Date Format
All dates: `MM/DD/YYYY` (e.g., "12/31/2024")

## Environment Variables

```bash
# Required
DOTLOOP_API_TOKEN=your_token
DOTLOOP_PROFILE_ID=12345

# Optional
DOTLOOP_TEMPLATE_ID=67890
```

## Testing

```bash
# Full connection test
python3 test_dotloop.py

# Quick profile check
python3 -c "from dotloop_client_enhanced import DotloopClient; \
client = DotloopClient(); \
print(client.list_profiles())"
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| No profiles found | Check API token |
| Rate limit (429) | Wait 60 seconds |
| Template required | Get template ID from test script |
| Participant exists | Normal - duplicates are skipped |

## Quick Debug

```python
import os
from dotloop_client_enhanced import DotloopClient

# Print config
print(f"Token: {os.getenv('DOTLOOP_API_TOKEN')[:20]}...")
print(f"Profile: {os.getenv('DOTLOOP_PROFILE_ID')}")

# Test connection
with DotloopClient() as c:
    profiles = c.list_profiles()
    print(f"Found {len(profiles)} profiles")
    for p in profiles:
        print(f"  - {p['name']} (ID: {p['id']})")
```

## Support

- Full docs: `DOTLOOP_INTEGRATION.md`
- API reference: https://dotloop.github.io/public-api/
- Code comments: See source files
