# Doc Intel - Backend

Real estate document intelligence backend with Dotloop API integration.

## Features

- Email monitoring (Gmail IMAP)
- PDF document extraction
- LangChain-based field extraction
- Dotloop OAuth2 integration
- Human review API (FastAPI)
- Deal storage (JSON file-based)

## Tech Stack

- Python 3.10+
- FastAPI
- LangChain
- Dotloop API v2
- Gmail IMAP

## Setup

### 1. Install Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required variables:
```bash
DOTLOOP_CLIENT_ID=your-client-id
DOTLOOP_CLIENT_SECRET=your-client-secret
DOTLOOP_PROFILE_ID=your-profile-id
IMAP_USERNAME=your-email@gmail.com
IMAP_PASSWORD=your-gmail-app-password
```

### 3. Setup Dotloop OAuth

```bash
python3 dotloop_oauth.py
```

### 4. Test Connection

```bash
python3 test_dotloop.py
```

## Running

### API Server (Development)
```bash
uvicorn server:app --reload --port 8000
```

### Email Pipeline (Manual)
```bash
python main.py
```

## API Endpoints

- `GET /api/deals` - List all deals pending review
- `GET /api/deals/{deal_id}` - Get specific deal
- `PUT /api/deals/{deal_id}` - Update deal fields
- `POST /api/deals/{deal_id}/approve` - Approve and send to Dotloop
- `POST /api/deals/{deal_id}/reject` - Reject with feedback

## Project Structure

```
backend/
├── main.py                      # Email pipeline entry point
├── server.py                    # FastAPI server
├── state.py                     # DealState data model
├── deal_storage.py              # File-based storage
├── dotloop_client_enhanced.py   # Dotloop API client
├── dotloop_mapping.py           # Data mapping utilities
├── dotloop_oauth.py             # OAuth setup script
├── test_dotloop.py              # Connection test
├── nodes/                       # LangGraph nodes
│   ├── imap_listener.py
│   ├── splitter.py
│   ├── classifier.py
│   ├── extractor.py
│   ├── mapper.py
│   └── executor_enhanced.py
├── storage/
│   ├── deals/                   # JSON deal files
│   └── pdfs/                    # Uploaded PDFs
└── tests/
```

## Development

### Run Tests
```bash
pytest
```

### Add Mock Mode (for testing without external services)
```bash
export USE_MOCK_IMAP=true
export USE_MOCK_SPLITTER=true
export USE_MOCK_CLASSIFIER=true
```

## Dotloop Integration

See Notion docs under Project GBTM:
- Dotloop Integration Guide
- Quick Reference
