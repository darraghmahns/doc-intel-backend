"""
FastAPI Server for Doc Intel Human Review API

Provides endpoints for:
- Listing deals awaiting review
- Fetching deal details (extracted data)
- Serving PDF files
- Submitting review corrections
- Seeding test deals
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid
import shutil

from state import (
    DealState,
    PropertyAddress,
    ParticipantInfo,
    FinancialDetails,
    ContractDates,
    ValidationError,
    ApprovalChainConfig,
    ApprovalStatus,
    ApproverInfo,
    RejectionInfo,
)
import json
from deal_storage import save_deal, load_deal, list_all_deals, delete_deal, update_deal
from datetime import datetime

# ============================================================================
# In-Memory Deal Storage (MVP - replace with DB later)
# ============================================================================

deals_store: Dict[str, DealState] = {}

# Storage directory for PDFs
STORAGE_DIR = Path(__file__).parent / "storage" / "pdfs"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Doc Intel API",
    description="Human Review API for Real Estate Document Intelligence",
    version="0.1.0",
)

# CORS for React frontend (dev server typically on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models for API
# ============================================================================

class DealSummary(BaseModel):
    """Summary of a deal for list view."""
    deal_id: str
    status: str
    property_address: Optional[str]
    created_at: Optional[str] = None


class ReviewSubmission(BaseModel):
    """Data submitted when user completes review."""
    approved: bool
    corrections: Optional[Dict[str, Any]] = None


class SeedDealRequest(BaseModel):
    """Request to create a seed deal for testing."""
    property_address: Optional[str] = "123 Test Street, Denver, CO 80202"
    buyer_name: Optional[str] = "John Buyer"
    seller_name: Optional[str] = "Jane Seller"
    purchase_price: Optional[float] = 450000.00


class RejectionRequest(BaseModel):
    """Request to reject a deal with reason and action."""
    reason: str  # 'incorrect_extraction', 'missing_data', 'invalid_document', 'other'
    action: str  # 're_extract', 'manual_edit', 'request_new_document'
    notes: Optional[str] = None
    rejected_by: Optional[str] = "reviewer"


class ApprovalChainConfigRequest(BaseModel):
    """Request to configure approval chain rules."""
    deal_value_threshold: float = 500000.0
    deal_types_requiring_approval: List[str] = []
    required_approvers: List[str] = ["manager"]
    enabled: bool = True


class ManagerApprovalRequest(BaseModel):
    """Request for manager to approve their level in the chain."""
    approver_id: str
    approver_name: str
    approved: bool
    notes: Optional[str] = None


# Default approval chain config (in-memory, replace with DB later)
approval_chain_config_store: ApprovalChainConfig = {
    "deal_value_threshold": 500000.0,
    "deal_types_requiring_approval": [],
    "required_approvers": ["manager"],
    "enabled": True,
}


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "doc-intel-api"}


@app.get("/api/deals", response_model=List[DealSummary])
def list_deals():
    """List all deals awaiting review."""
    all_deals = list_all_deals()
    return [
        DealSummary(
            deal_id=deal["deal_id"],
            status=deal["status"],
            property_address=deal.get("property_address"),
        )
        for deal in all_deals
    ]


@app.get("/api/deals/{deal_id}")
def get_deal(deal_id: str) -> Dict[str, Any]:
    """Get full deal details for review UI."""
    deal = load_deal(deal_id)
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    # Return a JSON-serializable version of the deal state
    return {
        "deal_id": deal["deal_id"],
        "status": deal["status"],
        "property_address": deal.get("property_address"),
        "property_details": deal.get("property_details"),
        "buyers": deal.get("buyers", []),
        "sellers": deal.get("sellers", []),
        "participants": deal.get("participants", []),
        "financials": deal.get("financials", {}),
        "financial_details": deal.get("financial_details"),
        "contract_dates": deal.get("contract_dates"),
        "validation_errors": deal.get("validation_errors", []),
        "human_approval_status": deal.get("human_approval_status"),
        "approval_status": deal.get("approval_status"),
        "rejection_history": deal.get("rejection_history", []),
        "current_rejection": deal.get("current_rejection"),
        "dotloop_loop_id": deal.get("dotloop_loop_id"),
        "dotloop_loop_url": deal.get("dotloop_loop_url"),
        "sync_errors": deal.get("sync_errors", []),
        "dotloop_sync_action": deal.get("dotloop_sync_action"),
    }


@app.get("/api/deals/{deal_id}/pdf")
def get_deal_pdf(deal_id: str):
    """Serve the PDF file for a deal."""
    deal = load_deal(deal_id)
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    pdf_path = deal.get("raw_pdf_path")
    
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"deal_{deal_id}.pdf",
    )


@app.post("/api/deals/{deal_id}/review")
async def submit_review(deal_id: str, submission: ReviewSubmission):
    """Submit review decision and corrections."""
    deal = load_deal(deal_id)
    
    if not deal:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    if submission.approved:
        deal["human_approval_status"] = "Approved"
        deal["status"] = "Approved"
        
        # Apply corrections if provided
        if submission.corrections:
            # Update property details
            if "property_details" in submission.corrections:
                deal["property_details"] = submission.corrections["property_details"]
            
            # Update participants
            if "participants" in submission.corrections:
                deal["participants"] = submission.corrections["participants"]
            
            # Update financials
            if "financial_details" in submission.corrections:
                deal["financial_details"] = submission.corrections["financial_details"]
            
            # Update contract dates
            if "contract_dates" in submission.corrections:
                deal["contract_dates"] = submission.corrections["contract_dates"]
        
        # Clear validation errors after corrections
        deal["validation_errors"] = []
        
        # === DOTLOOP INTEGRATION ===
        # Run the Dotloop sync when deal is approved
        try:
            from nodes.executor_enhanced import api_executor_node
            import os
            
            # Make sure environment is loaded
            from dotenv import load_dotenv
            load_dotenv()
            
            print(f"   Running Dotloop sync for deal {deal_id}...")
            print(f"   Profile ID: {os.getenv('DOTLOOP_PROFILE_ID')}")
            print(f"   API Token: {os.getenv('DOTLOOP_API_TOKEN')[:20] if os.getenv('DOTLOOP_API_TOKEN') else 'NOT SET'}...")
            
            result = api_executor_node(deal)
            
            # Update deal with Dotloop results
            deal["dotloop_loop_id"] = result.get("dotloop_loop_id")
            deal["dotloop_loop_url"] = result.get("dotloop_loop_url")
            deal["sync_errors"] = result.get("sync_errors", [])
            deal["dotloop_sync_action"] = result.get("dotloop_sync_action")
            
            if result.get("status") == "Synced":
                print(f"   ✓ Dotloop sync successful: {deal['dotloop_loop_url']}")
            elif result.get("status") == "Partial":
                print(f"   ⚠ Partial Dotloop sync with errors: {deal.get('sync_errors')}")
            else:
                print(f"   ✗ Dotloop sync failed: {result.get('status')}")
                print(f"   Errors: {deal.get('sync_errors')}")
                
        except Exception as e:
            import traceback
            print(f"   ✗ Dotloop sync error: {e}")
            print(traceback.format_exc())
            deal["sync_errors"] = [f"Dotloop sync failed: {str(e)}"]
    
    else:
        deal["human_approval_status"] = "Rejected"
        deal["status"] = "Needs_Review"
        
        # Apply corrections if provided
        if submission.corrections:
            # Update property details
            if "property_details" in submission.corrections:
                deal["property_details"] = submission.corrections["property_details"]
            
            # Update participants
            if "participants" in submission.corrections:
                deal["participants"] = submission.corrections["participants"]
            
            # Update financials
            if "financial_details" in submission.corrections:
                deal["financial_details"] = submission.corrections["financial_details"]
            
            # Update contract dates
            if "contract_dates" in submission.corrections:
                deal["contract_dates"] = submission.corrections["contract_dates"]
        
        # Clear validation errors after corrections
        deal["validation_errors"] = []
    
    # Save the updated deal
    save_deal(deal)
    
    return {
        "message": "Review submitted successfully",
        "deal_id": deal_id,
        "status": deal["status"],
        "human_approval_status": deal["human_approval_status"],
        "dotloop_loop_url": deal.get("dotloop_loop_url"),
        "sync_errors": deal.get("sync_errors", []),
    }


@app.post("/api/seed")
def seed_deal(request: SeedDealRequest):
    """Create a mock deal for testing without sending real emails."""
    deal_id = f"SEED-{uuid.uuid4().hex[:8].upper()}"
    
    # Create sample property details
    property_details: PropertyAddress = {
        "street_number": "123",
        "street_name": "Test Street",
        "city": "Denver",
        "state": "CO",
        "zip_code": "80202",
        "country": "US",
        "full_address": request.property_address or "123 Test Street, Denver, CO 80202",
    }
    
    # Create sample participants
    participants: List[ParticipantInfo] = [
        {
            "full_name": request.buyer_name or "John Buyer",
            "email": "buyer@example.com",
            "phone": "555-123-4567",
            "role": "BUYER",
        },
        {
            "full_name": request.seller_name or "Jane Seller",
            "email": "seller@example.com",
            "phone": "555-987-6543",
            "role": "SELLER",
        },
    ]
    
    # Create sample financial details
    financial_details: FinancialDetails = {
        "purchase_sale_price": request.purchase_price or 450000.00,
        "earnest_money_amount": 10000.00,
        "sale_commission_rate": "6%",
        "commission_split_buy_side_percent": "3%",
        "commission_split_sell_side_percent": "3%",
    }
    
    # Create sample contract dates
    contract_dates: ContractDates = {
        "contract_agreement_date": "01/15/2026",
        "closing_date": "02/28/2026",
        "offer_date": "01/10/2026",
        "offer_expiration_date": "01/12/2026",
        "inspection_date": "01/20/2026",
    }
    
    # Sample validation errors for demo
    validation_errors: List[ValidationError] = [
        {
            "field": "participants[0].email",
            "message": "Buyer email may need verification",
            "expected_format": "Valid email address",
            "severity": "Warning",
        },
    ]
    
    # Determine if approval chain is needed based on deal value
    purchase_price = request.purchase_price or 450000.00
    requires_approval = (
        approval_chain_config_store.get("enabled", False) and
        purchase_price >= approval_chain_config_store.get("deal_value_threshold", 500000.0)
    )
    
    approval_status: ApprovalStatus = {
        "requires_chain_approval": requires_approval,
        "chain_config": approval_chain_config_store if requires_approval else None,
        "approvers": [
            {"user_id": "mgr-001", "name": "Manager", "role": role, "approved": False}
            for role in approval_chain_config_store.get("required_approvers", [])
        ] if requires_approval else [],
        "current_level": 0,
        "fully_approved": False,
    }
    
    # Create the deal state
    deal_state: DealState = {
        "deal_id": deal_id,
        "status": "Pending_Review",
        "email_metadata": {"sender": "test@example.com", "subject": "Test Deal"},
        "raw_pdf_path": "",  # No PDF for seed deals
        "split_docs": [],
        "property_address": request.property_address,
        "property_details": property_details,
        "buyers": [request.buyer_name or "John Buyer"],
        "sellers": [request.seller_name or "Jane Seller"],
        "participants": participants,
        "financials": {"purchase_price": purchase_price},
        "financial_details": financial_details,
        "contract_dates": contract_dates,
        "validation_errors": validation_errors,
        "signature_fields": [],
        "signature_mapping": {},
        "missing_docs": [],
        "human_approval_status": "Pending",
        "approval_status": approval_status,
        "rejection_history": [],
        "current_rejection": None,
        "target_system": "dotloop",
        "dotloop_payload": None,
        "brokerage_settings": None,
        "dotloop_loop_id": None,
        "dotloop_loop_url": None,
        "sync_errors": [],
        "dotloop_sync_action": None,
    }
    
    # Store the deal
    save_deal(deal_state)
    
    return {
        "message": "Seed deal created successfully",
        "deal_id": deal_id,
        "status": deal_state["status"],
    }


@app.delete("/api/deals/{deal_id}")
def delete_deal_endpoint(deal_id: str):
    """Delete a deal (for testing cleanup)."""
    success = delete_deal(deal_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Deal not found")
    return {"message": "Deal deleted", "deal_id": deal_id}


# ============================================================================
# Rejection Endpoints (H4)
# ============================================================================

@app.post("/api/deals/{deal_id}/reject")
def reject_deal(deal_id: str, rejection: RejectionRequest):
    """Reject a deal with reason and action (H4)."""
    if deal_id not in deals_store:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    deal = deals_store[deal_id]
    
    # Create rejection info
    rejection_info: RejectionInfo = {
        "reason": rejection.reason,
        "action": rejection.action,
        "notes": rejection.notes,
        "rejected_by": rejection.rejected_by or "reviewer",
        "rejected_at": datetime.now().isoformat(),
    }
    
    # Add to rejection history
    if "rejection_history" not in deal:
        deal["rejection_history"] = []
    deal["rejection_history"].append(rejection_info)
    deal["current_rejection"] = rejection_info
    
    # Update status based on action
    deal["human_approval_status"] = "Rejected"
    
    if rejection.action == "re_extract":
        deal["status"] = "Pending_ReExtraction"
    elif rejection.action == "manual_edit":
        deal["status"] = "Pending_Manual_Edit"
    elif rejection.action == "request_new_document":
        deal["status"] = "Awaiting_New_Document"
    else:
        deal["status"] = "Rejected"
    
    return {
        "message": "Deal rejected",
        "deal_id": deal_id,
        "status": deal["status"],
        "action": rejection.action,
        "rejection_info": rejection_info,
    }


# ============================================================================
# Approval Chain Endpoints (H3)
# ============================================================================

@app.get("/api/config/approval-chains")
def get_approval_chain_config():
    """Get current approval chain configuration."""
    return approval_chain_config_store


@app.put("/api/config/approval-chains")
def update_approval_chain_config(config: ApprovalChainConfigRequest):
    """Update approval chain configuration."""
    global approval_chain_config_store
    approval_chain_config_store = {
        "deal_value_threshold": config.deal_value_threshold,
        "deal_types_requiring_approval": config.deal_types_requiring_approval,
        "required_approvers": config.required_approvers,
        "enabled": config.enabled,
    }
    return {
        "message": "Approval chain config updated",
        "config": approval_chain_config_store,
    }


@app.get("/api/deals/{deal_id}/approval-chain")
def get_deal_approval_chain(deal_id: str):
    """Get approval chain status for a deal."""
    if deal_id not in deals_store:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    deal = deals_store[deal_id]
    return deal.get("approval_status", {
        "requires_chain_approval": False,
        "approvers": [],
        "current_level": 0,
        "fully_approved": True,
    })


@app.post("/api/deals/{deal_id}/approve-level")
def approve_level(deal_id: str, approval: ManagerApprovalRequest):
    """Manager approves their level in the approval chain."""
    if deal_id not in deals_store:
        raise HTTPException(status_code=404, detail="Deal not found")
    
    deal = deals_store[deal_id]
    approval_status = deal.get("approval_status")
    
    if not approval_status or not approval_status.get("requires_chain_approval"):
        raise HTTPException(status_code=400, detail="Deal does not require chain approval")
    
    approvers = approval_status.get("approvers", [])
    current_level = approval_status.get("current_level", 0)
    
    if current_level >= len(approvers):
        raise HTTPException(status_code=400, detail="All approval levels completed")
    
    # Update current approver
    current_approver = approvers[current_level]
    current_approver["approved"] = approval.approved
    current_approver["approved_at"] = datetime.now().isoformat()
    current_approver["notes"] = approval.notes
    current_approver["user_id"] = approval.approver_id
    current_approver["name"] = approval.approver_name
    
    if approval.approved:
        # Move to next level
        approval_status["current_level"] = current_level + 1
        
        # Check if fully approved
        if approval_status["current_level"] >= len(approvers):
            approval_status["fully_approved"] = True
            deal["human_approval_status"] = "Approved"
            deal["status"] = "Approved"
    else:
        # Rejection at any level rejects the deal
        deal["human_approval_status"] = "Rejected"
        deal["status"] = "Rejected_By_Manager"
    
    return {
        "message": "Approval level processed",
        "deal_id": deal_id,
        "approval_status": approval_status,
        "deal_status": deal["status"],
    }


# ============================================================================
# Run with: uvicorn server:app --reload
# ============================================================================
