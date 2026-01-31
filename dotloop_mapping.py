"""
Utilities for mapping DealState to Dotloop API payloads.
Handles conversion of extracted data to Dotloop's expected formats.
"""
from typing import Dict, List, Any
from state import DealState, ParticipantInfo, PropertyAddress, FinancialDetails, ContractDates


def build_loop_name(state: DealState) -> str:
    """
    Build a descriptive loop name from deal state.
    Format: "Buyer Name, Property Address" or just property address if no buyer
    
    Examples:
        "Brian Erwin, 2100 Waterview Dr, San Francisco, CA 94114"
        "3059 Main St, Chicago, IL 60614"
    """
    name_parts = []
    
    # Add first buyer name if available
    if state.get("buyers") and len(state["buyers"]) > 0:
        name_parts.append(state["buyers"][0])
    
    # Add property address
    if state.get("property_details"):
        prop = state["property_details"]
        addr_parts = []
        
        if prop.get("street_number"):
            addr_parts.append(prop["street_number"])
        if prop.get("street_name"):
            addr_parts.append(prop["street_name"])
        
        street = " ".join(addr_parts) if addr_parts else None
        
        if street:
            name_parts.append(street)
        
        if prop.get("city"):
            name_parts.append(prop["city"])
        
        if prop.get("state"):
            name_parts.append(prop["state"])
        
        if prop.get("zip_code"):
            name_parts.append(prop["zip_code"])
    elif state.get("property_address"):
        # Fallback to simple string address
        name_parts.append(state["property_address"])
    
    if not name_parts:
        # Last resort: use deal_id
        return f"Transaction {state.get('deal_id', 'Unknown')}"
    
    return ", ".join(name_parts)


def map_transaction_type(state: DealState) -> str:
    """
    Determine transaction type from state.
    
    Returns one of:
        - PURCHASE_OFFER (most common)
        - LISTING_FOR_SALE
        - LISTING_FOR_LEASE
        - LEASE_OFFER
        - REAL_ESTATE_OTHER
        - OTHER
    """
    # You could add logic here to determine from document types or other indicators
    # For now, default to PURCHASE_OFFER as it's most common
    return "PURCHASE_OFFER"


def map_transaction_status(state: DealState, transaction_type: str) -> str:
    """
    Determine transaction status from state and type.
    
    For PURCHASE_OFFER:
        - PRE_OFFER, UNDER_CONTRACT, SOLD, ARCHIVED
    For LISTING_FOR_SALE:
        - PRE_LISTING, PRIVATE_LISTING, ACTIVE_LISTING, UNDER_CONTRACT, SOLD, ARCHIVED
    """
    if transaction_type == "PURCHASE_OFFER":
        # Check if contract is signed
        if state.get("contract_dates") and state["contract_dates"].get("contract_agreement_date"):
            return "UNDER_CONTRACT"
        return "PRE_OFFER"
    
    elif transaction_type == "LISTING_FOR_SALE":
        return "PRE_LISTING"
    
    # Default
    return "PRE_OFFER"


def map_property_details_to_dotloop(state: DealState) -> Dict[str, Any]:
    """
    Convert DealState property details to Dotloop Loop Details format.
    
    Returns:
        Dict with "Property Address" section
    """
    details = {}
    
    if state.get("property_details"):
        prop = state["property_details"]
        details["Property Address"] = {
            "Country": prop.get("country", "US"),
            "Street Number": prop.get("street_number", ""),
            "Street Name": prop.get("street_name", ""),
            "Unit Number": prop.get("unit", ""),
            "City": prop.get("city", ""),
            "State/Prov": prop.get("state", ""),
            "Zip/Postal Code": prop.get("zip_code", ""),
            "County": prop.get("county", ""),
            "MLS Number": prop.get("mls_number", ""),
            "Parcel/Tax ID": prop.get("parcel_tax_id", ""),
        }
    
    return details


def map_financial_details_to_dotloop(state: DealState) -> Dict[str, Any]:
    """
    Convert DealState financial details to Dotloop format.
    
    Returns:
        Dict with "Financials" section
    """
    details = {}
    
    if state.get("financial_details"):
        fin = state["financial_details"]
        details["Financials"] = {}
        
        if fin.get("purchase_sale_price"):
            details["Financials"]["Purchase/Sale Price"] = str(fin["purchase_sale_price"])
        
        if fin.get("earnest_money_amount"):
            details["Financials"]["Earnest Money Amount"] = str(fin["earnest_money_amount"])
        
        if fin.get("earnest_money_held_by"):
            details["Financials"]["Earnest Money Held By"] = fin["earnest_money_held_by"]
        
        if fin.get("sale_commission_rate"):
            details["Financials"]["Sale Commission Rate"] = fin["sale_commission_rate"]
        
        if fin.get("sale_commission_total"):
            details["Financials"]["Sale Commission Total"] = str(fin["sale_commission_total"])
        
        if fin.get("commission_split_buy_side_percent"):
            details["Financials"]["Sale Commission Split % - Buy Side"] = fin["commission_split_buy_side_percent"]
        
        if fin.get("commission_split_sell_side_percent"):
            details["Financials"]["Sale Commission Split % - Sell Side"] = fin["commission_split_sell_side_percent"]
    
    return details


def map_contract_dates_to_dotloop(state: DealState) -> Dict[str, Any]:
    """
    Convert DealState contract dates to Dotloop format.
    
    Returns:
        Dict with "Contract Dates" and "Offer Dates" sections
    """
    details = {}
    
    if state.get("contract_dates"):
        dates = state["contract_dates"]
        
        # Contract Dates section
        contract_dates = {}
        if dates.get("contract_agreement_date"):
            contract_dates["Contract Agreement Date"] = dates["contract_agreement_date"]
        if dates.get("closing_date"):
            contract_dates["Closing Date"] = dates["closing_date"]
        
        if contract_dates:
            details["Contract Dates"] = contract_dates
        
        # Offer Dates section
        offer_dates = {}
        if dates.get("offer_date"):
            offer_dates["Offer Date"] = dates["offer_date"]
        if dates.get("offer_expiration_date"):
            offer_dates["Offer Expiration Date"] = dates["offer_expiration_date"]
        if dates.get("inspection_date"):
            offer_dates["Inspection Date"] = dates["inspection_date"]
        if dates.get("occupancy_date"):
            offer_dates["Occupancy Date"] = dates["occupancy_date"]
        
        if offer_dates:
            details["Offer Dates"] = offer_dates
    
    return details


def map_all_loop_details(state: DealState) -> Dict[str, Any]:
    """
    Combine all detail sections for a complete loop details update.
    
    Returns:
        Complete details dict with Property Address, Financials, Contract Dates, etc.
    """
    details = {}
    
    # Merge all sections
    details.update(map_property_details_to_dotloop(state))
    details.update(map_financial_details_to_dotloop(state))
    details.update(map_contract_dates_to_dotloop(state))
    
    return details


def map_participants_to_dotloop(state: DealState) -> List[Dict[str, Any]]:
    """
    Convert ParticipantInfo list to Dotloop participant format.
    
    Returns:
        List of participant dicts ready for Dotloop API
    """
    dotloop_participants = []
    
    participants = state.get("participants", [])
    
    for p in participants:
        participant = {
            "fullName": p.get("full_name", ""),
            "email": p.get("email", ""),
            "role": map_participant_role(p.get("role", "OTHER")),
        }
        
        # Optional contact fields
        if p.get("phone"):
            participant["Phone"] = p["phone"]
        if p.get("company_name"):
            participant["Company Name"] = p["company_name"]
        if p.get("license_number"):
            participant["License #"] = p["license_number"]
        
        # Address fields
        if p.get("street_number"):
            participant["Street Number"] = p["street_number"]
        if p.get("street_name"):
            participant["Street Name"] = p["street_name"]
        if p.get("city"):
            participant["City"] = p["city"]
        if p.get("state"):
            participant["State/Prov"] = p["state"]
        if p.get("zip_code"):
            participant["Zip/Postal Code"] = p["zip_code"]
        
        dotloop_participants.append(participant)
    
    return dotloop_participants


def map_participant_role(role: str) -> str:
    """
    Map generic role names to Dotloop's specific role constants.
    
    Dotloop roles:
        BUYER, SELLER, LISTING_AGENT, BUYING_AGENT, LISTING_BROKER, BUYING_BROKER,
        ESCROW_TITLE_REP, LOAN_OFFICER, APPRAISER, INSPECTOR, etc.
    
    Args:
        role: Generic role from extraction (may be varied)
    
    Returns:
        Dotloop-compatible role constant
    """
    role_upper = role.upper().strip()
    
    # Direct matches
    dotloop_roles = {
        "BUYER", "SELLER", "LISTING_AGENT", "BUYING_AGENT", 
        "LISTING_BROKER", "BUYING_BROKER", "ESCROW_TITLE_REP",
        "LOAN_OFFICER", "APPRAISER", "INSPECTOR", "ADMIN",
        "BUYER_ATTORNEY", "SELLER_ATTORNEY", "TRANSACTION_COORDINATOR",
        "LOAN_PROCESSOR", "MANAGING_BROKER", "PROPERTY_MANAGER",
        "TENANT", "LANDLORD", "OTHER"
    }
    
    if role_upper in dotloop_roles:
        return role_upper
    
    # Fuzzy matching
    if "BUYER" in role_upper:
        if "AGENT" in role_upper or "BROKER" in role_upper:
            return "BUYING_AGENT"
        if "ATTORNEY" in role_upper or "LAWYER" in role_upper:
            return "BUYER_ATTORNEY"
        return "BUYER"
    
    if "SELLER" in role_upper:
        if "AGENT" in role_upper or "BROKER" in role_upper:
            return "LISTING_AGENT"
        if "ATTORNEY" in role_upper or "LAWYER" in role_upper:
            return "SELLER_ATTORNEY"
        return "SELLER"
    
    if "LISTING" in role_upper and "AGENT" in role_upper:
        return "LISTING_AGENT"
    
    if "TITLE" in role_upper or "ESCROW" in role_upper:
        return "ESCROW_TITLE_REP"
    
    if "LOAN" in role_upper or "LENDER" in role_upper:
        if "PROCESSOR" in role_upper:
            return "LOAN_PROCESSOR"
        return "LOAN_OFFICER"
    
    if "ATTORNEY" in role_upper or "LAWYER" in role_upper:
        return "OTHER"  # Generic attorney
    
    if "AGENT" in role_upper or "BROKER" in role_upper:
        return "BUYING_AGENT"  # Default agent
    
    # Default fallback
    return "OTHER"


def map_document_to_folder(doc_type: str) -> str:
    """
    Map document type to appropriate Dotloop folder name.
    
    Args:
        doc_type: Document type from classification
    
    Returns:
        Folder name (e.g., "Contracts", "Disclosures", "Addenda")
    """
    doc_type_upper = doc_type.upper().strip()
    
    # Contract documents
    if any(term in doc_type_upper for term in ["BUY-SELL", "PURCHASE", "SALE", "CONTRACT", "AGREEMENT"]):
        return "Contracts"
    
    # Disclosure documents
    if "DISCLOSURE" in doc_type_upper:
        return "Disclosures"
    
    # Counter offers and addenda
    if any(term in doc_type_upper for term in ["COUNTER", "ADDEND", "AMENDMENT"]):
        return "Addenda"
    
    # Inspection reports
    if "INSPECTION" in doc_type_upper:
        return "Inspections"
    
    # Title documents
    if "TITLE" in doc_type_upper:
        return "Title"
    
    # Default catch-all
    return "Documents"
