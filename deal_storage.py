"""
Shared deal storage using JSON files.
Allows main.py and server.py to share deal data.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from state import DealState

STORAGE_DIR = Path(__file__).parent / "storage" / "deals"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def save_deal(deal: DealState) -> None:
    """Save a deal to disk."""
    deal_id = deal["deal_id"]
    file_path = STORAGE_DIR / f"{deal_id}.json"
    
    # Convert deal to JSON-serializable format
    deal_json = {k: v for k, v in deal.items() if v is not None}
    
    with open(file_path, 'w') as f:
        json.dump(deal_json, f, indent=2, default=str)
    
    print(f"   💾 Saved deal {deal_id} to disk")


def load_deal(deal_id: str) -> Optional[DealState]:
    """Load a deal from disk."""
    file_path = STORAGE_DIR / f"{deal_id}.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)


def list_all_deals() -> List[DealState]:
    """List all deals from disk."""
    deals = []
    
    for file_path in STORAGE_DIR.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                deal = json.load(f)
                deals.append(deal)
        except Exception as e:
            print(f"   ⚠ Error loading {file_path}: {e}")
    
    return deals


def delete_deal(deal_id: str) -> bool:
    """Delete a deal from disk."""
    file_path = STORAGE_DIR / f"{deal_id}.json"
    
    if file_path.exists():
        file_path.unlink()
        print(f"   🗑️  Deleted deal {deal_id}")
        return True
    
    return False


def update_deal(deal_id: str, updates: Dict) -> Optional[DealState]:
    """Update specific fields of a deal."""
    deal = load_deal(deal_id)
    
    if not deal:
        return None
    
    # Apply updates
    deal.update(updates)
    
    # Save back to disk
    save_deal(deal)
    
    return deal
