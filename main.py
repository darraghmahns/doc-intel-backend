from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Import State
from state import DealState

# Import Nodes
from nodes.imap_listener import imap_listener_node
from nodes.splitter import intelligent_splitter_node
from nodes.classifier import doc_type_classifier_node
from nodes.extractor import extraction_node
from nodes.mapper import field_mapper_node
from nodes.executor_enhanced import api_executor_node
from nodes.feedback import imap_tagger_node

# Load Env
load_dotenv()

def human_review_node(state: DealState):
    """
    Node F: Human In The Loop
    This is a placeholder node that acts as a breakpoint.
    """
    print("--- NODE: Human Review (Breakpoint) ---")
    # In a real app, we would pause here. 
    # For the script, we assume approval.
    return {"human_approval_status": "Approved"}

def build_graph():
    """
    Constructs the LangGraph state machine.
    """
    builder = StateGraph(DealState)
    
    # 1. Add Nodes
    builder.add_node("listener", imap_listener_node)
    builder.add_node("splitter", intelligent_splitter_node)
    builder.add_node("classifier", doc_type_classifier_node)
    builder.add_node("extractor", extraction_node)
    builder.add_node("mapper", field_mapper_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("executor", api_executor_node)
    builder.add_node("tagger", imap_tagger_node)
    
    # 2. Add Edges (The Flow)
    builder.add_edge(START, "listener")
    
    # Conditional logic: Did we find an email?
    def check_email(state):
        if state.get("email_metadata"):
            return "splitter"
        return END

    builder.add_conditional_edges("listener", check_email)
    
    builder.add_edge("splitter", "classifier")
    builder.add_edge("classifier", "extractor")
    builder.add_edge("extractor", "mapper")
    builder.add_edge("mapper", "human_review")
    
    # Conditional logic: Did the human approve?
    def check_approval(state):
        if state.get("human_approval_status") == "Approved":
            return "executor"
        return "mapper" # Loop back or go to notification node

    builder.add_conditional_edges("human_review", check_approval)
    
    builder.add_edge("executor", "tagger")
    builder.add_edge("tagger", END)
    
    # 3. Compile
    return builder.compile()

if __name__ == "__main__":
    app = build_graph()
    
    # Simulate an initial run
    print("Starting Doc Intel Agent...")
    initial_state: DealState = {
        "deal_id": "12345",
        "status": "Processing",
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
        "validation_errors": [],
        "signature_fields": [],
        "signature_mapping": {},
        "missing_docs": [],
        "human_approval_status": "Pending",
        "target_system": "dotloop",
        "dotloop_payload": None,
        "dotloop_loop_id": None,
        "dotloop_loop_url": None,
        "sync_errors": [],
        "dotloop_sync_action": None,
    }
    app.invoke(initial_state)