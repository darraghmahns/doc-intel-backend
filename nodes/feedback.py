from state import DealState

def imap_tagger_node(state: DealState) -> dict:
    """
    Node H: IMAP Tagger
    Tags the email as 'PROCESSED' so we don't read it again.
    """
    print("--- NODE: IMAP Tagger ---")
    print(f"   Tagging email {state['email_metadata'].get('msg_id')} as PROCESSED")
    
    return {}