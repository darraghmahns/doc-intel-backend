"""
Enhanced API Executor Node - Comprehensive Dotloop Integration
Handles: Loop creation, details updates, participants, folder management, document uploads
"""
import os
import time
from typing import Any, Dict, List
from pathlib import Path

from state import DealState
from dotloop_client_enhanced import DotloopClient, DotloopAPIError
from dotloop_mapping import (
    build_loop_name,
    map_transaction_type,
    map_transaction_status,
    map_all_loop_details,
    map_participants_to_dotloop,
    map_document_to_folder,
)


def api_executor_node(state: DealState) -> Dict[str, Any]:
    """
    Node G: Enhanced API Executor
    
    Comprehensive Dotloop integration:
    1. Create or find loop
    2. Update loop details (property, financials, dates)
    3. Add participants
    4. Create folders
    5. Upload documents
    6. Handle errors and retries
    
    Returns:
        Updated state with loop_id, loop_url, sync status, and errors
    """
    print("--- NODE: Enhanced API Executor ---")

    system = state.get("target_system", "dotloop")
    if system != "dotloop":
        return {"status": "Skipped", "sync_errors": ["Not configured for Dotloop"]}

    api_token = os.environ.get("DOTLOOP_API_TOKEN")
    if not api_token:
        return {
            "status": "Failed",
            "sync_errors": ["DOTLOOP_API_TOKEN not configured"],
        }

    # Get profile_id from environment or use default
    profile_id = os.environ.get("DOTLOOP_PROFILE_ID")
    if not profile_id:
        # We'll try to get the default profile
        print("   Warning: DOTLOOP_PROFILE_ID not set, will fetch default profile")

    errors: List[str] = []
    loop_id = state.get("dotloop_loop_id")
    loop_url = state.get("dotloop_loop_url")
    
    try:
        with DotloopClient(api_token=api_token) as client:
            
            # ================================================================
            # STEP 1: Get or Set Profile ID
            # ================================================================
            if not profile_id:
                print("   Fetching profiles...")
                profiles = client.list_profiles()
                if not profiles:
                    raise DotloopAPIError(500, "No profiles found for this account")
                
                # Use first profile or default profile
                for p in profiles:
                    if p.get("default"):
                        profile_id = p["id"]
                        break
                
                if not profile_id and profiles:
                    profile_id = profiles[0]["id"]
                
                print(f"   Using profile_id: {profile_id}")
            else:
                profile_id = int(profile_id)
            
            # ================================================================
            # STEP 2: Create or Find Loop
            # ================================================================
            loop_name = build_loop_name(state)
            transaction_type = map_transaction_type(state)
            status = map_transaction_status(state, transaction_type)
            
            if not loop_id:
                print(f"   Creating/finding loop: {loop_name}")
                
                # Try to find existing loop first
                existing = client.find_existing_loop(profile_id, loop_name)
                
                if existing:
                    loop_id = existing["id"]
                    loop_url = existing.get("loopUrl")
                    print(f"   Found existing loop {loop_id}")
                    sync_action = "Found"
                else:
                    # Create new loop
                    print(f"   Creating new loop with type={transaction_type}, status={status}")
                    created = client.create_loop(
                        profile_id=profile_id,
                        name=loop_name,
                        transaction_type=transaction_type,
                        status=status,
                    )
                    loop_id = created["id"]
                    loop_url = created.get("loopUrl")
                    print(f"   Created loop {loop_id}: {loop_url}")
                    sync_action = "Created"
            else:
                # Loop already exists, we'll update it
                print(f"   Using existing loop {loop_id}")
                sync_action = "Updated"
            
            # ================================================================
            # STEP 3: Update Loop Details
            # ================================================================
            print("   Updating loop details...")
            details = map_all_loop_details(state)
            
            if details:
                try:
                    client.update_loop_details(
                        profile_id=profile_id,
                        loop_id=loop_id,
                        details=details,
                    )
                    print(f"   ✓ Updated {len(details)} detail sections")
                except DotloopAPIError as e:
                    errors.append(f"Failed to update loop details: {e.message}")
                    print(f"   ✗ Error updating details: {e.message}")
            else:
                print("   No details to update")
            
            # ================================================================
            # STEP 4: Add Participants
            # ================================================================
            participants = map_participants_to_dotloop(state)
            
            if participants:
                print(f"   Adding {len(participants)} participants...")
                
                # Get existing participants to avoid duplicates
                try:
                    existing_participants = client.list_participants(profile_id, loop_id)
                    existing_emails = {p.get("email", "").lower() for p in existing_participants if p.get("email")}
                except DotloopAPIError:
                    existing_emails = set()
                
                added_count = 0
                for participant in participants:
                    email = participant.get("email", "").lower()
                    
                    # Skip if already exists
                    if email and email in existing_emails:
                        print(f"   - Skipping {participant['fullName']} (already exists)")
                        continue
                    
                    try:
                        client.add_participant(
                            profile_id=profile_id,
                            loop_id=loop_id,
                            full_name=participant.get("fullName", ""),
                            email=participant.get("email", ""),
                            role=participant.get("role", "OTHER"),
                            **{k: v for k, v in participant.items() if k not in ["fullName", "email", "role"]}
                        )
                        added_count += 1
                        print(f"   ✓ Added {participant.get('fullName')} ({participant.get('role')})")
                    except DotloopAPIError as e:
                        errors.append(f"Failed to add participant {participant.get('fullName')}: {e.message}")
                        print(f"   ✗ Error adding {participant.get('fullName')}: {e.message}")
                
                print(f"   Added {added_count}/{len(participants)} participants")
            else:
                print("   No participants to add")
            
            # ================================================================
            # STEP 5: Upload Documents
            # ================================================================
            split_docs = state.get("split_docs", [])
            
            if split_docs and state.get("raw_pdf_path"):
                print(f"   Processing {len(split_docs)} documents...")
                
                uploaded_count = 0
                for doc in split_docs:
                    doc_type = doc.get("doc_type", "Unknown")
                    folder_name = map_document_to_folder(doc_type)
                    
                    # Find or create folder
                    try:
                        folder = client.find_or_create_folder(
                            profile_id=profile_id,
                            loop_id=loop_id,
                            folder_name=folder_name,
                        )
                        folder_id = folder["id"]
                    except DotloopAPIError as e:
                        errors.append(f"Failed to create folder {folder_name}: {e.message}")
                        print(f"   ✗ Error creating folder {folder_name}: {e.message}")
                        continue
                    
                    # Upload document
                    # NOTE: This assumes you have the individual split PDFs saved
                    # You may need to adjust based on how your splitter saves files
                    raw_pdf_path = state.get("raw_pdf_path")
                    
                    if raw_pdf_path and Path(raw_pdf_path).exists():
                        try:
                            # For now, upload the original PDF to each folder
                            # In production, you'd upload the specific split document
                            file_name = f"{doc_type}.pdf"
                            
                            client.upload_document(
                                profile_id=profile_id,
                                loop_id=loop_id,
                                folder_id=folder_id,
                                file_path=raw_pdf_path,
                                file_name=file_name,
                            )
                            uploaded_count += 1
                            print(f"   ✓ Uploaded {file_name} to {folder_name}")
                        except DotloopAPIError as e:
                            errors.append(f"Failed to upload {doc_type}: {e.message}")
                            print(f"   ✗ Error uploading {doc_type}: {e.message}")
                        except FileNotFoundError as e:
                            errors.append(f"File not found for upload: {e}")
                            print(f"   ✗ File not found: {e}")
                
                print(f"   Uploaded {uploaded_count}/{len(split_docs)} documents")
            else:
                print("   No documents to upload")
            
            # ================================================================
            # STEP 6: Rate Limit Check
            # ================================================================
            if client.rate_limit_remaining:
                print(f"   Rate limit: {client.rate_limit_remaining} requests remaining")
                if int(client.rate_limit_remaining) < 10:
                    print(f"   ⚠ Warning: Low rate limit remaining!")
            
            # ================================================================
            # SUCCESS
            # ================================================================
            print(f"   ✓ Sync complete: {sync_action}")
            
            return {
                "status": "Synced" if not errors else "Partial",
                "dotloop_loop_id": str(loop_id),
                "dotloop_loop_url": loop_url,
                "sync_errors": errors,
                "dotloop_sync_action": sync_action,
            }

    except DotloopAPIError as exc:
        # Handle rate limiting specially
        if exc.status_code == 429:
            errors.append(f"Rate limit exceeded. Reset in {exc.response.get('reset_ms', 'unknown')}ms")
        else:
            errors.append(f"Dotloop API error ({exc.status_code}): {exc.message}")
        
        print(f"   ✗ Dotloop API Error: {exc}")
        
        return {
            "status": "Failed",
            "dotloop_loop_id": loop_id,
            "dotloop_loop_url": loop_url,
            "sync_errors": errors,
        }
    
    except Exception as exc:
        errors.append(f"Unexpected error: {str(exc)}")
        print(f"   ✗ Unexpected Error: {exc}")
        
        return {
            "status": "Failed",
            "dotloop_loop_id": loop_id,
            "dotloop_loop_url": loop_url,
            "sync_errors": errors,
        }
