#!/usr/bin/env python3
"""
Quick test script for Dotloop integration.
Tests connection, gets profile info, and validates setup.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the enhanced client
try:
    from dotloop_client_enhanced import DotloopClient, DotloopAPIError
except ImportError:
    print("ERROR: Could not import dotloop_client_enhanced")
    print("Make sure you're in the backend directory")
    sys.exit(1)


def test_connection():
    """Test basic Dotloop API connection"""
    print("=" * 60)
    print("DOTLOOP INTEGRATION TEST")
    print("=" * 60)
    print()
    
    # Check for API token
    token = os.getenv("DOTLOOP_API_TOKEN")
    if not token:
        print("❌ ERROR: DOTLOOP_API_TOKEN not found in .env")
        print("   Add your token to backend/.env")
        return False
    
    print(f"✓ API Token found: {token[:20]}...")
    print()
    
    # Test connection
    try:
        client = DotloopClient(api_token=token)
        print("✓ Client initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Get profiles
    print("Fetching profiles...")
    try:
        profiles = client.list_profiles()
        print(f"✓ Found {len(profiles)} profile(s)")
        print()
        
        if not profiles:
            print("⚠ WARNING: No profiles found!")
            print("  Your Dotloop account may not be set up correctly")
            return False
        
        print("YOUR DOTLOOP PROFILES:")
        print("-" * 60)
        for p in profiles:
            default_marker = " ⭐ DEFAULT" if p.get("default") else ""
            print(f"ID: {p['id']:<10} Name: {p['name']}{default_marker}")
            print(f"            Type: {p.get('type')}, Company: {p.get('company', 'N/A')}")
            if p.get("requiresTemplate"):
                print(f"            ⚠ Requires Template!")
            print()
        
        # Check if profile_id is set
        profile_id = os.getenv("DOTLOOP_PROFILE_ID")
        if profile_id:
            print(f"✓ DOTLOOP_PROFILE_ID is set: {profile_id}")
            
            # Validate it exists
            profile_ids = [str(p['id']) for p in profiles]
            if profile_id not in profile_ids:
                print(f"⚠ WARNING: Profile ID {profile_id} not found in your profiles!")
        else:
            print("⚠ DOTLOOP_PROFILE_ID not set in .env")
            print()
            print("SETUP INSTRUCTIONS:")
            print("1. Add this line to your backend/.env file:")
            
            # Suggest default profile
            for p in profiles:
                if p.get("default"):
                    print(f"   DOTLOOP_PROFILE_ID={p['id']}")
                    break
            else:
                print(f"   DOTLOOP_PROFILE_ID={profiles[0]['id']}")
            
            print("2. Then run this test again")
        
        print()
        
        # Check for templates if required
        for p in profiles:
            if p.get("requiresTemplate"):
                print(f"Checking templates for profile {p['id']}...")
                try:
                    templates = client.list_templates(p['id'])
                    if templates:
                        print(f"✓ Found {len(templates)} template(s):")
                        for t in templates[:3]:  # Show first 3
                            print(f"   - {t.get('name')} (ID: {t['id']})")
                        print()
                        print("  If needed, add to .env:")
                        print(f"  DOTLOOP_TEMPLATE_ID={templates[0]['id']}")
                    else:
                        print("⚠ No templates found - you may need to create one in Dotloop")
                except DotloopAPIError as e:
                    print(f"⚠ Could not fetch templates: {e.message}")
                print()
        
        # Test rate limit info
        if client.rate_limit_remaining:
            print(f"Rate Limit Status: {client.rate_limit_remaining} requests remaining")
        
        print()
        print("=" * 60)
        print("✓ CONNECTION TEST SUCCESSFUL!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Make sure DOTLOOP_PROFILE_ID is set in .env")
        print("2. Update main.py to use executor_enhanced")
        print("3. Run your pipeline!")
        print()
        
        return True
        
    except DotloopAPIError as e:
        print(f"❌ API Error: {e.message}")
        print(f"   Status Code: {e.status_code}")
        if e.response:
            print(f"   Response: {e.response}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        client.close()


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
