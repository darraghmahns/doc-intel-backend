#!/usr/bin/env python3
"""
OAuth2 authentication flow for Dotloop.
Run this to get an access token.
"""
import os
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("DOTLOOP_CLIENT_ID")
CLIENT_SECRET = os.getenv("DOTLOOP_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/callback"

# This will store the authorization code
auth_code = None


class CallbackHandler(BaseHTTPRequestHandler):
    """Handle the OAuth callback"""
    
    def do_GET(self):
        global auth_code
        
        # Parse the callback URL
        query = urlparse(self.path).query
        params = parse_qs(query)
        
        if 'code' in params:
            auth_code = params['code'][0]
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
                <html>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: green;">Authorization Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            # Error
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            error = params.get('error', ['Unknown error'])[0]
            html = f"""
                <html>
                <body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1 style="color: red;">Authorization Failed</h1>
                    <p>Error: {error}</p>
                </body>
                </html>
            """
            self.wfile.write(html.encode('utf-8'))
    
    def log_message(self, format, *args):
        # Suppress logging
        pass


def get_access_token():
    """Complete OAuth2 flow and get access token"""
    
    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ ERROR: DOTLOOP_CLIENT_ID and DOTLOOP_CLIENT_SECRET not found in .env")
        print("   Make sure you added your OAuth credentials from Dotloop")
        return None
    
    print("=" * 60)
    print("DOTLOOP OAUTH2 AUTHENTICATION")
    print("=" * 60)
    print()
    
    # Step 1: Build authorization URL
    # Dotloop uses simple scopes - try without wildcards
    auth_url = (
        f"https://auth.dotloop.com/oauth/authorize"
        f"?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
    )
    
    print("Step 1: Opening browser for authorization...")
    print(f"   If browser doesn't open, visit: {auth_url}")
    print()
    
    # Open browser
    webbrowser.open(auth_url)
    
    # Step 2: Start local server to receive callback
    print("Step 2: Waiting for authorization callback...")
    print("   (A local server is running on port 8000)")
    print()
    
    server = HTTPServer(('localhost', 8000), CallbackHandler)
    server.handle_request()  # Handle one request then stop
    
    if not auth_code:
        print("❌ Failed to get authorization code")
        return None
    
    print("✓ Authorization code received!")
    print()
    
    # Step 3: Exchange code for access token
    print("Step 3: Exchanging code for access token...")
    
    token_url = "https://auth.dotloop.com/oauth/token"
    
    # Dotloop requires Basic Auth with client credentials
    auth = (CLIENT_ID, CLIENT_SECRET)
    
    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": REDIRECT_URI,
    }
    
    try:
        response = httpx.post(token_url, data=data, auth=auth)
        response.raise_for_status()
        token_data = response.json()
        
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in")
        
        print("✓ Access token received!")
        print()
        print("=" * 60)
        print("SUCCESS! Add these to your .env file:")
        print("=" * 60)
        print()
        print(f"DOTLOOP_API_TOKEN={access_token}")
        print(f"DOTLOOP_REFRESH_TOKEN={refresh_token}")
        print()
        print(f"Note: This token expires in {expires_in} seconds (~{expires_in//3600} hours)")
        print()
        
        # Update .env file
        update_env = input("Update .env file automatically? (y/n): ").lower()
        if update_env == 'y':
            with open('.env', 'r') as f:
                env_content = f.read()
            
            # Replace or add tokens
            if 'DOTLOOP_API_TOKEN=' in env_content:
                # Replace existing token
                lines = env_content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith('DOTLOOP_API_TOKEN='):
                        new_lines.append(f'DOTLOOP_API_TOKEN={access_token}')
                    elif line.startswith('DOTLOOP_REFRESH_TOKEN='):
                        continue  # Skip old refresh token
                    else:
                        new_lines.append(line)
                
                # Add refresh token after API token
                for i, line in enumerate(new_lines):
                    if line.startswith('DOTLOOP_API_TOKEN='):
                        new_lines.insert(i + 1, f'DOTLOOP_REFRESH_TOKEN={refresh_token}')
                        break
                
                env_content = '\n'.join(new_lines)
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print("✓ .env file updated!")
            print()
        
        print("Now run: python3 test_dotloop.py")
        return access_token
        
    except httpx.HTTPError as e:
        print(f"❌ Failed to get access token: {e}")
        if hasattr(e, 'response'):
            print(f"   Response: {e.response.text}")
        return None


if __name__ == "__main__":
    get_access_token()
