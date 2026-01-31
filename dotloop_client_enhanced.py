"""
Enhanced Dotloop API Client with full v2 API support.
Includes: Loops, Loop Details, Participants, Folders, Documents, Templates, Profiles
"""
import httpx
import os
from typing import Any, Dict, List, Optional, BinaryIO
from pathlib import Path


class DotloopAPIError(Exception):
    """Custom exception for Dotloop API errors"""
    def __init__(self, status_code: int, message: str, response: Dict = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"Dotloop API Error {status_code}: {message}")


class DotloopClient:
    """
    Enhanced Dotloop API v2 Client
    
    Supports:
    - OAuth2 and Bearer token authentication
    - Loop management (create, update, get, find)
    - Loop details (property, financials, dates)
    - Participants management
    - Folder management
    - Document upload/download
    - Templates
    - Profiles
    - Rate limiting awareness
    """

    def __init__(
        self,
        api_token: str = None,
        base_url: str = "https://api-gateway.dotloop.com/public/v2",
        timeout: float = 30.0,
    ):
        """
        Initialize Dotloop client.
        
        Args:
            api_token: Bearer token for authentication
            base_url: Base URL for Dotloop API (default: production)
            timeout: Request timeout in seconds
        """
        if not api_token:
            api_token = os.getenv("DOTLOOP_API_TOKEN")
        
        if not api_token:
            raise ValueError("Dotloop API token is required")
        
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        
        # Rate limit tracking
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def close(self) -> None:
        """Close the HTTP client"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _handle_response(self, resp: httpx.Response) -> Dict[str, Any]:
        """
        Handle API response and extract rate limit info.
        
        Raises:
            DotloopAPIError: If response status is not 2xx
        """
        # Track rate limits
        self.rate_limit_remaining = resp.headers.get("X-RateLimit-Remaining")
        self.rate_limit_reset = resp.headers.get("X-RateLimit-Reset")
        
        if resp.status_code == 429:
            raise DotloopAPIError(
                429,
                "Rate limit exceeded. Wait before retrying.",
                {"reset_ms": self.rate_limit_reset}
            )
        
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                error_data = resp.json()
            except:
                error_data = {"detail": resp.text}
            
            raise DotloopAPIError(
                resp.status_code,
                error_data.get("detail", str(e)),
                error_data
            )
        
        if resp.status_code == 204:  # No content
            return {}
        
        return resp.json()

    # ========================================================================
    # PROFILE MANAGEMENT
    # ========================================================================

    def list_profiles(self) -> List[Dict[str, Any]]:
        """
        List all profiles associated with the authenticated user.
        
        Returns:
            List of profile objects
        """
        resp = self._client.get("/profile")
        data = self._handle_response(resp)
        return data.get("data", [])

    def get_profile(self, profile_id: int) -> Dict[str, Any]:
        """Get a specific profile by ID"""
        resp = self._client.get(f"/profile/{profile_id}")
        data = self._handle_response(resp)
        return data.get("data", {})

    # ========================================================================
    # LOOP MANAGEMENT
    # ========================================================================

    def list_loops(
        self,
        profile_id: int,
        batch_size: int = 20,
        batch_number: int = 1,
        sort: str = "updated:desc",
        filter_updated_min: str = None,
        transaction_type: str = None,
        include_details: bool = False,
    ) -> Dict[str, Any]:
        """
        List loops for a profile with optional filtering.
        
        Args:
            profile_id: The profile ID
            batch_size: Results per page (max 100)
            batch_number: Page number
            sort: Sort order (e.g., "updated:desc", "created:asc")
            filter_updated_min: ISO timestamp for filtering
            transaction_type: Filter by type (e.g., "PURCHASE_OFFER")
            include_details: Include full loop details
        
        Returns:
            Dict with 'meta' and 'data' keys
        """
        params = {
            "batch_size": min(batch_size, 100),
            "batch_number": batch_number,
            "sort": sort,
            "include_details": str(include_details).lower(),
        }
        
        if filter_updated_min:
            params["filter"] = f"updated_min={filter_updated_min}"
        if transaction_type:
            params["filter"] = f"transaction_type={transaction_type}"
        
        resp = self._client.get(f"/profile/{profile_id}/loop", params=params)
        return self._handle_response(resp)

    def find_existing_loop(self, profile_id: int, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a loop by name (searches recent loops).
        
        Args:
            profile_id: The profile ID
            name: Loop name to search for
        
        Returns:
            Loop object if found, None otherwise
        """
        result = self.list_loops(profile_id, batch_size=50, sort="updated:desc")
        loops = result.get("data", [])
        
        for loop in loops:
            if loop.get("name") == name:
                return loop
        
        return None

    def get_loop(self, profile_id: int, loop_id: int) -> Dict[str, Any]:
        """Get a specific loop by ID"""
        resp = self._client.get(f"/profile/{profile_id}/loop/{loop_id}")
        data = self._handle_response(resp)
        return data.get("data", {})

    def create_loop(
        self,
        profile_id: int,
        name: str,
        transaction_type: str = "PURCHASE_OFFER",
        status: str = "PRE_OFFER",
    ) -> Dict[str, Any]:
        """
        Create a new loop.
        
        Args:
            profile_id: The profile ID
            name: Loop name (usually property address or buyer name)
            transaction_type: Type of transaction
            status: Initial status
        
        Returns:
            Created loop object
        """
        payload = {
            "name": name,
            "transactionType": transaction_type,
            "status": status,
        }
        
        resp = self._client.post(f"/profile/{profile_id}/loop", json=payload)
        data = self._handle_response(resp)
        return data.get("data", {})

    def update_loop(
        self,
        profile_id: int,
        loop_id: int,
        name: str = None,
        transaction_type: str = None,
        status: str = None,
    ) -> Dict[str, Any]:
        """
        Update an existing loop (partial updates supported).
        
        Args:
            profile_id: The profile ID
            loop_id: The loop ID
            name: New name (optional)
            transaction_type: New transaction type (optional)
            status: New status (optional)
        
        Returns:
            Updated loop object
        """
        payload = {}
        if name is not None:
            payload["name"] = name
        if transaction_type is not None:
            payload["transactionType"] = transaction_type
        if status is not None:
            payload["status"] = status
        
        resp = self._client.patch(f"/profile/{profile_id}/loop/{loop_id}", json=payload)
        data = self._handle_response(resp)
        return data.get("data", {})

    # ========================================================================
    # LOOP-IT (SIMPLIFIED LOOP CREATION)
    # ========================================================================

    def loop_it(
        self,
        profile_id: int,
        name: str,
        transaction_type: str,
        status: str,
        street_name: str = None,
        street_number: str = None,
        city: str = None,
        state: str = None,
        zip_code: str = None,
        participants: List[Dict] = None,
        template_id: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a loop with property and participants in one call (Loop-It API).
        
        This is a facade API that creates the loop, adds participants to contacts,
        and populates property details all at once.
        
        Args:
            profile_id: The profile ID
            name: Loop name
            transaction_type: Type of transaction
            status: Initial status
            street_name, street_number, city, state, zip_code: Property address
            participants: List of participant dicts with fullName, email, role
            template_id: Loop template ID (may be required)
            **kwargs: Additional fields (unit, county, country, mlsPropertyId, etc.)
        
        Returns:
            Created loop with loopUrl
        """
        payload = {
            "name": name,
            "transactionType": transaction_type,
            "status": status,
        }
        
        # Property address
        if street_name:
            payload["streetName"] = street_name
        if street_number:
            payload["streetNumber"] = street_number
        if city:
            payload["city"] = city
        if state:
            payload["state"] = state
        if zip_code:
            payload["zipCode"] = zip_code
        
        # Participants
        if participants:
            payload["participants"] = participants
        
        # Template
        if template_id:
            payload["templateId"] = template_id
        
        # Additional fields
        payload.update(kwargs)
        
        resp = self._client.post(f"/loop-it?profile_id={profile_id}", json=payload)
        data = self._handle_response(resp)
        return data.get("data", {})

    # ========================================================================
    # LOOP DETAILS
    # ========================================================================

    def get_loop_details(self, profile_id: int, loop_id: int) -> Dict[str, Any]:
        """
        Get detailed loop information (property, financials, dates, etc.).
        
        Returns:
            Dict with sections like "Property Address", "Financials", "Contract Dates"
        """
        resp = self._client.get(f"/profile/{profile_id}/loop/{loop_id}/detail")
        data = self._handle_response(resp)
        return data.get("data", {})

    def update_loop_details(
        self,
        profile_id: int,
        loop_id: int,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update loop details (partial updates supported).
        
        Args:
            profile_id: The profile ID
            loop_id: The loop ID
            details: Dict with sections like:
                {
                    "Property Address": {"Street Name": "Main", ...},
                    "Financials": {"Purchase/Sale Price": "500000", ...},
                    "Contract Dates": {"Closing Date": "12/31/2024", ...}
                }
        
        Returns:
            Updated loop details
        """
        resp = self._client.patch(
            f"/profile/{profile_id}/loop/{loop_id}/detail",
            json=details
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    # ========================================================================
    # PARTICIPANTS
    # ========================================================================

    def list_participants(self, profile_id: int, loop_id: int) -> List[Dict[str, Any]]:
        """List all participants in a loop"""
        resp = self._client.get(f"/profile/{profile_id}/loop/{loop_id}/participant")
        data = self._handle_response(resp)
        return data.get("data", [])

    def get_participant(
        self, profile_id: int, loop_id: int, participant_id: int
    ) -> Dict[str, Any]:
        """Get a specific participant"""
        resp = self._client.get(
            f"/profile/{profile_id}/loop/{loop_id}/participant/{participant_id}"
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    def add_participant(
        self,
        profile_id: int,
        loop_id: int,
        full_name: str,
        email: str,
        role: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add a participant to a loop.
        
        Args:
            profile_id: The profile ID
            loop_id: The loop ID
            full_name: Participant's full name
            email: Participant's email
            role: Participant role (BUYER, SELLER, LISTING_AGENT, etc.)
            **kwargs: Optional fields (Phone, Company Name, License #, address fields)
        
        Returns:
            Created participant object
        """
        payload = {
            "fullName": full_name,
            "email": email,
            "role": role,
        }
        payload.update(kwargs)
        
        resp = self._client.post(
            f"/profile/{profile_id}/loop/{loop_id}/participant",
            json=payload
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    def update_participant(
        self,
        profile_id: int,
        loop_id: int,
        participant_id: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Update a participant (partial updates supported).
        
        Args:
            participant_id: The participant ID
            **kwargs: Fields to update
        """
        resp = self._client.patch(
            f"/profile/{profile_id}/loop/{loop_id}/participant/{participant_id}",
            json=kwargs
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    def delete_participant(
        self, profile_id: int, loop_id: int, participant_id: int
    ) -> None:
        """Delete a participant from a loop"""
        resp = self._client.delete(
            f"/profile/{profile_id}/loop/{loop_id}/participant/{participant_id}"
        )
        self._handle_response(resp)

    # ========================================================================
    # FOLDERS
    # ========================================================================

    def list_folders(
        self,
        profile_id: int,
        loop_id: int,
        include_documents: bool = False,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all folders in a loop.
        
        Args:
            include_documents: Include document list in each folder
            include_archived: Include archived folders/documents
        """
        params = {
            "include_documents": str(include_documents).lower(),
            "include_archived": str(include_archived).lower(),
        }
        resp = self._client.get(
            f"/profile/{profile_id}/loop/{loop_id}/folder",
            params=params
        )
        data = self._handle_response(resp)
        return data.get("data", [])

    def get_folder(
        self,
        profile_id: int,
        loop_id: int,
        folder_id: int,
        include_documents: bool = True,
    ) -> Dict[str, Any]:
        """Get a specific folder"""
        params = {"include_documents": str(include_documents).lower()}
        resp = self._client.get(
            f"/profile/{profile_id}/loop/{loop_id}/folder/{folder_id}",
            params=params
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    def create_folder(
        self, profile_id: int, loop_id: int, name: str
    ) -> Dict[str, Any]:
        """Create a new folder in a loop"""
        payload = {"name": name}
        resp = self._client.post(
            f"/profile/{profile_id}/loop/{loop_id}/folder",
            json=payload
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    def update_folder(
        self, profile_id: int, loop_id: int, folder_id: int, name: str
    ) -> Dict[str, Any]:
        """Rename a folder"""
        payload = {"name": name}
        resp = self._client.patch(
            f"/profile/{profile_id}/loop/{loop_id}/folder/{folder_id}",
            json=payload
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    def find_or_create_folder(
        self, profile_id: int, loop_id: int, folder_name: str
    ) -> Dict[str, Any]:
        """
        Find a folder by name or create it if it doesn't exist.
        
        Returns:
            Folder object with 'id' key
        """
        folders = self.list_folders(profile_id, loop_id)
        
        for folder in folders:
            if folder.get("name") == folder_name:
                return folder
        
        # Create if not found
        return self.create_folder(profile_id, loop_id, folder_name)

    # ========================================================================
    # DOCUMENTS
    # ========================================================================

    def list_documents(
        self, profile_id: int, loop_id: int, folder_id: int
    ) -> List[Dict[str, Any]]:
        """List all documents in a folder"""
        resp = self._client.get(
            f"/profile/{profile_id}/loop/{loop_id}/folder/{folder_id}/document"
        )
        data = self._handle_response(resp)
        return data.get("data", [])

    def get_document(
        self, profile_id: int, loop_id: int, folder_id: int, document_id: int
    ) -> Dict[str, Any]:
        """Get document metadata"""
        resp = self._client.get(
            f"/profile/{profile_id}/loop/{loop_id}/folder/{folder_id}/document/{document_id}"
        )
        data = self._handle_response(resp)
        return data.get("data", {})

    def upload_document(
        self,
        profile_id: int,
        loop_id: int,
        folder_id: int,
        file_path: str,
        file_name: str = None,
    ) -> Dict[str, Any]:
        """
        Upload a document to a folder.
        
        Args:
            profile_id: The profile ID
            loop_id: The loop ID
            folder_id: The folder ID
            file_path: Path to the PDF file
            file_name: Optional custom filename (uses file_path name if not provided)
        
        Returns:
            Created document object
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_name is None:
            file_name = path.name
        
        # Dotloop requires multipart/form-data with specific format
        with open(file_path, "rb") as f:
            files = {
                "file": (file_name, f, "application/pdf")
            }
            
            # Need to create a new client without JSON content-type for multipart
            upload_client = httpx.Client(
                base_url=self._client.base_url,
                headers={
                    "Authorization": self._client.headers["Authorization"],
                    "Accept": "application/json",
                },
                timeout=self._client.timeout,
            )
            
            try:
                resp = upload_client.post(
                    f"/profile/{profile_id}/loop/{loop_id}/folder/{folder_id}/document",
                    files=files
                )
                data = self._handle_response(resp)
                return data.get("data", {})
            finally:
                upload_client.close()

    # ========================================================================
    # TEMPLATES
    # ========================================================================

    def list_templates(self, profile_id: int) -> List[Dict[str, Any]]:
        """List all loop templates for a profile"""
        resp = self._client.get(f"/profile/{profile_id}/loop-template")
        data = self._handle_response(resp)
        return data.get("data", [])

    def get_template(self, profile_id: int, template_id: int) -> Dict[str, Any]:
        """Get a specific template"""
        resp = self._client.get(f"/profile/{profile_id}/loop-template/{template_id}")
        data = self._handle_response(resp)
        return data.get("data", {})

    # ========================================================================
    # CONTACTS
    # ========================================================================

    def list_contacts(
        self, batch_size: int = 20, batch_number: int = 1
    ) -> Dict[str, Any]:
        """List all contacts in the user's directory"""
        params = {"batch_size": min(batch_size, 100), "batch_number": batch_number}
        resp = self._client.get("/contact", params=params)
        return self._handle_response(resp)

    def get_contact(self, contact_id: int) -> Dict[str, Any]:
        """Get a specific contact"""
        resp = self._client.get(f"/contact/{contact_id}")
        data = self._handle_response(resp)
        return data.get("data", {})

    def create_contact(self, **kwargs) -> Dict[str, Any]:
        """
        Create a new contact.
        
        Args:
            **kwargs: firstName, lastName, email, phone, address, city, state, zipCode, etc.
        """
        resp = self._client.post("/contact", json=kwargs)
        data = self._handle_response(resp)
        return data.get("data", {})
