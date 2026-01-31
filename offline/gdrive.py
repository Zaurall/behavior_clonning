from typing import Optional
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

class GDriveManager:
    def __init__(self, root_folder_name: str):
        # Calculate paths relative to this script file
        # script is in behavior_clonner/offline/
        # secrets are in behavior_clonner/
        base_dir = Path(__file__).parent.parent.absolute()
        secrets_path = base_dir / "client_secrets.json"
        
        # We store credentials in settings.yaml in the same folder as secrets
        creds_path = base_dir / "mycreds.txt"

        if not secrets_path.exists():
             raise FileNotFoundError(f"Could not find client_secrets.json at {secrets_path}")

        # Configure GoogleAuth with explicit paths to prevent "expired" errors
        self.gauth = GoogleAuth()
        
        # FORCE the use of specific file paths so PyDrive doesn't get lost
        self.gauth.settings['client_config_backend'] = 'file'
        self.gauth.settings['client_config_file'] = str(secrets_path)
        self.gauth.settings['save_credentials'] = True
        self.gauth.settings['save_credentials_backend'] = 'file'
        self.gauth.settings['save_credentials_file'] = str(creds_path)
        self.gauth.settings['get_refresh_token'] = True

        # Try to load existing credentials
        if creds_path.exists():
            self.gauth.LoadCredentialsFile(str(creds_path))

        if self.gauth.credentials is None:
            # First run: Authenticate via browser
            print("Authenticating: Please check your browser...")
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # Explicit refresh logic
            print("Token expired, refreshing...")
            try:
                self.gauth.Refresh()
            except Exception as e:
                print(f"Token refresh failed: {e}. Re-authenticating...")
                self.gauth.LocalWebserverAuth()
        else:
            self.gauth.Authorize()
            
        # Save updates back to yaml immediately
        self.gauth.SaveCredentialsFile(str(creds_path))
        
        self.drive = GoogleDrive(self.gauth)
        self.root_folder_name = root_folder_name
        self.root_id = self._get_or_create_folder(root_folder_name)
        self.file_cache = {} # Cache for fast existence checks

    def _get_or_create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> str:
        """Finds or creates a folder by name."""
        query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        file_list = self.drive.ListFile({'q': query}).GetList()
        
        if file_list:
            return file_list[0]['id']
        else:
            folder_metadata = {
                'title': folder_name, 
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                folder_metadata['parents'] = [{'id': parent_id}]
                
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()
            return folder['id']

    def sync_file_cache(self):
        """Builds a cache of standard files in the root folder."""
        # This is optimization for checking existence of thousands of files
        query = f"'{self.root_id}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()
        self.file_cache = {f['title']: f['id'] for f in file_list}

    def file_exists(self, filename: str) -> bool:
        if not self.file_cache:
            try:
                self.sync_file_cache()
            except Exception as e:
                print(f"Warning: Failed to sync file cache: {e}")
                return False
        return filename in self.file_cache

    def upload_file(self, local_path: Path, filename: Optional[str] = None):
        if filename is None:
            filename = local_path.name
            
        import time
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                # Check if file already exists (update it)
                if self.file_exists(filename):
                     file_id = self.file_cache[filename]
                     drive_file = self.drive.CreateFile({'id': file_id})
                else:
                     file_metadata = {
                        'title': filename, 
                        'parents': [{'id': self.root_id}]
                    }
                     drive_file = self.drive.CreateFile(file_metadata)
                     
                drive_file.SetContentFile(str(local_path))
                drive_file.Upload()
                # Update cache
                self.file_cache[filename] = drive_file['id']
                return # Success
            except Exception as e:
                print(f"Failed to upload {filename} (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Check for the specific config error and try to re-inject settings
                if "Backend" in str(e) and "not configured" in str(e):
                     print("Attempting to re-configure GAuth settings...")
                     base_dir = Path(__file__).parent.parent.absolute()
                     secrets_path = base_dir / "client_secrets.json"
                     creds_path = base_dir / "mycreds.txt"
                     self.gauth.settings['client_config_backend'] = 'file'
                     self.gauth.settings['client_config_file'] = str(secrets_path)
                     self.gauth.settings['save_credentials_backend'] = 'file'
                     self.gauth.settings['save_credentials_file'] = str(creds_path)
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Reraise the last exception so it can be logged by caller if needed
                    # But run.py just prints it, which is fine.
                    print(f"Giving up on {filename}")
                    raise e

    def download_file(self, filename: str, local_path: Path):
        """Downloads a file from GDrive to local path."""
        if not self.file_exists(filename):
            raise FileNotFoundError(f"File {filename} not found in GDrive folder {self.root_folder_name}")
            
        file_id = self.file_cache[filename]
        drive_file = self.drive.CreateFile({'id': file_id})
        drive_file.GetContentFile(str(local_path))

    def sync_folder_down(self, local_folder: Path):
        """Downloads all files from GDrive folder to local folder if they don't exist locally."""
        if not self.file_cache:
            self.sync_file_cache()
            
        local_folder.mkdir(parents=True, exist_ok=True)
        
        for filename, file_id in self.file_cache.items():
            # Check filtering if needed, e.g. only .pt
             local_file_path = local_folder / filename
             if not local_file_path.exists():
                 print(f"Downloading {filename}...")
                 try:
                    self.download_file(filename, local_file_path)
                 except Exception as e:
                    print(f"Failed to download {filename}: {e}")