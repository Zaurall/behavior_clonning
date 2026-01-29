from typing import Optional
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

class GDriveManager:
    def __init__(self, root_folder_name: str, settings_file: str = "settings.yaml"):
        # Configure GoogleAuth
        self.gauth = GoogleAuth(settings_file=settings_file)
        
        # Try to load saved credentials
        self.gauth.LoadCredentialsFile(settings_file)
        
        if self.gauth.credentials is None:
            # Authenticate via browser if no credentials (first run)
            # This might require user interaction
            print("Please authenticate via browser...")
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # Refresh token if expired
            self.gauth.Refresh()
        else:
            # Initialize the saved credentials
            self.gauth.Authorize()
            
        # Save credentials back to file
        self.gauth.SaveCredentialsFile(settings_file)
        
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
            self.sync_file_cache()
        return filename in self.file_cache

    def upload_file(self, local_path: Path, filename: Optional[str] = None):
        if filename is None:
            filename = local_path.name
            
        file_metadata = {
            'title': filename, 
            'parents': [{'id': self.root_id}]
        }
        
        # Check if file already exists (update it)
        if self.file_exists(filename):
             file_id = self.file_cache[filename]
             drive_file = self.drive.CreateFile({'id': file_id})
        else:
             drive_file = self.drive.CreateFile(file_metadata)
             
        drive_file.SetContentFile(str(local_path))
        try:
            drive_file.Upload()
            # Update cache
            self.file_cache[filename] = drive_file['id']
        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

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
