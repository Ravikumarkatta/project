import os
import json
import requests
from typing import List, Dict, Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class BibleHandler:
    def __init__(self, download_dir: str = "downloads", data_dir: str = "data/raw/bibles"):
        self.download_dir = download_dir
        self.data_dir = data_dir
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

    async def download_bible_version(self, version: str) -> str:
        """
        Download a specific Bible version from known sources.
        Returns the path to the downloaded file.
        """
        version = version.lower()
        sources = {
            "kjv": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
            "asv": "https://ebible.org/asv/eng-asv_usfx.xml",
            "web": "https://ebible.org/web/engwebp_usfx.xml"
        }
        
        if version not in sources:
            raise HTTPException(status_code=400, detail=f"Version {version} not supported")
        
        try:
            url = sources[version]
            output_file = os.path.join(self.download_dir, f"{version}.txt")
            
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded {version} Bible to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error downloading {version} Bible: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to download {version} Bible")

    async def process_uploaded_bible(self, file_content: str, version: str) -> Dict:
        """
        Process an uploaded Bible file and store it in the raw data directory.
        Returns metadata about the processed file.
        """
        try:
            output_file = os.path.join(self.data_dir, f"{version.lower()}.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            # Basic validation of the uploaded content
            verse_count = len([line for line in file_content.split('\n') if line.strip()])
            
            metadata = {
                "version": version,
                "verse_count": verse_count,
                "file_path": output_file,
                "size_bytes": len(file_content)
            }
            
            logger.info(f"Successfully processed uploaded {version} Bible")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing uploaded Bible: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process uploaded Bible")

    async def list_available_versions(self) -> List[str]:
        """
        List all available Bible versions in the data directory.
        """
        try:
            files = os.listdir(self.data_dir)
            versions = [f.split('.')[0] for f in files if f.endswith('.txt')]
            return versions
        except Exception as e:
            logger.error(f"Error listing Bible versions: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to list Bible versions")