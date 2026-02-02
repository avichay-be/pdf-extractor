import os
import argparse
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_file(
    session: aiohttp.ClientSession,
    file_path: Path,
    output_dir: Path,
    api_url: str,
    api_key: str,
    query: str = "04_Banks_Statments"
):
    """
    Process a single PDF file: send to API and save result.
    """
    filename = file_path.name
    logger.info(f"Processing {filename}...")
    
    try:
        # Prepare the file for upload
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=filename, content_type='application/pdf')
            
            # Add query parameter if needed
            url = f"{api_url}/extract?query={query}"
            
            headers = {
                "x-api-key": api_key
            }
            
            async with session.post(url, data=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to process {filename}: Status {response.status}, Error: {error_text}")
                    return False
                
                # Check content type to determine if it's a zip or markdown
                content_type = response.headers.get('Content-Type', '')
                
                if 'application/zip' in content_type:
                    # Save as zip
                    output_file = output_dir / f"{file_path.stem}_sections.zip"
                    with open(output_file, 'wb') as out_f:
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            out_f.write(chunk)
                    logger.info(f"Saved ZIP result for {filename} to {output_file}")
                    
                else:
                    # Save as markdown
                    content = await response.text()
                    output_file = output_dir / f"{file_path.stem}.md"
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.write(content)
                    logger.info(f"Saved Markdown result for {filename} to {output_file}")
                    
                return True

    except Exception as e:
        logger.error(f"Exception processing {filename}: {str(e)}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Process a folder of PDFs using the Extraction API")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Directory to save output MD/ZIP files")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--api-key", required=True, help="API Key for authentication")
    parser.add_argument("--query", default="דוחות כספיים", help="Query to filter sections (default: 'דוחות כספיים')")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    
    args = parser.parse_args()
    
    input_dir = "data/bank_statements/"
    output_dir = "out/bank_state/"
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
        
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    # Process files with semaphore for concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async with aiohttp.ClientSession() as session:
        async def bound_process(file_path):
            async with semaphore:
                return await process_file(session, file_path, output_dir, args.api_url, args.api_key, args.query)
        
        tasks = [bound_process(pdf) for pdf in pdf_files]
        results = await asyncio.gather(*tasks)
        
    success_count = sum(results)
    logger.info(f"Processing complete. Successfully processed {success_count}/{len(pdf_files)} files.")

if __name__ == "__main__":
    asyncio.run(main())
