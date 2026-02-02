"""
Google Gemini client for PDF content cross-validation.
"""
import logging
from os import getenv
from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from google import genai
from google.genai import types
from src.core.config import settings

logger = logging.getLogger(__name__)


class GeminiDocumentClient:
    """Client for extracting PDF content using Google Gemini Flash."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key (uses settings if not provided)
            model_name: Model name (uses settings if not provided)
        """
        # Get credentials from environment variables or settings
        self.api_key = api_key or getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL

        if not self.api_key:
            raise ValueError(
                "Gemini API key must be provided either "
                "via parameters or environment variables (GEMINI_API_KEY)"
            )

        # Initialize Gemini client with API key
        self.client = genai.Client(api_key=self.api_key)

        logger.info(f"Initialized Gemini client with model: {self.model_name}")

    def _extract_single_page_pdf(self, pdf_bytes: bytes, page_number: int) -> bytes:
        """
        Extract a single page from PDF as separate PDF bytes.

        Args:
            pdf_bytes: Full PDF file content as bytes
            page_number: Page number (0-based) to extract

        Returns:
            PDF bytes containing only the specified page

        Raises:
            Exception: If extraction fails
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Check page number is valid
            if page_number >= len(pdf_document):
                raise ValueError(f"Page {page_number} does not exist (PDF has {len(pdf_document)} pages)")

            # Create a new PDF with only the target page
            single_page_pdf = fitz.open()  # Create empty PDF
            single_page_pdf.insert_pdf(pdf_document, from_page=page_number, to_page=page_number)

            # Convert to bytes
            page_pdf_bytes = single_page_pdf.tobytes()

            # Clean up
            pdf_document.close()
            single_page_pdf.close()

            return page_pdf_bytes

        except Exception as e:
            logger.error(f"Failed to extract page {page_number}: {e}")
            raise

    def extract_page_content(
        self,
        pdf_bytes: bytes,
        page_number: int,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt_template: Optional[str] = None
    ) -> str:
        """
        Extract markdown content from a single PDF page using Gemini.

        Args:
            pdf_bytes: PDF file content as bytes
            page_number: Page number (0-based) to extract
            custom_system_prompt: Optional custom system prompt (overrides settings)
            custom_user_prompt_template: Optional custom user prompt template (overrides settings)

        Returns:
            Extracted markdown content

        Raises:
            Exception: If extraction fails
        """
        try:
            logger.info(f"Extracting page {page_number} with Gemini")

            # Extract single page as PDF
            logger.debug(f"Extracting page {page_number} from PDF...")
            page_pdf_bytes = self._extract_single_page_pdf(pdf_bytes, page_number)
            logger.debug(f"Page extracted ({len(page_pdf_bytes)} bytes)")

            # Use custom prompts if provided, otherwise use settings
            system_instruction = custom_system_prompt or settings.get_system_prompt("gemini")
            user_template = custom_user_prompt_template or settings.get_user_prompt_template("gemini")
            user_prompt = user_template.format(page_number=page_number + 1)

            # Combine system and user prompts
            full_prompt = f"{system_instruction}\n\n{user_prompt}"

            # Prepare contents list with PDF and prompt
            # Gemini can handle PDF directly - no need to convert to images
            contents = [
                types.Part.from_bytes(
                    data=page_pdf_bytes,
                    mime_type='application/pdf',
                ),
                full_prompt
            ]

            # Generate content
            logger.debug("Sending request to Gemini...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )

            # Extract the text from response
            content = response.text.strip()

            logger.info(f"Successfully extracted page {page_number} ({len(content)} chars)")

            return content

        except Exception as e:
            logger.error(f"Failed to extract page {page_number} with Gemini: {e}")
            raise


if __name__ == "__main__":
    # Simple test of the Gemini client
    print("Testing Google Gemini Flash API...")

    try:
        client = GeminiDocumentClient()

        # Perform a test extraction on a sample PDF file
        sample_pdf_path = Path(__file__).parent.parent / "data" / "sample.pdf"

        if not sample_pdf_path.exists():
            print(f"❌ Sample PDF not found at {sample_pdf_path}")
            exit(1)

        with open(sample_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        response = client.extract_page_content(pdf_bytes, page_number=3)

        print("\nGemini response:")
        print("="*80)
        print(response[:500] + "..." if len(response) > 500 else response)
        print("="*80)

        # Optionally save to file
        output_path = Path(__file__).parent.parent / "test_output_gemini.md"
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(response)
        print(f"\nFull output saved to: {output_path}")

        print("\n✅ Gemini client test successful!")

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nPlease set GEMINI_API_KEY environment variable or add it to .env file")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
