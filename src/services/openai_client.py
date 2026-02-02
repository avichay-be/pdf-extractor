"""
Azure OpenAI client for PDF content cross-validation.
"""
import logging
from typing import Optional
import base64
from pathlib import Path
import sys

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from openai import AzureOpenAI
from src.core.config import settings

logger = logging.getLogger(__name__)


class OpenAIDocumentClient:
    """Client for extracting PDF content using Azure OpenAI GPT-4o."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """
        Initialize Azure OpenAI client.

        Args:
            api_key: Azure OpenAI API key (uses settings if not provided)
            endpoint: Azure OpenAI endpoint URL (uses settings if not provided)
            deployment: Deployment name (uses settings if not provided)
            api_version: API version (uses settings if not provided)
        """
        # Get credentials from settings (which loads from .env)
        self.api_key = settings.AZURE_OPENAI_API_KEY if api_key is None else api_key
        self.endpoint = settings.AZURE_OPENAI_ENDPOINT if endpoint is None else endpoint
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT if deployment is None else deployment
        self.api_version = settings.AZURE_OPENAI_API_VERSION if api_version is None else api_version

        if not self.api_key or not self.endpoint:
            raise ValueError(
                "Azure OpenAI API key and endpoint must be provided either "
                "via parameters or environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)"
            )

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )

        logger.info(f"Initialized OpenAI client with deployment: {self.deployment}")

    def _pdf_page_to_images(self, pdf_bytes: bytes, page_number: int, dpi: int = 150) -> list[str]:
        """
        Convert a single PDF page to base64-encoded images (PNG format).

        Args:
            pdf_bytes: PDF file content as bytes
            page_number: Page number (0-based) to convert
            dpi: Resolution for rendering (higher = better quality but larger size)

        Returns:
            List of base64-encoded PNG images (usually just one image per page)

        Raises:
            Exception: If conversion fails
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Check page number is valid
            if page_number >= len(pdf_document):
                raise ValueError(f"Page {page_number} does not exist (PDF has {len(pdf_document)} pages)")

            # Get the page
            page = pdf_document[page_number]

            # Render page to pixmap (image)
            # zoom factor: higher = better quality, dpi/72 is a good scaling factor
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)

            # Convert pixmap to PNG bytes
            png_bytes = pixmap.tobytes("png")

            # Encode to base64
            base64_image = base64.b64encode(png_bytes).decode('utf-8')

            # Close PDF
            pdf_document.close()

            return [base64_image]

        except Exception as e:
            logger.error(f"Failed to convert page {page_number} to image: {e}")
            raise

    def _is_responses_api(self) -> bool:
        """
        Determine if we should use the Responses API based on API version.

        Returns:
            True if using Responses API (2025-02-01-preview or later), False otherwise
        """
        if not self.api_version:
            return False

        # Responses API was introduced in 2025-02-01-preview
        # API versions format: YYYY-MM-DD or YYYY-MM-DD-preview
        try:
            # Extract date portion
            version_date = self.api_version.split('-preview')[0]
            year, month, day = map(int, version_date.split('-'))
            version_num = year * 10000 + month * 100 + day

            # Responses API threshold: 2025-02-01
            responses_api_threshold = 2025 * 10000 + 2 * 100 + 1

            return version_num >= responses_api_threshold
        except (ValueError, IndexError):
            logger.warning(f"Could not parse API version {self.api_version}, defaulting to Chat Completions API")
            return False

    def extract_page_content(
        self,
        pdf_bytes: bytes,
        page_number: int,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt_template: Optional[str] = None
    ) -> str:
        """
        Extract markdown content from a single PDF page using Azure OpenAI.

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
            logger.info(f"Extracting page {page_number} with OpenAI")

            # Convert PDF page to base64 images
            logger.debug(f"Converting page {page_number} to image...")
            base64_images = self._pdf_page_to_images(pdf_bytes, page_number)
            logger.debug(f"Page converted to {len(base64_images)} image(s)")

            # Determine which API to use
            use_responses_api = self._is_responses_api()
            logger.debug(f"Using {'Responses' if use_responses_api else 'Chat Completions'} API")

            if use_responses_api:
                # Use Responses API format (2025-08-07-preview and later)
                return self._extract_with_responses_api(base64_images, page_number, custom_system_prompt, custom_user_prompt_template)
            else:
                # Use Chat Completions API format (standard)
                return self._extract_with_chat_api(base64_images, page_number, custom_system_prompt, custom_user_prompt_template)

        except Exception as e:
            logger.error(f"Failed to extract page {page_number} with OpenAI: {e}")
            raise

    def _extract_with_chat_api(
        self,
        base64_images: list[str],
        page_number: int,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt_template: Optional[str] = None
    ) -> str:
        """
        Extract content using Chat Completions API (standard GPT-4o with vision).

        Args:
            base64_images: List of base64-encoded PNG images
            page_number: Page number (for logging)
            custom_system_prompt: Optional custom system prompt
            custom_user_prompt_template: Optional custom user prompt template

        Returns:
            Extracted markdown content
        """
        # Use custom prompts if provided, otherwise use settings
        system_prompt = custom_system_prompt or settings.get_system_prompt("openai")
        user_template = custom_user_prompt_template or settings.get_user_prompt_template("openai")
        user_prompt = user_template.format(page_number=page_number + 1)

        message_content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]

        # Add all images to the message
        for base64_image in base64_images:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })

        # Call GPT-4o with vision using Chat Completions API
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            temperature=0.0,  # Deterministic output for extraction
            max_tokens=4096  # Allow long responses for full page content
        )

        # Extract content from response
        content = response.choices[0].message.content.strip()
        logger.info(f"Successfully extracted page {page_number} ({len(content)} chars)")

        return content

    def _extract_with_responses_api(
        self,
        base64_images: list[str],
        page_number: int,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt_template: Optional[str] = None
    ) -> str:
        """
        Extract content using Responses API (2025-02-01-preview and later).

        Args:
            base64_images: List of base64-encoded PNG images
            page_number: Page number (for logging)
            custom_system_prompt: Optional custom system prompt
            custom_user_prompt_template: Optional custom user prompt template

        Returns:
            Extracted markdown content
        """
        # Use custom prompts if provided, otherwise use settings
        system_prompt = custom_system_prompt or settings.get_system_prompt("openai")
        user_template = custom_user_prompt_template or settings.get_user_prompt_template("openai")
        user_prompt = user_template.format(page_number=page_number + 1)

        # Build input content with images (Responses API format)
        # Supported types: 'input_text', 'input_image', 'output_text', 'refusal',
        # 'input_file', 'computer_screenshot', 'summary_text', 'tether_browsing_display'
        user_content = []

        # Add the text instruction first
        user_content.append({
            "type": "input_text",
            "text": user_prompt
        })

        # Add all images using input_image type
        for base64_image in base64_images:
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}"
            })

        # Call using Responses API (uses 'responses' endpoint, not 'chat.completions')
        response = self.client.responses.create(
            model=self.deployment,
            input=[  # Note: Responses API uses 'input' instead of 'messages'
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            # Note: temperature parameter is not supported in Responses API
        )

        # Extract content from response (Responses API format)
        try:
            # Responses API uses 'output_text' field
            content = response.output_text.strip()
        except (AttributeError, KeyError):
            # Fallback: try standard format
            try:
                content = response.choices[0].message.content.strip()
            except (AttributeError, KeyError, IndexError):
                # Fallback: get the raw response
                content = str(response)
                logger.warning(f"Using fallback content extraction for Responses API: {content[:100]}...")

        logger.info(f"Successfully extracted page {page_number} ({len(content)} chars)")

        return content

    def extract_from_image(self, base64_image: str, prompt: str) -> str:
        """
        Extract content from a single base64 image using Azure OpenAI.

        Args:
            base64_image: Base64-encoded image string
            prompt: User prompt for extraction

        Returns:
            Extracted text content
        """
        try:
            logger.info("Extracting content from image with OpenAI")

            # Determine which API to use
            use_responses_api = self._is_responses_api()
            
            if use_responses_api:
                return self._extract_image_with_responses_api(base64_image, prompt)
            else:
                return self._extract_image_with_chat_api(base64_image, prompt)

        except Exception as e:
            logger.error(f"Failed to extract from image with OpenAI: {e}")
            raise

    def _extract_image_with_chat_api(self, base64_image: str, prompt: str) -> str:
        """Extract from image using Chat Completions API."""
        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": settings.get_system_prompt("openai")},
                {"role": "user", "content": message_content}
            ],
            temperature=0.0,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()

    def _extract_image_with_responses_api(self, base64_image: str, prompt: str) -> str:
        """Extract from image using Responses API."""
        user_content = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"}
        ]

        response = self.client.responses.create(
            model=self.deployment,
            input=[
                {"role": "system", "content": settings.get_system_prompt("openai")},
                {"role": "user", "content": user_content}
            ]
        )
        
        try:
            return response.output_text.strip()
        except (AttributeError, KeyError):
            try:
                return response.choices[0].message.content.strip()
            except (AttributeError, KeyError, IndexError):
                return str(response)


if __name__ == "__main__":
    # Simple test of the OpenAI client
    print("Testing Azure OpenAI Responses API...")
    client = OpenAIDocumentClient()

    # Perform a test extraction on a sample PDF file
    sample_pdf_path = Path(__file__).parent.parent / "data" / "sample.pdf"
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    response = client.extract_page_content(pdf_bytes, page_number=0)

    with open("test_output.md", "w", encoding="utf-8") as out_file:
        out_file.write(response)
        
    print("✅ Azure OpenAI client test successful!")
