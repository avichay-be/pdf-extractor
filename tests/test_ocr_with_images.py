import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.extraction_service import process_ocr_with_images
from src.models.mistral_models import MistralOCRResponse, Page, Dimensions, UsageInfo

@pytest.mark.asyncio
async def test_process_ocr_with_images_with_images():
    # Mock Mistral client
    with patch("src.services.extraction_service.mistral_client", new_callable=AsyncMock) as mock_mistral:
        # Mock OpenAI client
        with patch("src.services.extraction_service.openai_client") as mock_openai:
            # Setup Mistral mock response
            mock_images = [{"image_base64": "fake_base64_data", "page_index": 0}]
            mock_metadata = {"images": mock_images}
            mock_mistral.process_document.return_value = ("Mistral Content", mock_metadata)
            
            # Setup OpenAI mock response
            mock_openai.extract_from_image.return_value = "Extracted Image Content"
            
            # Run function
            content, metadata = await process_ocr_with_images(
                pdf_path="fake.pdf",
                query="test query"
            )
            
            # Verify Mistral called correctly
            mock_mistral.process_document.assert_called_once_with(
                pdf_path="fake.pdf",
                pdf_base64=None,
                include_images=True,
                enable_validation=False
            )
            
            # Verify OpenAI called correctly
            mock_openai.extract_from_image.assert_called_once_with("fake_base64_data", "test query")
            
            # Verify content includes image extraction
            assert "Mistral Content" in content
            assert "Extracted Image Content" in content
            assert "## Extracted Image Data" in content
            
            # Verify metadata
            assert metadata["workflow"] == "ocr_with_images"
            assert metadata["images_processed"] == 1
            assert metadata["prompt_used"] == "test query"

@pytest.mark.asyncio
async def test_process_ocr_with_images_no_images():
    # Mock Mistral client
    with patch("src.services.extraction_service.mistral_client", new_callable=AsyncMock) as mock_mistral:
        # Mock OpenAI client
        with patch("src.services.extraction_service.openai_client") as mock_openai:
            # Setup Mistral mock response (no images)
            mock_mistral.process_document.return_value = ("Mistral Content", {})
            
            # Run function
            content, metadata = await process_ocr_with_images(
                pdf_path="fake.pdf",
                query="test query"
            )
            
            # Verify Mistral called
            mock_mistral.process_document.assert_called_once()
            
            # Verify OpenAI NOT called
            mock_openai.extract_from_image.assert_not_called()
            
            # Verify content is just Mistral content
            assert content == "Mistral Content"
            
            # Verify metadata
            assert metadata == {}
