"""
Configuration settings for the application.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Key Authentication
    API_KEY: Optional[str] = None  # Required for production - set in environment or .env file
    REQUIRE_API_KEY: bool = True  # Set to False to disable API key authentication (not recommended for production)

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"  # Options: "text", "json"
    LOG_INCLUDE_REQUEST_ID: bool = True  # Include X-Request-ID in logs

    # HTTP Client Configuration
    HTTP_CLIENT_TIMEOUT: float = 120.0  # Default timeout for HTTP clients (seconds)
    HTTP_CLIENT_HEALTH_CHECK_TIMEOUT: float = 10.0  # Timeout for health check requests (seconds)
    HTTP_MAX_KEEPALIVE_CONNECTIONS: int = 10  # Maximum number of keepalive connections
    HTTP_MAX_CONNECTIONS: int = 20  # Maximum total connections
    HTTP_RETRY_ATTEMPTS: int = 3  # Default retry attempts for transient errors
    HTTP_RETRY_BACKOFF_SECONDS: float = 2.0  # Base backoff for retries
    HTTP_RETRY_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)

    # Performance Monitoring
    RESPONSE_TIME_WARNING_THRESHOLD_MS: int = 30000  # Warn if requests take longer than 30s (milliseconds)

    # Mistral API Configuration
    AZURE_API_KEY: str  # Required: Must be set in environment or .env file
    MISTRAL_API_URL: str = "https://abracrm-4614-resource.services.ai.azure.com/providers/mistral/azure/ocr"
    MISTRAL_MODEL: str = "mistral-document-ai-2505"
    MAX_PAGES_PER_CHUNK: int = 15  # Increased from 10 to 15 for better performance (fewer API calls)
    INCLUDE_IMAGES: bool = False  # Set to True to include image references in output

    # Input Guardrails
    MAX_UPLOAD_MB: int = 25  # Max upload size for PDFs (uncompressed)
    MAX_BASE64_LENGTH: int = 40_000_000  # Max base64 characters (~30 MB decoded)
    MAX_PDF_PAGES: int = 600  # Hard cap to avoid runaway processing

    # Mistral API Rate Limiting
    MISTRAL_REQUESTS_PER_MINUTE: int = 50  # API limit: 60 requests per minute
    MISTRAL_MIN_REQUEST_INTERVAL: float = 1.0  # Minimum seconds between requests (60/min = 1 req/sec)
    MISTRAL_RETRY_ATTEMPTS: int = 3  # Number of retry attempts for 429 errors
    MISTRAL_RETRY_DELAY: float = 5.0  # Initial delay in seconds for exponential backoff

    # Google Gemini Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Smart Extraction Configuration
    SMART_EXTRACTION_AUTO_DETECT: bool = True  # Auto-detect mixed PDFs and route to smart extraction
    SMART_EXTRACTION_TEXT_THRESHOLD: int = 50  # Min chars to consider a page as having text
    SMART_EXTRACTION_MISTRAL_BATCH_SIZE: int = 10  # Max consecutive image pages per Mistral call
    SMART_EXTRACTION_POLISH_ENABLED: bool = True  # Enable Gemini polish pass
    SMART_EXTRACTION_POLISH_BATCH_SIZE: int = 50  # Pages per Gemini polish call
    SMART_EXTRACTION_POLISH_CONCURRENCY: int = 3  # Max parallel Gemini polish calls
    SMART_EXTRACTION_POLISH_MODEL: str = "gemini-3-1-flash-lite-preview"
    SMART_EXTRACTION_POLISH_PROMPT: Optional[str] = None  # Custom polish prompt (None = use default)
    DEFAULT_SMART_EXTRACTION_POLISH_PROMPT: str = """You are a document formatting expert. Clean up extracted PDF content into well-formatted markdown optimized for LLM consumption.

RULES:
1. Fix OCR errors (common: 0/O, 1/l, rn/m, Hebrew character confusions)
2. Normalize table formatting (consistent columns, proper alignment)
3. Ensure consistent heading hierarchy (H2 for sections, H3 for subsections)
4. Remove duplicate content at page boundaries
5. Fix encoding artifacts and garbled characters
6. Preserve ALL numerical data EXACTLY as-is (never modify numbers)
7. Preserve ALL text content (never summarize or omit)
8. Clean up excessive whitespace
9. Output ONLY the cleaned markdown - no explanations"""

    # Cross-Validation Settings
    ENABLE_CROSS_VALIDATION: bool = True
    VALIDATION_SAMPLE_RATE: int = 5  # Validate every Nth page
    VALIDATION_SIMILARITY_THRESHOLD: float = 0.95  # 95% similarity = 5% error tolerance
    VALIDATION_SIMILARITY_METHOD: str = "number_frequency"  # Options: "number_frequency", "levenshtein"
    VALIDATION_SKIP_SAMPLE_IF_CLEAN: bool = True  # Skip sample validation if no problems detected (optimization)

    # Enhanced Validation: Problem Detection (13 patterns)
    # Comma-separated list of enabled problems, or "all" to enable all
    # Available: empty_tables, low_content_density, missing_numbers, inconsistent_columns,
    #           repeated_characters, garbled_text, header_only_tables, very_short_pages,
    #           missing_keywords, malformed_structure, duplicate_content, unknown_characters,
    #           repetitive_numbers
    VALIDATION_PROBLEMS_ENABLED: str = "empty_tables,low_content_density,missing_numbers,inconsistent_columns,garbled_text,missing_keywords,repetitive_numbers"

    @property
    def validation_problems_list(self) -> List[str]:
        """Parse comma-separated validation problems into list."""
        if self.VALIDATION_PROBLEMS_ENABLED.lower() == "all":
            return [
                'empty_tables', 'low_content_density', 'missing_numbers',
                'inconsistent_columns', 'repeated_characters', 'garbled_text',
                'header_only_tables', 'very_short_pages', 'missing_keywords',
                'malformed_structure', 'duplicate_content', 'unknown_characters',
                'repetitive_numbers', 'markdown_images'
            ]
        return [p.strip() for p in self.VALIDATION_PROBLEMS_ENABLED.split(',') if p.strip()]

    # Shared prompts for Gemini validation and polish fallback
    DEFAULT_SYSTEM_PROMPT: str = """You are an expert PDF content extractor. Your task is to extract text content from PDF pages and convert it to clean markdown format.

Key Requirements:
1. Extract ALL text content - do not skip anything
2. Convert to clean, well-structured markdown
3. For tables: Use proper markdown table syntax with | separators
4. Preserve document structure and hierarchy (headings, lists, paragraphs)
5. If you see empty table cells (| | |), look carefully at the image to detect if there's actually content
6. Maintain original formatting and layout as much as possible
7. Do NOT add explanations or metadata - return ONLY the extracted markdown content

Be thorough and accurate."""

    DEFAULT_USER_PROMPT_TEMPLATE: str = """Extract all text content from this PDF page (originally page {page_number}) and convert it to markdown format. Include tables with proper markdown syntax. Do not skip any content. Preserve the original structure and formatting as much as possible.
The content is finance data, so be extra careful with tables and numbers."""

    GEMINI_SYSTEM_PROMPT: Optional[str] = None
    GEMINI_USER_PROMPT_TEMPLATE: Optional[str] = None

    # Image-Specific Validation Prompts
    IMAGE_VALIDATION_SYSTEM_PROMPT: Optional[str] = None
    IMAGE_VALIDATION_USER_PROMPT_TEMPLATE: Optional[str] = None

    # Defaults if not configured in .env
    DEFAULT_IMAGE_VALIDATION_SYSTEM_PROMPT: str = """You are an expert PDF content extractor specializing in documents with charts, diagrams, and images. Your task is to extract ALL content from PDF pages, paying special attention to visual elements.

Key Requirements:
1. Extract ALL text content - do not skip anything
2. For images/charts/diagrams: Describe them thoroughly with data values if visible
3. Convert to clean, well-structured markdown
4. For tables: Use proper markdown table syntax with | separators
5. Preserve document structure and hierarchy (headings, lists, paragraphs)
6. Maintain original formatting and layout as much as possible
7. Do NOT add explanations or metadata - return ONLY the extracted markdown content

Be thorough and accurate, especially with visual data."""

    DEFAULT_IMAGE_VALIDATION_USER_PROMPT_TEMPLATE: str = """Extract all text content from this PDF page (originally page {page_number}) and convert it to markdown format. This page contains images, charts, or diagrams - please describe them thoroughly and extract any visible data values. Include tables with proper markdown syntax. Do not skip any content."""

    def get_system_prompt(self, provider: str) -> str:
        """Get system prompt for a specific provider, with fallback to default."""
        if provider.lower() == "gemini" and self.GEMINI_SYSTEM_PROMPT:
            return self.GEMINI_SYSTEM_PROMPT
        return self.DEFAULT_SYSTEM_PROMPT

    def get_user_prompt_template(self, provider: str) -> str:
        """Get user prompt template for a specific provider, with fallback to default."""
        if provider.lower() == "gemini" and self.GEMINI_USER_PROMPT_TEMPLATE:
            return self.GEMINI_USER_PROMPT_TEMPLATE
        return self.DEFAULT_USER_PROMPT_TEMPLATE

    def get_image_validation_system_prompt(self, provider: str) -> str:
        """Get image-specific system prompt with fallback to defaults."""
        return (
            self.IMAGE_VALIDATION_SYSTEM_PROMPT
            or self.DEFAULT_IMAGE_VALIDATION_SYSTEM_PROMPT
        )

    def get_image_validation_user_prompt_template(self, provider: str) -> str:
        """Get image-specific user prompt template with fallback to defaults."""
        return (
            self.IMAGE_VALIDATION_USER_PROMPT_TEMPLATE
            or self.DEFAULT_IMAGE_VALIDATION_USER_PROMPT_TEMPLATE
        )

    def get_smart_extraction_polish_prompt(self) -> str:
        """Get the Gemini polish prompt with fallback to the default."""
        return (
            self.SMART_EXTRACTION_POLISH_PROMPT
            or self.DEFAULT_SMART_EXTRACTION_POLISH_PROMPT
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
