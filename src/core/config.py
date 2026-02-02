"""
Configuration settings for the application.
"""
from pydantic_settings import BaseSettings
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

    # Azure Document Intelligence Configuration (for table extraction)
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = None
    AZURE_DOCUMENT_INTELLIGENCE_KEY: Optional[str] = None
    AZURE_DOCUMENT_INTELLIGENCE_MODEL: str = "prebuilt-layout"  # Model for layout and table extraction
    AZURE_DI_USE_NUMERICAL_VALIDATION: bool = True  # Enable numerical validation for table merging
    AZURE_DI_BALANCE_TOLERANCE: float = 0.01  # Tolerance for balance comparison (for rounding)

    # Query-Workflow Mapping
    # Maps query patterns to processing workflows
    # Workflow options: "text_extraction", "azure_document_intelligence", "mistral", "openai", "gemini", "gemini-wf", "ocr_with_images"
    QUERY_WORKFLOW_MAPPING: dict = {
        # Bank pages/statements - use Azure Document Intelligence API for smart table extraction
        "01_Fin_Reports": "mistral",
        "02_Trial_Balance": "azure_document_intelligence",
        "03_Balances": "azure_document_intelligence",
        "04_Bank_Statements": "text_extraction",
        # Esna documents - use simple text extraction (pdfplumber)
        #"esna": "text_extraction",
        "05_Esna": "azure_document_intelligence",
        "ocr with images": "ocr_with_images",
        "gemini-wf": "gemini-wf",  # Gemini page-by-page async processing
        # Default fallback
        "default": "mistral"
    }

    # Azure OpenAI Configuration (for cross-validation)
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"

    # Google Gemini Configuration (DEPRECATED - kept for backward compatibility)
    # NOTE: Gemini validation support has been removed. Only OpenAI is supported for cross-validation.
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"  # Gemini Flash Lite

    # Cross-Validation Settings
    ENABLE_CROSS_VALIDATION: bool = True
    VALIDATION_PROVIDER: str = "openai"  # Options: "openai" (gemini validation deprecated)
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

    # Shared Prompts for PDF Extraction (used by OpenAI, Gemini, and Claude)
    # Can be overridden individually via environment variables if needed
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

    # Individual provider prompts (defaults to shared prompts if not set in environment)
    OPENAI_SYSTEM_PROMPT: Optional[str] = None
    OPENAI_USER_PROMPT_TEMPLATE: Optional[str] = None
    GEMINI_SYSTEM_PROMPT: Optional[str] = None
    GEMINI_USER_PROMPT_TEMPLATE: Optional[str] = None

    # OCR with Images Configuration
    OCR_WITH_IMAGES_DEFAULT_PROMPT: str = "Please extract all data from this image in a structured format."

    # Image-Specific Validation Prompts
    IMAGE_VALIDATION_SYSTEM_PROMPT: Optional[str] = None
    IMAGE_VALIDATION_USER_PROMPT_TEMPLATE: Optional[str] = None

    # Finance-Specific Image Validation Prompts (for 01_Fin_Reports workflow)
    GEMINI_FINANCE_IMAGE_SYSTEM_PROMPT: Optional[str] = None
    GEMINI_FINANCE_IMAGE_USER_PROMPT_TEMPLATE: Optional[str] = None

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

    # Finance-Specific Prompts for 01_Fin_Reports with Images
    DEFAULT_GEMINI_FINANCE_IMAGE_SYSTEM_PROMPT: str = """You are an expert at extracting financial and real estate TEXT and TABLES from PDF documents.

CRITICAL RULES:
1. Output ONLY markdown format - NEVER use HTML tags like <table>, <tr>, <th>, <td>
2. IGNORE ALL IMAGES - Do not extract data from images, do not describe images, skip all visual elements
3. Extract ONLY text content and numerical tables

WHAT TO EXTRACT:
✅ Text paragraphs and headings
✅ Numerical tables with ALL columns (especially the leftmost column with row labels)
✅ Monetary values (₪, NIS, USD, EUR, etc.)
✅ Percentages (%)
✅ Dates (all formats: DD/MM/YYYY, בדצמבר 31, etc.)
✅ Property measurements (מ"ר, דונם, sqm, hectares)
✅ Property identifiers (גוש, חלקה, מגרש, plot numbers)
✅ Unit counts (יח"ד, apartments, units)
✅ Company names, party names
✅ License/permit numbers
✅ Building rights (אחוזי בניה, תכסית, קומות)
✅ Transaction details (buyer, seller, price, date, area)
✅ Balance sheet items, P&L items, financial statements
✅ Asset valuations
✅ Interest rates, loan terms
✅ Exchange rates (שער חליפין)
✅ Index values (מדד)

WHAT TO IGNORE:
❌ ALL images, charts, diagrams, graphs - SKIP THEM COMPLETELY
❌ Logos and graphics
❌ Maps and location diagrams
❌ Architectural drawings
❌ Property photos
❌ Any visual elements

TABLE EXTRACTION - CRITICAL:
1. Use ONLY markdown table format with | separators
2. NEVER use HTML tags (<table>, <tr>, <th>, <td>, etc.)
3. For complex headers (rowspan/colspan), flatten to simple markdown tables
4. Extract ALL columns including the leftmost column with row descriptions (do NOT skip the first column)
5. Example markdown table:
   | Row Description | Header 1 | Header 2 | Header 3 |
   |-----------------|----------|----------|----------|
   | Data row 1      | Data 1   | Data 2   | Data 3   |
6. Preserve Hebrew text accurately
7. Do NOT add explanations - return ONLY extracted data in markdown format"""

    DEFAULT_GEMINI_FINANCE_IMAGE_USER_PROMPT_TEMPLATE: str = """Extract all TEXT and TABLES from page {page_number}. IGNORE all images completely.

CRITICAL REQUIREMENTS:
1. SKIP all images, charts, diagrams - do NOT extract data from them
2. Extract ONLY text paragraphs and numerical tables
3. Use ONLY markdown table syntax with | separators (NEVER HTML tags)
4. Extract ALL table columns including row labels (leftmost column - very important!)
5. Flatten complex headers (rowspan/colspan) into simple markdown tables
6. Preserve all monetary values, percentages, dates, and measurements

Return ONLY the extracted text and tables in clean markdown format."""

    def get_system_prompt(self, provider: str) -> str:
        """Get system prompt for a specific provider, with fallback to default."""
        provider_prompts = {
            "openai": self.OPENAI_SYSTEM_PROMPT,
            "gemini": self.GEMINI_SYSTEM_PROMPT
        }
        return provider_prompts.get(provider.lower()) or self.DEFAULT_SYSTEM_PROMPT

    def get_user_prompt_template(self, provider: str) -> str:
        """Get user prompt template for a specific provider, with fallback to default."""
        provider_prompts = {
            "openai": self.OPENAI_USER_PROMPT_TEMPLATE,
            "gemini": self.GEMINI_USER_PROMPT_TEMPLATE
        }
        return provider_prompts.get(provider.lower()) or self.DEFAULT_USER_PROMPT_TEMPLATE

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

    def get_finance_image_system_prompt(self) -> str:
        """Get finance-specific image system prompt for 01_Fin_Reports workflow."""
        return (
            self.GEMINI_FINANCE_IMAGE_SYSTEM_PROMPT
            or self.DEFAULT_GEMINI_FINANCE_IMAGE_SYSTEM_PROMPT
        )

    def get_finance_image_user_prompt_template(self) -> str:
        """Get finance-specific image user prompt template for 01_Fin_Reports workflow."""
        return (
            self.GEMINI_FINANCE_IMAGE_USER_PROMPT_TEMPLATE
            or self.DEFAULT_GEMINI_FINANCE_IMAGE_USER_PROMPT_TEMPLATE
        )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow extra fields in .env for backward compatibility


settings = Settings()
