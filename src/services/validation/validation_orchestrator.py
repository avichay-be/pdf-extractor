"""
Validation orchestration for PDF content quality assurance.

Coordinates problem detection, similarity calculation, and cross-validation.
"""
import asyncio
import logging
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from src.core.config import settings
from src.services.openai_client import OpenAIDocumentClient
from src.models.mistral_models import MistralOCRResponse

from .problem_detector import ProblemDetector
from .similarity_calculator import SimilarityCalculator
from .content_normalizer import ContentNormalizer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single page."""
    page_number: int
    similarity_score: float
    passed: bool
    has_problem_pattern: bool
    alternative_content: Optional[str]
    processing_time: float
    error: Optional[str] = None


@dataclass
class CrossValidationReport:
    """Comprehensive report of cross-validation process."""
    total_pages: int
    validated_pages: int
    problem_pages: List[int] = field(default_factory=list)
    failed_validations: List[int] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    total_time: float = 0.0
    total_cost: float = 0.0  # Estimated cost in USD


class ValidationService:
    """Service for cross-validating PDF extraction results."""

    # Cost estimation (approximate)
    VALIDATOR_COST_PER_1K_TOKENS = 0.01  # Approximate cost for validator (OpenAI/Gemini)
    AVG_TOKENS_PER_PAGE = 500  # Rough estimate

    def __init__(
        self,
        openai_client: Optional[OpenAIDocumentClient] = None,
        gemini_client: Optional['GeminiDocumentClient'] = None
    ):
        """
        Initialize validation service with configurable validator.

        Args:
            openai_client: Pre-initialized OpenAI client (optional, will be created if not provided)
            gemini_client: Pre-initialized Gemini client (optional, will be created if not provided)
        """
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.validator_client = None

        # Initialize specialized components
        self.normalizer = ContentNormalizer()
        self.problem_detector = ProblemDetector(number_extractor=self.normalizer.extract_numbers)
        self.similarity_calculator = SimilarityCalculator(normalizer=self.normalizer)

        if not settings.ENABLE_CROSS_VALIDATION:
            logger.info("Cross-validation is disabled")
            return

        # Select validator based on settings.VALIDATION_PROVIDER
        if settings.VALIDATION_PROVIDER == "gemini":
            # Initialize Gemini client if not provided
            if not self.gemini_client:
                try:
                    from src.services.gemini_client import GeminiDocumentClient
                    self.gemini_client = GeminiDocumentClient()
                    logger.info("Validation service initialized with Gemini validator")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini client: {e}")
                    logger.warning("Cross-validation will be disabled")
                    return
            self.validator_client = self.gemini_client
        else:  # Default to OpenAI
            # Initialize OpenAI client if not provided
            if not self.openai_client:
                try:
                    self.openai_client = OpenAIDocumentClient()
                    logger.info("Validation service initialized with OpenAI validator")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
                    logger.warning("Cross-validation will be disabled")
                    return
            self.validator_client = self.openai_client

        # Log validation configuration at INFO level for observability
        logger.info("=" * 60)
        logger.info("Validation Configuration:")
        logger.info(f"  Provider: {settings.VALIDATION_PROVIDER}")
        logger.info(f"  Similarity Method: {settings.VALIDATION_SIMILARITY_METHOD}")
        logger.info(f"  Similarity Threshold: {settings.VALIDATION_SIMILARITY_THRESHOLD:.2%}")
        logger.info(f"  Sample Rate: 1/{settings.VALIDATION_SAMPLE_RATE} pages")
        logger.info(f"  Skip Sample If Clean: {settings.VALIDATION_SKIP_SAMPLE_IF_CLEAN}")
        logger.info(f"  Enabled Problems ({len(settings.validation_problems_list)}): {', '.join(settings.validation_problems_list)}")
        logger.info("=" * 60)

    # ============================================================================
    # DELEGATED METHODS (delegate to specialized components)
    # ============================================================================

    def detect_problem_pattern(self, markdown_content: str) -> bool:
        """Delegate to problem detector."""
        return self.problem_detector.detect_problem_pattern(markdown_content)

    def detect_all_problems(self, markdown_content: str) -> Dict[str, bool]:
        """Delegate to problem detector."""
        return self.problem_detector.detect_all_problems(markdown_content)

    def has_any_problem(self, markdown_content: str, enabled_problems: Optional[List[str]] = None) -> tuple[bool, List[str]]:
        """Delegate to problem detector."""
        return self.problem_detector.has_any_problem(markdown_content, enabled_problems)

    def detect_problems_batch(
        self,
        pages_content: List[tuple[int, str]],
        enabled_problems: Optional[List[str]] = None
    ) -> dict[int, tuple[bool, List[str]]]:
        """Delegate to problem detector."""
        return self.problem_detector.detect_problems_batch(pages_content, enabled_problems)

    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Delegate to similarity calculator."""
        return self.similarity_calculator.calculate_similarity(content1, content2)

    def calculate_similarity_number_frequency(self, content1: str, content2: str) -> float:
        """Delegate to similarity calculator."""
        return self.similarity_calculator.calculate_similarity_number_frequency(content1, content2)

    def calculate_similarity_levenshtein(self, content1: str, content2: str) -> float:
        """Delegate to similarity calculator."""
        return self.similarity_calculator.calculate_similarity_levenshtein(content1, content2)

    # ============================================================================
    # VALIDATION ORCHESTRATION
    # ============================================================================

    def should_validate_page(
        self,
        page_index: int,
        total_pages: int,
        has_query: bool,
        random_offset: int
    ) -> bool:
        """
        Determine if a page should be sample-validated.

        Args:
            page_index: Zero-based page index
            total_pages: Total number of pages
            has_query: Whether query filtering is active
            random_offset: Random offset for sampling (0-9)

        Returns:
            True if page should be validated, False otherwise
        """
        if not has_query:
            return False  # No sampling if no query filtering

        # Check if this page falls on the sample interval
        sample_rate = settings.VALIDATION_SAMPLE_RATE
        return (page_index - random_offset) % sample_rate == 0

    async def validate_page(
        self,
        original_content: str,
        page_pdf_bytes: bytes,
        page_number: int,
        detected_problems: List[str] = None,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt_template: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a single page by comparing with validator extraction (async).

        Performance optimized: Uses asyncio.to_thread() for blocking I/O operations.

        Args:
            original_content: Content from Mistral extraction
            page_pdf_bytes: PDF bytes for this specific page
            page_number: Page number (0-based)
            detected_problems: Pre-detected problems (if None, will be detected)
            custom_system_prompt: Optional custom system prompt (for image-specific validation)
            custom_user_prompt_template: Optional custom user prompt template

        Returns:
            ValidationResult with comparison details
        """
        start_time = time.time()

        try:
            # Use pre-detected problems or detect if not provided (avoid duplicate detection)
            if detected_problems is None:
                has_problem, detected_problems = self.has_any_problem(original_content)
            else:
                has_problem = bool(detected_problems)

            if has_problem:
                logger.info(f"[Page {page_number}] Problems detected ({', '.join(detected_problems)}) - replacing with Gemini")

            # Extract with validator (Gemini) - run in thread pool to avoid blocking
            alternative_content = await asyncio.to_thread(
                self.validator_client.extract_page_content,
                page_pdf_bytes,
                page_number,
                custom_system_prompt,
                custom_user_prompt_template
            )

            # If problems exist, just use Gemini directly (no comparison needed)
            if has_problem:
                similarity_score = 0.0  # Indicate replacement
                passed = False  # Use alternative content
                logger.info(f"[Page {page_number}] Using Gemini extraction directly (problems detected)")
            else:
                # Only calculate similarity if no problems (clean page sample validation)
                # OPTIMIZATION: Run similarity calculation async to avoid blocking (can be CPU-intensive)
                similarity_score = await asyncio.to_thread(
                    self.calculate_similarity,
                    original_content,
                    alternative_content
                )
                passed = similarity_score >= settings.VALIDATION_SIMILARITY_THRESHOLD

            processing_time = time.time() - start_time

            # Get normalized versions for logging
            norm1 = self.normalizer.normalize_for_comparison(original_content)
            norm2 = self.normalizer.normalize_for_comparison(alternative_content)

            # Log content comparison
            # Determine validator name dynamically based on settings
            validator_name = settings.VALIDATION_PROVIDER.upper()
            logger.info(f"\n{'='*80}")
            logger.info(f"[Page {page_number}] VALIDATION CONTENT COMPARISON ({validator_name})")
            logger.info(f"{'='*80}")
            logger.info(f"Mistral Content ({len(original_content)} chars, {len(norm1)} alphanumeric):")
            logger.info(f"{'-'*80}")
            logger.info(f"{original_content[:500]}{'...' if len(original_content) > 500 else ''}")
            logger.info(f"{'-'*80}")
            logger.info(f"Validator ({validator_name}) Content ({len(alternative_content)} chars, {len(norm2)} alphanumeric):")
            logger.info(f"{'-'*80}")
            logger.info(f"{alternative_content[:500]}{'...' if len(alternative_content) > 500 else ''}")
            logger.info(f"{'-'*80}")
            logger.info(f"Normalized Mistral (first 200 chars): {norm1[:200]}{'...' if len(norm1) > 200 else ''}")
            logger.info(f"Normalized {validator_name} (first 200 chars): {norm2[:200]}{'...' if len(norm2) > 200 else ''}")
            logger.info(f"{'-'*80}")

            # Log result
            status = "PASSED" if passed else "FAILED"
            logger.info(
                f"[Page {page_number}] Similarity: {similarity_score:.2%} - {status} "
                f"(threshold: {settings.VALIDATION_SIMILARITY_THRESHOLD:.2%})"
            )
            logger.info(f"{'='*80}\n")

            return ValidationResult(
                page_number=page_number,
                similarity_score=similarity_score,
                passed=passed,
                has_problem_pattern=has_problem,
                alternative_content=alternative_content if (has_problem or not passed) else None,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[Page {page_number}] Validation failed with error: {e}")

            has_problem, _ = self.has_any_problem(original_content)

            return ValidationResult(
                page_number=page_number,
                similarity_score=0.0,
                passed=False,
                has_problem_pattern=has_problem,
                alternative_content=None,
                processing_time=processing_time,
                error=str(e)
            )

    async def cross_validate_pages(
        self,
        mistral_response: MistralOCRResponse,
        pdf_bytes: bytes,
        has_query: bool = False,
        workflow_name: Optional[str] = None
    ) -> CrossValidationReport:
        """
        Cross-validate pages from Mistral extraction with parallel processing.

        Performance optimized: Validates multiple pages concurrently using asyncio.gather().

        Args:
            mistral_response: Response from Mistral API
            pdf_bytes: PDF file bytes (in-memory to prevent file system race conditions)
            has_query: Whether query filtering is active
            workflow_name: Name of the workflow (e.g., "01_Fin_Reports") for workflow-specific prompts

        Returns:
            Comprehensive validation report
        """
        if not self.validator_client:
            logger.warning("Validator client not available - skipping validation")
            return CrossValidationReport(
                total_pages=len(mistral_response.pages),
                validated_pages=0
            )

        logger.info(f"Starting cross-validation for {len(mistral_response.pages)} pages")
        validator_name = settings.VALIDATION_PROVIDER.upper()
        logger.info(f"Using validator: {validator_name}")
        if has_query:
            logger.info(f"Query filtering active - sample validation every {settings.VALIDATION_SAMPLE_RATE}th page")

        start_time = time.time()

        # Random offset for sampling (0-9)
        random_offset = random.randint(0, settings.VALIDATION_SAMPLE_RATE - 1)
        logger.debug(f"Sample validation offset: {random_offset}")

        problem_pages = []
        pages_to_validate = []  # List of (page_index, page_content, reason, detected_problems, custom_system, custom_user) tuples

        # First pass: Detect problems for ALL pages in parallel (OPTIMIZATION #5)
        # This eliminates the 7-second sequential bottleneck by running problem detection concurrently
        logger.info(f"Running problem detection in parallel for {len(mistral_response.pages)} pages...")

        # CRITICAL: Capture enabled_problems in main thread before parallel processing
        # settings.validation_problems_list is a @property - accessing it in thread pool can be inconsistent
        enabled_problems = settings.validation_problems_list
        logger.debug(f"Enabled problem patterns for parallel detection: {enabled_problems}")

        # Create async tasks for problem detection (one per page)
        # Using asyncio.to_thread() to run CPU-bound detection in thread pool
        # Pass enabled_problems explicitly to ensure consistent detection across all threads
        detection_tasks = [
            asyncio.to_thread(
                self.has_any_problem,
                page.markdown,
                enabled_problems  # ← Explicit parameter for thread-safe access
            )
            for page in mistral_response.pages
        ]

        # Wait for all detections to complete in parallel
        # Expected: 100 pages in ~350ms (vs 7s sequential)
        detection_results = await asyncio.gather(*detection_tasks)

        # Process results to determine which pages need validation
        for page, (has_problem, detected_problems) in zip(mistral_response.pages, detection_results):
            page_index = page.index
            page_content = page.markdown

            should_validate = False
            reason = ""
            custom_system = None
            custom_user = None

            # Check if page has markdown images - use custom prompts
            if 'markdown_images' in detected_problems:
                # Use finance-specific prompts for 01_Fin_Reports workflow
                if workflow_name == "01_Fin_Reports":
                    custom_system = settings.get_finance_image_system_prompt()
                    custom_user = settings.get_finance_image_user_prompt_template()
                    logger.info(f"[Page {page_index}] Image detected in 01_Fin_Reports - using finance-specific validation prompts")
                else:
                    custom_system = settings.get_image_validation_system_prompt(validator_name)
                    custom_user = settings.get_image_validation_user_prompt_template(validator_name)
                    logger.info(f"[Page {page_index}] Image detected - using custom validation prompts")

            if has_problem:
                should_validate = True
                reason = f"problems detected: {', '.join(detected_problems)}"
                problem_pages.append(page_index)
                logger.info(f"[Page {page_index}] Problems found: {', '.join(detected_problems)}")

            # Check if should sample-validate (optimization: skip sampling if no problems and skip_sample_if_clean is enabled)
            elif not settings.VALIDATION_SKIP_SAMPLE_IF_CLEAN and self.should_validate_page(page_index, len(mistral_response.pages), has_query, random_offset):
                should_validate = True
                reason = "sample validation"

            # Add to validation queue if needed
            if should_validate:
                logger.info(f"[Page {page_index}] Queued for validation ({reason})")
                pages_to_validate.append((page_index, page_content, reason, detected_problems, custom_system, custom_user))

        # Second pass: Validate all pages in parallel
        logger.info(f"Validating {len(pages_to_validate)} pages in parallel...")

        validation_tasks = [
            self.validate_page(
                original_content=page_content,
                page_pdf_bytes=pdf_bytes,
                page_number=page_index,
                detected_problems=problems,
                custom_system_prompt=custom_sys,
                custom_user_prompt_template=custom_usr
            )
            for page_index, page_content, _, problems, custom_sys, custom_usr in pages_to_validate
        ]

        validation_results = await asyncio.gather(*validation_tasks) if validation_tasks else []

        # Collect failed validations
        failed_validations = [
            result.page_number
            for result in validation_results
            if not result.passed
        ]

        # Calculate total time and cost
        total_time = time.time() - start_time
        total_cost = len(validation_results) * self.AVG_TOKENS_PER_PAGE * self.VALIDATOR_COST_PER_1K_TOKENS / 1000

        logger.info(f"Cross-validation complete:")
        logger.info(f"  - Total pages: {len(mistral_response.pages)}")
        logger.info(f"  - Validated: {len(validation_results)} pages ({len(validation_results) / len(mistral_response.pages) * 100:.1f}%)")
        logger.info(f"  - Problem pages fixed: {len(problem_pages)}")
        logger.info(f"  - Failed validations: {len(failed_validations)}")
        logger.info(f"  - Total time: {total_time:.2f}s")
        logger.info(f"  - Estimated cost: ${total_cost:.4f}")

        return CrossValidationReport(
            total_pages=len(mistral_response.pages),
            validated_pages=len(validation_results),
            problem_pages=problem_pages,
            failed_validations=failed_validations,
            validation_results=validation_results,
            total_time=total_time,
            total_cost=total_cost
        )
