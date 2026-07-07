# main.py
"""
FastAPI Application Entry Point.
Responsibility: App setup, middleware, lifespan, and API endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.concurrency import run_in_threadpool

import fitz  # PyMuPDF
import instructor
from groq import Groq

from schemas import (
    PatientProfile,
    ProfileAnalysisResponse,
    TrialMatchResult,
    ChatRequest,
    ChatResponse,
    ManualProfileRequest,
)
from agents.oncology_agent import OncologyAgent
from core.entrypoint import run_trial_matching
from core.completeness import (
    get_missing_fields_in_order,
    build_improvement_suggestions,
)
from database import init_db, check_db_connection
from normalizer import normalize_patient_profile
from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    ALLOWED_ORIGINS,
    MAX_FILE_SIZE_BYTES,
    MAX_EXTRACTED_TEXT_CHARS,
    APP_TITLE,
    APP_VERSION,
    APP_DESCRIPTION,
)

logger = logging.getLogger("uvicorn.error")

# -----------------------------------------------------------
# LLM CLIENT — Instructor + Groq for structured extraction
# -----------------------------------------------------------

client = instructor.from_groq(
    Groq(api_key=GROQ_API_KEY),
    mode=instructor.Mode.JSON,
)
_chat_agent = OncologyAgent()

# -----------------------------------------------------------
# LIFESPAN
# -----------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: DB init + health check.
    Shutdown: Clean teardown log.
    """
    logger.info("Starting up %s v%s...", APP_TITLE, APP_VERSION)

    db_healthy = check_db_connection()
    if not db_healthy:
        logger.warning(
            "⚠️ Database unreachable on startup — "
            "DB features will be skipped."
        )
    else:
        logger.info("✅ Database connection verified.")
        init_db()
        logger.info("✅ Database tables ready.")

    logger.info("✅ Application startup complete.")
    yield
    logger.info("Application shutting down. Goodbye!")


# -----------------------------------------------------------
# APP
# -----------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -----------------------------------------------------------
# CORS MIDDLEWARE
# -----------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# VALIDATION ERROR HANDLER
# Surfaces exact 422 field errors in logs + response
# -----------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Catches Pydantic 422 validation errors and logs
    the exact failing fields for easy debugging.
    """
    logger.error(
        "422 Validation Error | url=%s | errors=%s",
        request.url,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "hint": (
                "Check that all required fields are correct types. "
                "age must be integer 1-120, "
                "empty optional fields must be null not empty string."
            ),
        },
    )


# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------

def _build_profile_analysis_response(
    profile: PatientProfile,
) -> ProfileAnalysisResponse:
    """
    Builds a consistent analysis response with ordered missing
    fields and user-facing improvement suggestions.
    """
    missing_fields = get_missing_fields_in_order(profile)
    improvement_suggestions = build_improvement_suggestions(missing_fields)
    is_complete = len(missing_fields) == 0

    if not profile.cancer_type:
        status_value = "NEEDS_CLARIFICATION"
    else:
        status_value = "PROFILE_READY"

    return ProfileAnalysisResponse(
        profile=profile,
        status=status_value,
        is_complete=is_complete,
        missing_fields=missing_fields,
        improvement_suggestions=improvement_suggestions,
        agent_suggestions=improvement_suggestions,
        trial_matches=None,
    )


def _coerce_trial_match_result(
    match_result,
    cancer_type: str = "",
) -> TrialMatchResult:
    """Normalizes the matcher output into TrialMatchResult."""
    if isinstance(match_result, TrialMatchResult):
        return match_result

    if isinstance(match_result, dict):
        return TrialMatchResult(
            final_recommendations=match_result.get(
                "final_recommendations", ""
            ),
            eligibility_results=match_result.get(
                "eligibility_results", []
            ),
            trials=match_result.get("trials", []),
            trial_count=match_result.get("trial_count", 0),
            cancer_type=match_result.get(
                "cancer_type", cancer_type or ""
            ),
            success=match_result.get("success", True),
            error=match_result.get("error"),
        )

    return TrialMatchResult(
        final_recommendations="",
        eligibility_results=[],
        trials=[],
        trial_count=0,
        cancer_type=cancer_type or "",
        success=False,
        error="Unexpected trial matching result format.",
    )


def _extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extracts raw text from PDF bytes using PyMuPDF.
    Always closes the document — even on error.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        return "\n".join(
            page.get_text() for page in doc
        ).strip()
    finally:
        doc.close()


def _build_extraction_messages(extracted_text: str) -> list:
    """Builds the LLM message payload for patient profile extraction."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert clinical data extraction AI. "
                "Extract the patient profile strictly into the "
                "provided JSON schema. "
                "If a field is missing, return null or an empty array. "
                "Do not guess. Do not hallucinate. "
                "Only use information explicitly present in the text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extract the medical data from this text:"
                f"\n\n{extracted_text}"
            ),
        },
    ]


async def _run_matching_if_possible(
    analysis_result: ProfileAnalysisResponse,
) -> ProfileAnalysisResponse:
    """
    Runs the trial-matching pipeline only when
    cancer_type is present. If matching fails,
    preserves the profile and suggestions.
    """
    if analysis_result.status != "PROFILE_READY":
        logger.warning(
            "Profile not ready for matching. "
            "Missing required fields: %s",
            analysis_result.missing_fields,
        )
        return analysis_result

    try:
        logger.info(
            "Profile ready. Routing to Clinical Matcher Agent..."
        )
        # Always pass dict to run_trial_matching
        patient_dict = analysis_result.profile.model_dump()
        match_result = await run_trial_matching(patient_dict)

        analysis_result.trial_matches = _coerce_trial_match_result(
            match_result,
            cancer_type=analysis_result.profile.cancer_type or "",
        )

        logger.info(
            "Trial matching complete. Found %d trials.",
            analysis_result.trial_matches.trial_count
            if analysis_result.trial_matches
            else 0,
        )
        return analysis_result

    except Exception:
        logger.exception(
            "Trial matching failed — "
            "returning profile without matches."
        )
        analysis_result.status = "MATCHING_FAILED"
        analysis_result.trial_matches = TrialMatchResult(
            final_recommendations="",
            eligibility_results=[],
            trials=[],
            trial_count=0,
            cancer_type=analysis_result.profile.cancer_type or "",
            success=False,
            error="Trial matching pipeline failed.",
        )
        return analysis_result


# -----------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------

@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="API Health Check",
)
async def health_check():
    """Returns API and database health status."""
    db_status = check_db_connection()
    return {
        "api": "healthy",
        "database": "healthy" if db_status else "unreachable",
        "version": APP_VERSION,
    }


# -----------------------------------------------------------
# ANALYZE PDF ENDPOINT
# -----------------------------------------------------------

@app.post(
    "/api/v1/profile/analyze",
    response_model=ProfileAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Profile"],
    summary="Analyze patient PDF and match clinical trials",
)
async def analyze_patient_document(
    file: UploadFile = File(...)
):
    """
    Full pipeline:
        1. Validate PDF upload
        2. Extract text via PyMuPDF
        3. Extract PatientProfile via LLM (Groq + Instructor)
        4. Normalize profile fields
        5. Evaluate completeness + suggest missing details
        6. Run trial matching if cancer_type is available
        7. Return ProfileAnalysisResponse
    """
    if (
        not file.filename
        or not file.filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Please upload a PDF.",
        )

    try:
        file_bytes = await file.read()

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File too large. Maximum allowed size is "
                    f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
                ),
            )

        if not file_bytes.startswith(b"%PDF"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file format.",
            )

        extracted_text = _extract_text_from_pdf_bytes(file_bytes)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "No readable text found. "
                    "Ensure the PDF is not a scanned image."
                ),
            )

        if len(extracted_text) > MAX_EXTRACTED_TEXT_CHARS:
            logger.warning(
                "Extracted text too long (%d chars). "
                "Truncating to %d chars.",
                len(extracted_text),
                MAX_EXTRACTED_TEXT_CHARS,
            )
            extracted_text = extracted_text[:MAX_EXTRACTED_TEXT_CHARS]

        logger.info("Extracting patient profile from PDF text...")
        extracted_profile: PatientProfile = await run_in_threadpool(
            lambda: client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                response_model=PatientProfile,
                messages=_build_extraction_messages(extracted_text),
            )
        )
        logger.info("LLM extraction complete.")

        logger.info("Normalizing patient profile...")
        try:
            normalized_profile = normalize_patient_profile(
                extracted_profile
            )
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve),
            )
        logger.info("Normalization complete.")

        logger.info("Auditing profile completeness...")
        analysis_result = _build_profile_analysis_response(
            normalized_profile
        )
        analysis_result = await _run_matching_if_possible(
            analysis_result
        )
        return analysis_result

    except HTTPException:
        raise
    except Exception:
        logger.exception("Pipeline execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )
    finally:
        await file.close()


# -----------------------------------------------------------
# MANUAL PROFILE ENDPOINT
# Single clean endpoint — no duplicate
# -----------------------------------------------------------

@app.post(
    "/api/v1/profile/manual",
    response_model=ProfileAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Profile"],
    summary="Submit patient details manually and match trials",
)
async def analyze_manual_profile(
    request: ManualProfileRequest,
) -> ProfileAnalysisResponse:
    """
    Manual profile pipeline:
        1. Accept ManualProfileRequest JSON body
        2. Normalize profile fields
        3. Evaluate completeness + suggest missing details
        4. Run trial matching if cancer_type is available
        5. Return ProfileAnalysisResponse

    ManualProfileRequest wraps PatientProfile so the
    JSON body is:
        { "profile": { "age": 62, "cancer_type": "..." } }
    """
    try:
        logger.info(
            "Manual profile received | cancer_type=%s | age=%s",
            request.profile.cancer_type,
            request.profile.age,
        )

        # Normalize
        try:
            normalized_profile = normalize_patient_profile(
                request.profile
            )
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(ve),
            )

        # Build response + run matching
        analysis_result = _build_profile_analysis_response(
            normalized_profile
        )
        analysis_result = await _run_matching_if_possible(
            analysis_result
        )
        return analysis_result

    except HTTPException:
        raise
    except Exception:
        logger.exception("Manual profile analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Manual profile analysis failed.",
        )


# -----------------------------------------------------------
# CHAT ENDPOINTS
# -----------------------------------------------------------

@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Chat with the oncology assistant",
)
async def chat_with_agent(request: ChatRequest):
    """
    Multi-turn conversational endpoint.
    Accepts a message + optional patient profile and trial context.
    Returns AI response with session tracking.
    """
    try:
        result = await _chat_agent.run({
            "session_id": request.session_id,
            "message": request.message,
            "patient_profile": request.patient_profile,
            "trial_matches": request.trial_matches,
        })
        return ChatResponse(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception:
        logger.exception("Chat endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat service temporarily unavailable.",
        )


@app.delete(
    "/api/v1/chat/{session_id}",
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Clear chat session history",
)
async def clear_chat_session(session_id: str):
    """Clears conversation history for a given session."""
    await _chat_agent.clear_session(session_id)
    return {
        "message": f"Session {session_id} cleared successfully."
    }


@app.get(
    "/api/v1/chat/{session_id}/history",
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Get chat history for a session",
)
async def get_chat_history(session_id: str):
    """Returns full conversation history for a session."""
    history = await _chat_agent.get_history(session_id)
    return {
        "session_id": session_id,
        "history": history,
    }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Catches Pydantic 422 validation errors and returns
    clean, JSON-serializable field-level error messages.

    Strips non-serializable objects (like ValueError)
    from the ctx dict before returning.
    """
    # ── Sanitize errors — strip non-serializable ctx values ──
    clean_errors = []
    for error in exc.errors():
        clean_error = {
            "type":  error.get("type", ""),
            "loc":   list(error.get("loc", [])),
            "msg":   error.get("msg", ""),
            "input": str(error.get("input", "")),
            # ✅ Convert ctx.error to string — NOT the object
            "ctx": {
                k: str(v)
                for k, v in (error.get("ctx") or {}).items()
            } if error.get("ctx") else {},
        }
        clean_errors.append(clean_error)

    logger.error(
        "422 Validation Error | url=%s | errors=%s",
        request.url,
        clean_errors,
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": clean_errors,
            "hint": (
                "Check that all required fields are correct types. "
                "Age must be an integer between 1 and 120. "
                "Empty optional fields must be null not empty string."
            ),
        },
    )
