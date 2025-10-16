"""
C1: API Gateway - FastAPI application for LLM inference.

Responsibilities:
- Expose /generate and /embed APIs (FR 001)
- Handle authentication and rate limiting
- Stream tokens via SSE/gRPC/WebSocket
- Integrate with Ray cluster
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncIterator
import logging
import json
import asyncio

from faasinfer.system import FaaSInfer
from faasinfer.config import FaaSInferConfig
from faasinfer.core.types import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


# Request/Response models
class GenerateRequest(BaseModel):
    """Generate request model."""
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stream: bool = True


class GenerateResponse(BaseModel):
    """Generate response model."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[dict]


class EmbedRequest(BaseModel):
    """Embed request model."""
    input: str
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    """Embed response model."""
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


# Create FastAPI app
app = FastAPI(
    title="FaaSInfer API",
    description="Low-Latency Serverless Inference for LLMs",
    version="1.0.0",
)


# Global FaaSInfer instance
faasinfer_system: Optional[FaaSInfer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize FaaSInfer system on startup."""
    global faasinfer_system
    
    logger.info("Starting FaaSInfer API Gateway...")
    
    # Load configuration
    config = FaaSInferConfig()
    
    # Initialize system
    faasinfer_system = FaaSInfer(config)
    await faasinfer_system.initialize()
    
    logger.info("FaaSInfer API Gateway ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown FaaSInfer system."""
    global faasinfer_system
    
    logger.info("Shutting down FaaSInfer API Gateway...")
    
    if faasinfer_system:
        await faasinfer_system.shutdown()
    
    logger.info("FaaSInfer API Gateway stopped")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "faasinfer"}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    if faasinfer_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    models = [
        {
            "id": model.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "faasinfer",
        }
        for model in faasinfer_system.config.models
    ]
    
    return {
        "object": "list",
        "data": models,
    }


@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    """
    Generate text completion.
    
    Implements FR 001: Stateless /generate API with streaming.
    """
    if faasinfer_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        if request.stream:
            # Stream response via SSE
            async def event_generator():
                async for response in faasinfer_system.generate(
                    prompt=request.prompt,
                    model_id=request.model,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                ):
                    # Format as SSE
                    data = {
                        "id": response.request_id,
                        "object": "text_completion.chunk",
                        "created": 0,
                        "model": request.model or "default",
                        "choices": [{
                            "index": 0,
                            "text": response.text,
                            "finish_reason": response.finish_reason,
                        }],
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    if response.finish_reason:
                        yield "data: [DONE]\n\n"
                        break
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )
        
        else:
            # Non-streaming response
            full_text = ""
            final_response = None
            
            async for response in faasinfer_system.generate(
                prompt=request.prompt,
                model_id=request.model,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                full_text = response.text
                final_response = response
                
                if response.finish_reason:
                    break
            
            return {
                "id": final_response.request_id if final_response else "unknown",
                "object": "text_completion",
                "created": 0,
                "model": request.model or "default",
                "choices": [{
                    "index": 0,
                    "text": full_text,
                    "finish_reason": final_response.finish_reason if final_response else None,
                }],
            }
    
    except Exception as e:
        logger.error(f"Generate request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbedRequest):
    """
    Create embeddings.
    
    Implements FR 001: Stateless /embed API.
    """
    if faasinfer_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Mock implementation
    # In real system, would use embedding model
    return {
        "object": "list",
        "data": [{
            "object": "embedding",
            "embedding": [0.0] * 768,  # Mock embedding
            "index": 0,
        }],
        "model": request.model or "default",
        "usage": {
            "prompt_tokens": len(request.input.split()),
            "total_tokens": len(request.input.split()),
        },
    }


@app.get("/v1/status")
async def get_status():
    """Get system status."""
    if faasinfer_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = await faasinfer_system.get_status()
        return status
    except Exception as e:
        logger.error(f"Status request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )