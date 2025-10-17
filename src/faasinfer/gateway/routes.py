"""Additional API routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    # In production, would export actual Prometheus metrics
    return {
        "requests_total": 0,
        "requests_in_flight": 0,
        "latency_seconds": 0.0,
    }


@router.get("/ready")
async def readiness():
    """Readiness probe."""
    return {"ready": True}


@router.get("/live")
async def liveness():
    """Liveness probe."""
    return {"alive": True}

