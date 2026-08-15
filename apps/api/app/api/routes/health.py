from fastapi import APIRouter, Request


router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "mode": request.app.state.mode, "version": "v3.5"}
