"""
System Router

GET  /api/v1/system/memory
POST /api/v1/system/memory/cleanup
POST /api/v1/system/memory/emergency
GET  /api/v1/system/metrics
GET  /api/v1/system/models
"""

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_current_user
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/system/memory")
async def get_memory_usage(user_id: str = Depends(get_current_user)):
    from src.models.memory.memory_manager import memory_manager
    usage = memory_manager.get_memory_usage()
    return {"success": True, "data": usage}


@router.post("/system/memory/cleanup")
async def memory_cleanup(user_id: str = Depends(get_current_user)):
    if user_id != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from src.models.memory.memory_manager import memory_manager
    usage = memory_manager.get_memory_usage()
    return {"success": True, "data": {"message": "Cleanup completed", "memory_after": usage}}


@router.post("/system/memory/emergency")
async def emergency_cleanup(user_id: str = Depends(get_current_user)):
    if user_id != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    from src.models.memory.memory_manager import memory_manager
    memory_manager.aggressive_cleanup()
    usage = memory_manager.get_memory_usage()
    return {"success": True, "data": {"message": "Emergency cleanup completed", "memory_after": usage}}


@router.get("/system/metrics")
async def get_metrics(user_id: str = Depends(get_current_user)):
    import psutil
    import torch

    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    metrics = {
        "cpu_percent": cpu_percent,
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_percent": memory.percent,
    }

    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        gpu_used = torch.cuda.memory_allocated(0)
        metrics["gpu_total_gb"] = round(gpu_total / (1024**3), 2)
        metrics["gpu_used_gb"] = round(gpu_used / (1024**3), 2)
        metrics["gpu_percent"] = round(gpu_used / gpu_total * 100, 1) if gpu_total > 0 else 0
        metrics["gpu_name"] = torch.cuda.get_device_name(0)

    return {"success": True, "data": metrics}


@router.get("/system/models")
async def get_available_models(user_id: str = Depends(get_current_user)):
    """Return available generation models grouped by provider."""
    models: dict = {"cloud": [], "huggingface": [], "ollama": []}

    # Detect local Ollama models
    try:
        import httpx
        from src.api import config as app_config
        resp = httpx.get(f"{app_config.OLLAMA_LOCAL_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            for model in data.get("models", []):
                name = model.get("name", "")
                models["ollama"].append({"value": f"ollama-{name}", "label": name})
    except Exception:
        pass

    # Detect Ollama Cloud models (when API key is configured)
    try:
        from src.utils.ollama_client import get_ollama_api_key
        api_key = get_ollama_api_key()
        if api_key:
            resp = httpx.get(
                f"{app_config.OLLAMA_CLOUD_URL}/api/tags",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("models", []):
                    name = model.get("name", "")
                    # Avoid duplicates (cloud model already pulled locally)
                    cloud_value = f"ollama-{name}"
                    if not any(m["value"] == cloud_value for m in models["ollama"]):
                        models["cloud"].append({"value": cloud_value, "label": f"{name} (Cloud)"})
    except Exception:
        pass

    # Detect HuggingFace models from known locations
    try:
        import torch
        if torch.cuda.is_available():
            models["huggingface"] = [
                {"value": "Qwen/Qwen2.5-VL-7B-Instruct", "label": "Qwen2.5-VL 7B"},
            ]
    except Exception:
        pass

    return {"success": True, "data": models}
