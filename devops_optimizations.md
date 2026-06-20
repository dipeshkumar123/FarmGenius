# DevOps Optimizations Report

## 1. Dockerfile Enhancements

The Dockerfile has been refactored to use a multi-stage build, significantly reducing the final image size and improving security.
- **Multi-stage Builds**: Separated the build environment (which installs wheels and dependencies) from the runtime environment. The final image only copies the compiled virtual environment (`/opt/venv`).
- **Non-root User**: Added a dedicated `appuser` user to run the container. This follows the principle of least privilege, preventing the application from running as `root`.
- **Environment Variables**: Set `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1` to optimize Python execution and log outputs inside containers.

## 2. Dependency Tree Optimization

The Python dependency tree has been heavily pruned to prevent bloating the API container:
- **Separated Development/ML Tools**: Moved `pytest`, `requests` (unused), `scikit-learn`, and `Pillow` out of the core API requirements (`requirements-api.txt`).
- **Dynamic Imports**: Refactored `app/api/routes/disease.py` to move heavy ML libraries (`numpy`, `PIL`) inside the inference block. Since the backend relies on the Groq Vision API fallback for disease detection in production (as local detection happens on-device in Flutter), installing heavy packages like `tensorflow`, `numpy`, and `Pillow` is now completely bypassed in the Docker image.
- **Removed Redundant Libraries**: Removed unused libraries (like `requests`, since `httpx` is used throughout the API).
- **Added Missing Libraries**: Added `googlesearch-python` and `beautifulsoup4` to `requirements-api.txt` since they are dynamically used by `schemes_service.py` to fetch real-time government scheme data.

## Summary

The resulting backend API container will build faster, feature a massively reduced attack surface, and scale more effectively on Render's free tier by avoiding massive ML dependencies natively. 
