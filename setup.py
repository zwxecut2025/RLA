"""C9Agent package setup.

Install:
    pip install -e .                    # core only
    pip install -e ".[full]"            # with API server & document extraction
    pip install -e ".[dev]"             # development dependencies
    pip install -e ".[full,dev]"        # everything
"""

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

readme_path = ROOT / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="c9agent",
    version="0.1.0",
    description="ALS intelligent agent for prognosis prediction and personalized analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.27",
        "pydantic>=2.0",
        "pandas>=2.0",
        "numpy>=1.24",
        "lifelines>=0.28",      # CoxPH survival analysis
        "scipy>=1.10",          # statistical tests (fisher, ttest, chi2)
        "scikit-learn>=1.3",    # model selection / preprocessing utilities
    ],
    extras_require={
        "full": [
            "fastapi>=0.100",       # REST API server (scripts/run_api_server.py)
            "uvicorn>=0.23",        # ASGI server
            "python-docx>=0.8",     # Word document extraction
            "openpyxl>=3.0",        # Excel document extraction
            # "chromadb>=0.4",      # optional vector DB (currently unused)
            # "scikit-survival>=0.22",  # optional random survival forest (future)
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-asyncio>=0.21",
        ],
    },
)
