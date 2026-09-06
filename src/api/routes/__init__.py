"""
API Route Handlers Package
==========================
Exports route submodules for EDA, ML inference, and Talk-to-Data.
"""

from src.api.routes import chat, eda, ml

__all__ = ["eda", "ml", "chat"]
