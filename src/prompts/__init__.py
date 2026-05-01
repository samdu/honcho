"""Prompt template loading with optional user override directory.

Templates live in ``src/prompts/templates/*.md`` by default. If the env var
``HONCHO_PROMPTS_DIR`` is set, that directory is searched first; missing
templates fall back to the bundled defaults. This lets a deployment
override individual prompts without forking or re-vendoring the entire
prompts module on every upstream sync.
"""

from .loader import load_template

__all__ = ["load_template"]
