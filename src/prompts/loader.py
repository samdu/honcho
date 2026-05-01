import os
from functools import cache
from pathlib import Path

_BUNDLED_TEMPLATES_DIR = Path(__file__).parent / "templates"
_OVERRIDE_ENV_VAR = "HONCHO_PROMPTS_DIR"


def _resolve_template_path(name: str) -> Path:
    """Resolve a template name to a filesystem path.

    Looks in ``$HONCHO_PROMPTS_DIR`` first if set, then falls back to the
    bundled templates directory. Raises FileNotFoundError if neither has it.
    """
    override_dir = os.environ.get(_OVERRIDE_ENV_VAR)
    if override_dir:
        override_path = Path(override_dir).expanduser() / name
        if override_path.is_file():
            return override_path
    bundled_path = _BUNDLED_TEMPLATES_DIR / name
    if bundled_path.is_file():
        return bundled_path
    raise FileNotFoundError(
        f"Prompt template '{name}' not found in ${_OVERRIDE_ENV_VAR}={override_dir!r} or {_BUNDLED_TEMPLATES_DIR}"
    )


@cache
def load_template(name: str) -> str:
    """Load a prompt template by filename, e.g. ``load_template('deriver.md')``.

    Cached after first read. Use ``load_template.cache_clear()`` to force a
    reload (e.g. in tests after editing a template).
    """
    return _resolve_template_path(name).read_text(encoding="utf-8")
