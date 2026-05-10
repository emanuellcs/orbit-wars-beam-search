"""Local alias for the Kaggle ``main.py`` entrypoint.

Some local runners look for ``submission.agent`` instead of importing
``main.agent`` directly. Re-exporting the function keeps those workflows pointed
at the same native/JIT code path.
"""

from main import agent

__all__ = ["agent"]
