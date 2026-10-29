"""Shared test setup.

NoteTaker reads the student's Settings file when it starts. Tests must never
see the developer's own choices (a university reading level, a 45-minute
class), so every run points at a file that does not exist. This runs before
any test module imports notetaker.
"""

import os
import tempfile

os.environ["NOTETAKER_SETTINGS_FILE"] = os.path.join(
    tempfile.mkdtemp(prefix="notetaker-test-settings-"), "settings.json"
)
