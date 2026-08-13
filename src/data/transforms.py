"""Stage 1 transform boundary.

No preprocessing transforms are implemented until the user approves parameters
based on the inspection report. This module prevents accidental defaults.
"""

from __future__ import annotations


class PreprocessingNotApprovedError(RuntimeError):
    """Raised when preprocessing is attempted before Stage 1 approval."""


def build_preprocessing_transforms(*args: object, **kwargs: object) -> None:
    raise PreprocessingNotApprovedError(
        "Stage 1 performs inspection only. Approve dataset-specific preprocessing "
        "parameters before creating MONAI transforms."
    )
