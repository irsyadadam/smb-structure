"""Vendored anonymized subset of the smb-utils package.

Only the EHR text formatter is included here. Imaging utilities and the
upstream MEDS-creation pipeline are out of scope for this anonymous
release.
"""

from .ehr_process import process_ehr_info

__all__ = ["process_ehr_info"]
