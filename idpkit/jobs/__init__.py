"""IDP Kit Jobs — Background job management.

Usage::

    from idpkit.jobs import run_indexing_job

    await run_indexing_job(
        job_id="...",
        doc_id="...",
        file_path="/uploads/report.pdf",
        format=".pdf",
        options={"model": "gpt-4o"},
    )
"""

from .manager import run_indexing_job

__all__ = ["run_indexing_job"]
