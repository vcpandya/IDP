"""Processing pipeline — chainable document processing steps."""

import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Type alias for pipeline step functions.
StepFn = Callable[[dict], Coroutine[Any, Any, dict]]


class Pipeline:
    """A chainable document processing pipeline.

    Steps are added with :meth:`add_step` and executed sequentially via
    :meth:`run`.  Each step receives a dict and must return a dict; the
    output of one step becomes the input of the next.

    Usage::

        pipeline = Pipeline("my-pipeline")
        pipeline.add_step("parse", parse_step)
        pipeline.add_step("extract", extract_step)
        pipeline.add_step("summarize", summarize_step)

        result = await pipeline.run({"file_path": "/path/to/doc.pdf"})
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._steps: list[tuple[str, StepFn]] = []

    def add_step(self, name: str, fn: StepFn) -> "Pipeline":
        """Append a processing step.

        Parameters
        ----------
        name:
            A human-readable name for this step (used in logging).
        fn:
            An async callable that receives a ``dict`` and returns a ``dict``.

        Returns
        -------
        Pipeline
            ``self``, for fluent chaining.
        """
        self._steps.append((name, fn))
        return self

    async def run(self, input_data: dict) -> dict:
        """Execute all steps in order.

        Parameters
        ----------
        input_data:
            The initial data dict passed to the first step.

        Returns
        -------
        dict
            The final output dict.  Also includes ``"_pipeline"`` metadata
            with the pipeline name and step execution log.
        """
        logger.info("Pipeline '%s' starting with %d steps", self.name, len(self._steps))

        data = dict(input_data)  # shallow copy
        step_log: list[dict] = []

        for idx, (step_name, fn) in enumerate(self._steps, start=1):
            logger.info(
                "Pipeline '%s' — step %d/%d: %s",
                self.name, idx, len(self._steps), step_name,
            )
            try:
                data = await fn(data)
                step_log.append({"step": step_name, "status": "ok"})
            except Exception as exc:
                logger.error(
                    "Pipeline '%s' — step '%s' failed: %s",
                    self.name, step_name, exc,
                )
                step_log.append({"step": step_name, "status": "error", "error": str(exc)})
                data["_error"] = str(exc)
                data["_failed_step"] = step_name
                break

        data["_pipeline"] = {
            "name": self.name,
            "steps": step_log,
            "completed": len(step_log),
            "total": len(self._steps),
        }

        logger.info(
            "Pipeline '%s' finished — %d/%d steps completed",
            self.name, len(step_log), len(self._steps),
        )
        return data

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        step_names = [name for name, _ in self._steps]
        return f"Pipeline(name={self.name!r}, steps={step_names})"
