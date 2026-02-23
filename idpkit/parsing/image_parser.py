"""Image parser with optional OCR via pytesseract."""

import logging
import os

from .base import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """Extract text from images using OCR.

    Attempts to use ``pytesseract`` for optical character recognition.  If
    pytesseract is not installed, returns a placeholder message indicating
    that OCR is unavailable.
    """

    def supported_extensions(self) -> list[str]:
        return ["png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "gif"]

    def parse(self, file_path: str) -> ParseResult:
        """Parse an image file and return OCR text (if available).

        Parameters
        ----------
        file_path:
            Path to the image file.
        """
        filename = os.path.basename(file_path)

        text = self._try_ocr(file_path)

        metadata = {
            "filename": filename,
            "format": "image",
        }

        # Try to get image dimensions if PIL is available.
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["mode"] = img.mode
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Could not read image metadata from %s: %s", file_path, exc)

        pages = [{"page": 1, "text": text}]

        logger.info("Image parsing complete: %s", file_path)

        return ParseResult(
            text=text,
            pages=pages,
            metadata=metadata,
            page_count=1,
        )

    @staticmethod
    def _try_ocr(file_path: str) -> str:
        """Attempt OCR on the image; return fallback message on failure."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(file_path)
            text = pytesseract.image_to_string(img).strip()
            if text:
                return text
            return "(OCR returned no text for this image)"
        except ImportError:
            return (
                "[OCR not available] pytesseract and/or Pillow are not installed. "
                "Install them with: pip install pytesseract Pillow\n"
                "Tesseract OCR engine must also be installed on the system."
            )
        except Exception as exc:
            logger.warning("OCR failed for %s: %s", file_path, exc)
            return f"[OCR failed] {exc}"
