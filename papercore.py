"""
PaperCore: Convert academic PDFs to structured, compressed Markdown.

Uses a "Three-Zone" strategy:
  Zone A (Metadata)   - Title, Authors, Year
  Zone B (Full Text)  - Abstract, Introduction, Discussion, Conclusion
  Zone C (Compressed) - Methods, Materials, Results, Experiments
                        (keeps subheaders, captions, first sentences only)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel, GroupLabel

logger = logging.getLogger("papercore")

# ---------------------------------------------------------------------------
# Zone definitions and section-name synonyms
# ---------------------------------------------------------------------------


class Zone(Enum):
    A = "metadata"
    B = "full_retention"
    C = "compression"
    UNKNOWN = "unknown"  # treated as Zone B (safe default)


# Zone B: sections where ALL text is retained
ZONE_B_PATTERNS: dict[str, list[str]] = {
    "abstract": ["abstract", "summary", "executive summary"],
    "introduction": [
        "introduction",
        "background",
        "overview",
        "motivation",
    ],
    "discussion": ["discussion", "implications", "interpretation"],
    "conclusion": [
        "conclusion",
        "conclusions",
        "concluding remarks",
        "final remarks",
        "summary and conclusion",
        "summary and conclusions",
        "limitations",
        "future work",
        "future directions",
    ],
    "related_work": ["related work", "literature review", "prior work"],
}

# Zone C: sections where smart compression is applied
ZONE_C_PATTERNS: dict[str, list[str]] = {
    "methods": [
        "method",
        "methods",
        "methodology",
        "approach",
        "procedure",
        "procedures",
        "experimental setup",
        "experimental design",
        "study design",
        "research design",
        "proposed method",
        "proposed approach",
        "our approach",
        "framework",
        "algorithm",
    ],
    "materials": [
        "materials",
        "materials and methods",
        "data",
        "dataset",
        "datasets",
        "data collection",
        "participants",
        "subjects",
        "sample",
        "instruments",
        "measures",
        "apparatus",
    ],
    "results": [
        "results",
        "result",
        "findings",
        "observations",
        "experimental results",
        "empirical results",
        "quantitative results",
        "qualitative results",
        "analysis",
        "evaluation",
        "experiments",
        "performance",
        "comparison",
    ],
    "implementation": [
        "implementation",
        "implementation details",
        "technical details",
        "system architecture",
        "setup",
        "configuration",
        "hyperparameters",
        "training",
        "training details",
    ],
}


# ---------------------------------------------------------------------------
# SectionClassifier
# ---------------------------------------------------------------------------


class SectionClassifier:
    """Classifies section headers into Zone B or Zone C."""

    def __init__(
        self,
        zone_b_patterns: dict[str, list[str]] = ZONE_B_PATTERNS,
        zone_c_patterns: dict[str, list[str]] = ZONE_C_PATTERNS,
    ):
        self._zone_b_set: set[str] = set()
        for synonyms in zone_b_patterns.values():
            for s in synonyms:
                self._zone_b_set.add(self._normalize(s))

        self._zone_c_set: set[str] = set()
        for synonyms in zone_c_patterns.values():
            for s in synonyms:
                self._zone_c_set.add(self._normalize(s))

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize header text: strip numbering, lowercase, trim."""
        text = text.strip().lower()
        # Remove leading section numbers: "3.", "3.1.", "III.", "A.", "1)"
        text = re.sub(
            r"^(?:\d+\.?\d*\.?\s*|[ivxlc]+\.?\s*|[a-z]\)?\s*)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Remove trailing colon or period
        text = re.sub(r"[:.]\s*$", "", text)
        return text.strip()

    def classify(self, header_text: str) -> Zone:
        """Classify a section header into a zone.

        Strategy (in order):
          1. Exact match against normalized synonym sets
          2. Containment match (pattern in header or header in pattern)
          3. Default to UNKNOWN (treated as Zone B)

        Zone B is checked before Zone C so that ambiguous headers like
        "Results and Discussion" are preserved in full.
        """
        normalized = self._normalize(header_text)

        # Tier 1: exact match
        if normalized in self._zone_b_set:
            return Zone.B
        if normalized in self._zone_c_set:
            return Zone.C

        # Tier 2: containment (Zone B checked first for safety)
        for pattern in self._zone_b_set:
            if pattern in normalized or normalized in pattern:
                return Zone.B
        for pattern in self._zone_c_set:
            if pattern in normalized or normalized in pattern:
                return Zone.C

        return Zone.UNKNOWN


# ---------------------------------------------------------------------------
# MetadataExtractor (Zone A)
# ---------------------------------------------------------------------------


@dataclass
class PaperMetadata:
    title: str = ""
    authors: str = ""
    year: str = ""


class MetadataExtractor:
    """Extracts title, authors, and year from the document preamble."""

    @staticmethod
    def extract(doc, items_iterator) -> tuple[PaperMetadata, list]:
        """Consume leading items up to the first content SECTION_HEADER.

        The first section_header is treated as the title if no TITLE label
        exists (common in academic PDFs where docling misclassifies the
        paper title as a section header).

        Returns (metadata, remaining_items).
        """
        metadata = PaperMetadata()
        remaining: list = []
        found_title = False
        found_content_section = False
        preamble_texts: list[str] = []

        # Non-content headers that appear before the real body
        _skip_headers = {
            "key words", "keywords", "key-words",
        }

        for item, level in items_iterator:
            label = getattr(item, "label", None)

            if found_content_section:
                remaining.append((item, level))
                continue

            if label == DocItemLabel.TITLE:
                metadata.title = item.text.strip()
                found_title = True

            elif label == DocItemLabel.SECTION_HEADER:
                header_lower = item.text.strip().lower()

                # First section_header -> use as title if none found yet
                if not found_title:
                    metadata.title = item.text.strip()
                    found_title = True
                    continue

                # Skip non-content headers in the preamble
                if header_lower in _skip_headers:
                    continue

                # "Abstract" header: mark as start of content
                if "abstract" in header_lower:
                    found_content_section = True
                    remaining.append((item, level))
                    continue

                # Any other section_header -> start of body
                found_content_section = True
                remaining.append((item, level))

            elif label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH):
                preamble_texts.append(item.text.strip())

        metadata.authors = MetadataExtractor._extract_authors(preamble_texts)
        metadata.year = MetadataExtractor._extract_year(preamble_texts)

        return metadata, remaining

    @staticmethod
    def _extract_year(texts: list[str]) -> str:
        for text in texts:
            match = re.search(r"\b((?:19|20)\d{2})\b", text)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _extract_authors(texts: list[str]) -> str:
        for text in texts:
            # Skip full sentences (likely abstract or body text)
            if ". " in text and len(text) > 150:
                continue
            if len(text) < 500 and ("," in text or " and " in text.lower()):
                return text
        return texts[0] if texts else ""


# ---------------------------------------------------------------------------
# ZoneRenderer
# ---------------------------------------------------------------------------


class ZoneRenderer:
    """Renders document items according to zone rules.

    Zone B / UNKNOWN: render everything faithfully.
    Zone C: keep subheaders, captions, and first sentence of each paragraph.
    """

    def __init__(self, doc):
        self._doc = doc

    def render_item(self, item, level: int, zone: Zone) -> Optional[str]:
        label = getattr(item, "label", None)

        # Skip structural group nodes (GroupLabel.SECTION, etc.)
        if label is None or isinstance(label, GroupLabel):
            return None

        if zone in (Zone.B, Zone.UNKNOWN):
            return self._render_full(item, label)
        if zone == Zone.C:
            return self._render_compressed(item, label)
        return None

    @staticmethod
    def _heading_level(item) -> int:
        """Infer heading level from docling's level and text casing.

        Many PDFs produce all level-1 headers. In academic papers,
        ALL-CAPS headers are main sections (## level 2) and
        mixed-case headers are subsections (### level 3).
        """
        raw = getattr(item, "level", 1)
        if raw > 1:
            return min(raw + 1, 6)  # shift: docling level 1 -> md ##
        # All level-1: infer from casing
        text = item.text.strip()
        # Remove leading numbers/punctuation for casing check
        clean = re.sub(r"^[\d.\s]+", "", text)
        if clean and clean == clean.upper() and any(c.isalpha() for c in clean):
            return 2  # ALL CAPS -> main section
        return 3  # Mixed case -> subsection

    # -- Zone B rendering ---------------------------------------------------

    def _render_full(self, item, label) -> Optional[str]:
        if label == DocItemLabel.SECTION_HEADER:
            h = "#" * self._heading_level(item)
            return f"{h} {item.text}"

        if label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH):
            return item.text

        if label == DocItemLabel.LIST_ITEM:
            return f"- {item.text}"

        if label == DocItemLabel.TABLE:
            return self._render_table(item, full=True)

        if label == DocItemLabel.PICTURE:
            return self._render_figure(item)

        if label == DocItemLabel.CAPTION:
            return f"*{item.text}*"

        if label == DocItemLabel.FORMULA:
            return f"$${item.text}$$"

        if label == DocItemLabel.FOOTNOTE:
            return f"[^]: {item.text}"

        if label == DocItemLabel.REFERENCE:
            return item.text

        if label == DocItemLabel.CODE:
            return f"```\n{item.text}\n```"

        # Fallback for unknown labels with text
        if hasattr(item, "text") and item.text:
            return item.text
        return None

    # -- Zone C rendering ---------------------------------------------------

    def _render_compressed(self, item, label) -> Optional[str]:
        if label == DocItemLabel.SECTION_HEADER:
            h = "#" * self._heading_level(item)
            return f"{h} {item.text}"

        if label == DocItemLabel.CAPTION:
            return f"*{item.text}*"

        if label == DocItemLabel.PICTURE:
            return self._render_figure(item)

        if label == DocItemLabel.TABLE:
            return self._render_table(item, full=False)

        if label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH):
            first = self._extract_first_sentence(item.text)
            if not first:
                return None
            if first != item.text:
                return first + " [...]"
            return first

        if label == DocItemLabel.LIST_ITEM:
            first = self._extract_first_sentence(item.text)
            return f"- {first}" if first else None

        if label == DocItemLabel.FORMULA:
            return f"$${item.text}$$"

        return None

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        if not text:
            return ""
        # Match first sentence, avoiding splits on abbreviations / decimals
        match = re.match(
            r"(.+?"
            r"(?<!\b[A-Z])"
            r"(?<!\bet)"
            r"(?<!\bal)"
            r"(?<!\bvs)"
            r"(?<!\bDr)"
            r"(?<!\bMr)"
            r"(?<!\bMs)"
            r"(?<!\bFig)"
            r"(?<!\bEq)"
            r"(?<!\betc)"
            r"(?<!\bi\.e)"
            r"(?<!\be\.g)"
            r"(?<!\d)"
            r"[.!?]"
            r")\s",
            text + " ",  # trailing space so single-sentence paragraphs match
        )
        if match:
            return match.group(1).strip()
        if len(text) > 200:
            return text[:200].rsplit(" ", 1)[0] + " ..."
        return text

    def _render_table(self, item, full: bool) -> str:
        caption = self._get_caption(item)
        if full:
            try:
                df = item.export_to_dataframe(self._doc)
                md = df.to_markdown(index=False)
                if caption:
                    return f"*{caption}*\n\n{md}"
                return md
            except Exception:
                pass
        # Compressed or fallback: caption only
        if caption:
            return f"*[Table: {caption}]*"
        return "*[Table]*"

    def _render_figure(self, item) -> str:
        caption = self._get_caption(item)
        if caption:
            return f"**[Figure: {caption}]**"
        return "**[Figure]**"

    def _get_caption(self, item) -> str:
        try:
            return item.caption_text(self._doc).strip()
        except (AttributeError, Exception):
            return ""


# ---------------------------------------------------------------------------
# PaperConverter – main pipeline
# ---------------------------------------------------------------------------


class PaperConverter:
    """Converts academic PDFs to three-zone compressed Markdown."""

    def __init__(self):
        pipeline_opts = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=False,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_opts,
                ),
            }
        )
        self._classifier = SectionClassifier()

    def convert_folder(
        self, input_dir: Path, output_dir: Optional[Path] = None
    ) -> None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        if not input_dir.is_dir():
            logger.error(f"Not a directory: {input_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted(input_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return

        logger.info(f"Found {len(pdf_files)} PDF(s) to process")

        for pdf_path in pdf_files:
            md_path = output_dir / f"{pdf_path.stem}.md"
            logger.info(f"Processing: {pdf_path.name}")
            try:
                markdown = self.convert_single(pdf_path)
                md_path.write_text(markdown, encoding="utf-8")
                logger.info(f"  -> {md_path.name}")
            except Exception as e:
                logger.error(f"  FAILED: {pdf_path.name}: {e}")
                self._fallback_convert(pdf_path, md_path)

    def convert_single(self, pdf_path: Path) -> str:
        """Convert one PDF through the three-zone pipeline."""
        result = self._converter.convert(str(pdf_path))

        if result.status == ConversionStatus.FAILURE:
            msgs = [e.error_message for e in result.errors]
            raise RuntimeError(f"Docling conversion failed: {msgs}")

        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            logger.warning(
                f"Partial conversion: "
                f"{[e.error_message for e in result.errors]}"
            )

        doc = result.document

        # Check for meaningful structure
        if not self._has_structure(doc):
            logger.warning(
                f"No section structure in {pdf_path.name}, "
                f"falling back to full text."
            )
            return self._unstructured_fallback(doc, pdf_path)

        # Zone A: extract metadata from preamble
        items_iter = doc.iterate_items()
        metadata, remaining = MetadataExtractor.extract(doc, items_iter)

        # Walk body items, classify sections, render by zone
        renderer = ZoneRenderer(doc)
        parts: list[str] = [self._render_metadata(metadata)]

        current_zone = Zone.B  # default before any header
        for item, level in remaining:
            label = getattr(item, "label", None)

            if label == DocItemLabel.SECTION_HEADER:
                current_zone = self._classifier.classify(item.text)
                logger.debug(f"  [{current_zone.name}] {item.text}")

            rendered = renderer.render_item(item, level, current_zone)
            if rendered is not None:
                parts.append(rendered)

        return "\n\n".join(p for p in parts if p)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _has_structure(doc) -> bool:
        for item, _ in doc.iterate_items():
            if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
                return True
        return False

    @staticmethod
    def _render_metadata(meta: PaperMetadata) -> str:
        parts = []
        if meta.title:
            parts.append(f"# {meta.title}")
        lines = []
        if meta.authors:
            lines.append(f"**Authors:** {meta.authors}")
        if meta.year:
            lines.append(f"**Year:** {meta.year}")
        if lines:
            parts.append("  \n".join(lines))
        parts.append("---")
        return "\n\n".join(parts)

    def _unstructured_fallback(self, doc, pdf_path: Path) -> str:
        try:
            full_md = doc.export_to_markdown()
            warning = (
                "> **Warning:** Structure detection failed for this "
                "document. Full text extracted without three-zone "
                "compression.\n\n---\n"
            )
            return warning + full_md
        except Exception as e:
            logger.error(f"export_to_markdown failed: {e}")
            return self._raw_text_fallback(doc, pdf_path)

    @staticmethod
    def _raw_text_fallback(doc, pdf_path: Path) -> str:
        parts = [
            f"> **Warning:** Minimal text extraction for "
            f"{pdf_path.name}\n\n---"
        ]
        for item, _ in doc.iterate_items():
            if hasattr(item, "text") and item.text:
                parts.append(item.text)
        if len(parts) <= 1:
            parts.append("[No text could be extracted from this document]")
        return "\n\n".join(parts)

    def _fallback_convert(self, pdf_path: Path, md_path: Path) -> None:
        try:
            result = self._converter.convert(str(pdf_path))
            md = self._unstructured_fallback(result.document, pdf_path)
            md_path.write_text(md, encoding="utf-8")
            logger.info(f"  -> {md_path.name} (fallback)")
        except Exception as e2:
            logger.error(f"  Fallback also failed: {e2}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    # Ensure UTF-8 output on Windows (avoids GBK codec errors)
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

    parser = argparse.ArgumentParser(
        prog="papercore",
        description=(
            "PaperCore: Convert academic PDFs to structured, "
            "compressed Markdown using a Three-Zone strategy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Three-Zone Strategy:
  Zone A (Metadata)   - Title, Authors, Year
  Zone B (Full Text)  - Abstract, Introduction, Discussion, Conclusion
  Zone C (Compressed) - Methods, Materials, Results, Experiments
                        (subheaders + captions + first sentences only)

Examples:
  papercore ./papers/
  papercore ./papers/ -o ./markdown_output/
  papercore ./papers/single_paper.pdf
  papercore ./papers/ -v
""",
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a PDF file or a directory of PDFs",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (defaults to same as input)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable Zone C compression (treat all sections as Zone B)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical interface",
    )

    args = parser.parse_args()

    if args.gui:
        from papercore_gui import main as gui_main

        gui_main()
        return

    if args.input is None:
        parser.error("the following arguments are required: input")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    converter = PaperConverter()

    if args.no_compress:
        converter._classifier = SectionClassifier(zone_c_patterns={})

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        out_dir = output_path or input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / f"{input_path.stem}.md"
        try:
            markdown = converter.convert_single(input_path)
            md_path.write_text(markdown, encoding="utf-8")
            logger.info(f"Saved: {md_path}")
        except Exception as e:
            logger.error(f"Failed: {e}")
            sys.exit(1)
    elif input_path.is_dir():
        converter.convert_folder(input_path, output_path)
    else:
        logger.error(f"Input must be a PDF file or directory: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
