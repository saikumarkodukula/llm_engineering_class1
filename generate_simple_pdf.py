"""
Generate a simple PDF from a markdown-like text file without external packages.
"""

from pathlib import Path
import re
import sys


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 54
TOP_MARGIN = 56
BOTTOM_MARGIN = 56
BODY_SIZE = 11
LINE_GAP = 4
MAX_WIDTH = PAGE_WIDTH - 108


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_width_estimate(text: str, font_size: int) -> float:
    return len(text) * font_size * 0.52


def wrap_text(text: str, font_size: int, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if text_width_estimate(candidate, font_size) <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def parse_lines(source_text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    in_code = False

    for raw_line in source_text.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            records.append({"text": line or " ", "font": "F3", "size": 10, "indent": 18, "before": 0, "after": 0})
            continue

        if not line.strip():
            records.append({"text": "", "font": "F1", "size": BODY_SIZE, "indent": 0, "before": 0, "after": 6})
            continue

        if line.startswith("# "):
            records.append({"text": line[2:].strip(), "font": "F2", "size": 20, "indent": 0, "before": 10, "after": 10})
            continue
        if line.startswith("## "):
            records.append({"text": line[3:].strip(), "font": "F2", "size": 14, "indent": 0, "before": 8, "after": 6})
            continue
        if line.startswith("### "):
            records.append({"text": line[4:].strip(), "font": "F2", "size": 12, "indent": 0, "before": 6, "after": 4})
            continue

        if re.match(r"^(- |\d+\. )(.*)$", line):
            wrapped = wrap_text(line.strip(), BODY_SIZE, MAX_WIDTH - 18)
            for idx, wrapped_line in enumerate(wrapped):
                records.append(
                    {"text": wrapped_line, "font": "F1", "size": BODY_SIZE, "indent": 18, "before": 2 if idx == 0 else 0, "after": 0}
                )
            continue

        wrapped = wrap_text(line.strip(), BODY_SIZE, MAX_WIDTH)
        for idx, wrapped_line in enumerate(wrapped):
            records.append(
                {"text": wrapped_line, "font": "F1", "size": BODY_SIZE, "indent": 0, "before": 2 if idx == 0 else 0, "after": 0}
            )

    return records


def build_pages(records: list[dict[str, object]]) -> list[str]:
    pages: list[str] = []
    ops: list[str] = []
    y = PAGE_HEIGHT - TOP_MARGIN

    def flush() -> None:
        nonlocal ops, y
        pages.append("\n".join(ops))
        ops = []
        y = PAGE_HEIGHT - TOP_MARGIN

    for record in records:
        text = str(record["text"])
        font = str(record["font"])
        size = int(record["size"])
        indent = int(record["indent"])
        before = int(record["before"])
        after = int(record["after"])
        line_height = size + LINE_GAP

        y -= before
        if y - line_height - after < BOTTOM_MARGIN:
            flush()

        if text:
            ops.append(
                f"BT /{font} {size} Tf 1 0 0 1 {LEFT_MARGIN + indent} {y} Tm ({escape_pdf_text(text)}) Tj ET"
            )
        y -= line_height + after

    if ops:
        flush()
    return pages


def build_pdf(page_streams: list[str]) -> bytes:
    objects: list[bytes] = []

    def add_object(data: str | bytes) -> int:
        blob = data.encode("latin-1") if isinstance(data, str) else data
        objects.append(blob)
        return len(objects)

    font_regular = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_mono = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    pages_placeholder_index = len(objects)
    objects.append(b"")
    pages_id = len(objects)

    page_ids: list[int] = []
    for stream in page_streams:
        stream_bytes = stream.encode("latin-1")
        content_id = add_object(
            b"<< /Length " + str(len(stream_bytes)).encode("ascii") + b" >>\nstream\n"
            + stream_bytes
            + b"\nendstream"
        )
        page_id = add_object(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R /F3 {font_mono} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_placeholder_index] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n".encode("latin-1"))
    pdf.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    return bytes(pdf)


def main() -> None:
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    records = parse_lines(source.read_text(encoding="utf-8"))
    pages = build_pages(records)
    output.write_bytes(build_pdf(pages))
    print(output.resolve())


if __name__ == "__main__":
    main()
