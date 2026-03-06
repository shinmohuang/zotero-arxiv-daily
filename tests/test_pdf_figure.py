import pytest
import base64


pymupdf = pytest.importorskip("pymupdf")

from zotero_arxiv_daily.pdf_figure import extract_framework_figure


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9/rfUAAAAASUVORK5CYII="
)


def test_extract_framework_figure(tmp_path):
    pdf_path = tmp_path / "framework.pdf"

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Test paper title")
    page.insert_text((72, 120), "Intro paragraph before the figure.")

    diagram_rect = pymupdf.Rect(72, 170, 523, 350)
    page.draw_rect(diagram_rect, color=(0, 0, 0), fill=(0.95, 0.95, 0.95))
    page.insert_text((120, 220), "Encoder")
    page.insert_text((270, 220), "Fusion")
    page.insert_text((420, 220), "Decoder")
    page.insert_text((250, 290), "Input -> Output")

    caption_rect = pymupdf.Rect(72, 360, 523, 400)
    page.insert_textbox(caption_rect, "Figure 1: Overview of the framework.", fontsize=12)

    doc.save(pdf_path)
    doc.close()

    image_bytes = extract_framework_figure(str(pdf_path), max_pages=1)

    assert image_bytes is not None
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_extract_framework_figure_without_figure_prefix(tmp_path):
    pdf_path = tmp_path / "framework-no-prefix.pdf"

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    image_rect = pymupdf.Rect(72, 160, 523, 360)
    page.insert_image(image_rect, stream=PNG_BYTES)
    page.insert_textbox(
        pymupdf.Rect(72, 372, 523, 470),
        (
            "This overview of the proposed architecture summarizes the full workflow of the paper, "
            "including preprocessing, multimodal fusion, and prediction stages for the final system."
        ),
        fontsize=12,
    )

    doc.save(pdf_path)
    doc.close()

    image_bytes = extract_framework_figure(str(pdf_path), max_pages=1)

    assert image_bytes is not None
    assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
