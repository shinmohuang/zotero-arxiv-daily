import re
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from loguru import logger
import pymupdf


CAPTION_PATTERN = re.compile(r"\bfig(?:ure)?\.?\s*\d+\b", flags=re.IGNORECASE)
DEFAULT_FRAMEWORK_KEYWORDS = (
    "framework",
    "architecture",
    "overview",
    "pipeline",
    "method",
    "system",
    "approach",
    "model",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _get_text_blocks(page) -> list[tuple[pymupdf.Rect, str]]:
    try:
        blocks = page.get_text("blocks", sort=True)
    except TypeError:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda item: (item[1], item[0]))

    text_blocks = []
    for block in blocks:
        if len(block) < 5:
            continue
        text = _normalize_text(block[4])
        if not text:
            continue
        text_blocks.append((pymupdf.Rect(block[:4]), text))
    return text_blocks


def _is_framework_caption(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    if CAPTION_PATTERN.search(lowered) is None:
        return False
    return any(keyword in lowered for keyword in keywords)


def _has_horizontal_overlap(left: pymupdf.Rect, right: pymupdf.Rect, threshold: float = 0.25) -> bool:
    overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    return overlap >= min(left.width, right.width) * threshold


def _get_column_rect(page_rect: pymupdf.Rect, caption_rect: pymupdf.Rect, margin: float) -> pymupdf.Rect:
    page_width = page_rect.width
    if page_width <= 0:
        return page_rect

    caption_width_ratio = caption_rect.width / page_width
    if caption_width_ratio >= 0.55:
        return pymupdf.Rect(
            page_rect.x0 + margin,
            page_rect.y0 + margin,
            page_rect.x1 - margin,
            page_rect.y1 - margin,
        )

    page_mid = page_rect.x0 + page_width / 2
    caption_mid = (caption_rect.x0 + caption_rect.x1) / 2
    if caption_mid <= page_mid:
        return pymupdf.Rect(
            page_rect.x0 + margin,
            page_rect.y0 + margin,
            min(page_mid + margin, page_rect.x1 - margin),
            page_rect.y1 - margin,
        )
    return pymupdf.Rect(
        max(page_mid - margin, page_rect.x0 + margin),
        page_rect.y0 + margin,
        page_rect.x1 - margin,
        page_rect.y1 - margin,
    )


def _clip_between_blocks(
    column_rect: pymupdf.Rect,
    caption_rect: pymupdf.Rect,
    blocks: list[tuple[pymupdf.Rect, str]],
    min_width: float,
    min_height: float,
    margin: float,
) -> pymupdf.Rect | None:
    upper_bound = column_rect.y0
    for block_rect, _ in blocks:
        if block_rect.y1 > caption_rect.y0:
            continue
        if not _has_horizontal_overlap(block_rect, column_rect):
            continue
        upper_bound = max(upper_bound, block_rect.y1)

    above_clip = pymupdf.Rect(
        column_rect.x0,
        upper_bound + margin,
        column_rect.x1,
        caption_rect.y0 - margin,
    )
    if above_clip.width >= min_width and above_clip.height >= min_height:
        return above_clip

    lower_bound = column_rect.y1
    for block_rect, _ in blocks:
        if block_rect.y0 < caption_rect.y1:
            continue
        if not _has_horizontal_overlap(block_rect, column_rect):
            continue
        lower_bound = min(lower_bound, block_rect.y0)
        break

    below_clip = pymupdf.Rect(
        column_rect.x0,
        caption_rect.y1 + margin,
        column_rect.x1,
        lower_bound - margin,
    )
    if below_clip.width >= min_width and below_clip.height >= min_height:
        return below_clip
    return None


def extract_framework_figure(
    file_path: str,
    *,
    max_pages: int = 8,
    zoom: float = 2.0,
    min_width: float = 160.0,
    min_height: float = 120.0,
    caption_margin: float = 12.0,
    keywords: tuple[str, ...] = DEFAULT_FRAMEWORK_KEYWORDS,
) -> bytes | None:
    doc = pymupdf.open(file_path)
    try:
        best_candidate: tuple[float, int, pymupdf.Rect] | None = None
        total_pages = min(max_pages, len(doc))
        for page_index in range(total_pages):
            page = doc[page_index]
            blocks = _get_text_blocks(page)
            page_rect = page.rect
            page_area = max(page_rect.get_area(), 1.0)
            for caption_rect, text in blocks:
                if not _is_framework_caption(text, keywords):
                    continue

                column_rect = _get_column_rect(page_rect, caption_rect, caption_margin)
                clip = _clip_between_blocks(
                    column_rect,
                    caption_rect,
                    blocks,
                    min_width=min_width,
                    min_height=min_height,
                    margin=caption_margin,
                )
                if clip is None:
                    continue

                lowered = text.lower()
                keyword_hits = sum(keyword in lowered for keyword in keywords)
                area_score = clip.get_area() / page_area
                score = keyword_hits * 100.0 + area_score * 20.0 - page_index
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, page_index, clip)

        if best_candidate is None:
            return None

        _, page_index, clip = best_candidate
        matrix = pymupdf.Matrix(zoom, zoom)
        pixmap = doc[page_index].get_pixmap(matrix=matrix, clip=clip, alpha=False)
        return pixmap.tobytes()
    finally:
        doc.close()


def extract_framework_figure_from_url(
    pdf_url: str,
    *,
    max_pages: int = 8,
    zoom: float = 2.0,
    min_width: float = 160.0,
    min_height: float = 120.0,
    caption_margin: float = 12.0,
    keywords: tuple[str, ...] = DEFAULT_FRAMEWORK_KEYWORDS,
) -> bytes | None:
    with TemporaryDirectory() as temp_dir:
        pdf_path = f"{temp_dir}/paper.pdf"
        logger.debug(f"Downloading PDF for framework figure extraction: {pdf_url}")
        urlretrieve(pdf_url, pdf_path)
        return extract_framework_figure(
            pdf_path,
            max_pages=max_pages,
            zoom=zoom,
            min_width=min_width,
            min_height=min_height,
            caption_margin=caption_margin,
            keywords=keywords,
        )
