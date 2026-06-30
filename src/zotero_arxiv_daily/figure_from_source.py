"""从 arXiv 源码包(tar)里直接取"框架图"原图，命中则比从 PDF 渲染区域截图更清晰。

思路：解析 .tex 里的 figure 环境，按图注里的框架关键词挑出目标 figure，取其
\\includegraphics 引用的图片文件，从 tar 里读出来。PNG/JPG 直接用，PDF 矢量图渲染成
PNG；遇到 EPS / TikZ(无外部图片文件)等取不到的情况返回 None，由调用方退回 PDF 抠图。
"""
import re
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from loguru import logger
import pymupdf

from .pdf_figure import DEFAULT_FRAMEWORK_KEYWORDS

_FIGURE_ENV_RE = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.DOTALL)
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_CAPTION_RE = re.compile(r"\\caption\*?")
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:pdf|abs|e-print)/(.+?)(?:\.pdf)?$", re.IGNORECASE)

_RASTER_EXTS = (".png", ".jpg", ".jpeg")
_RENDER_EXTS = (".pdf",)
_RESOLVE_EXTS = _RASTER_EXTS + _RENDER_EXTS


def arxiv_source_url(pdf_url: str | None = None, abs_url: str | None = None) -> str | None:
    """从 pdf_url / 摘要页 url 推出 arXiv 源码下载地址。"""
    for url in (pdf_url, abs_url):
        if not url:
            continue
        match = _ARXIV_ID_RE.search(url)
        if match:
            return f"https://arxiv.org/e-print/{match.group(1)}"
    return None


def _braced_content(text: str, brace_index: int) -> str:
    """返回 text 中位于 brace_index 处 '{' 的配对花括号内容（处理嵌套）。"""
    depth = 0
    for i in range(brace_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_index + 1:i]
    return text[brace_index + 1:]


def _figure_caption(env: str) -> str:
    m = _CAPTION_RE.search(env)
    if not m:
        return ""
    brace = env.find("{", m.end())
    if brace == -1:
        return ""
    return _braced_content(env, brace)


def _read_all_tex(tar: tarfile.TarFile) -> str:
    parts = []
    for name in tar.getnames():
        if not name.endswith(".tex"):
            continue
        try:
            parts.append(tar.extractfile(name).read().decode("utf-8", errors="ignore"))
        except Exception:
            continue
    tex = "\n".join(parts)
    # 去掉行注释（保留 \% 转义），避免被注释掉的 figure 干扰
    return re.sub(r"(?<!\\)%.*", "", tex)


def _ordered_framework_refs(tex: str, keywords: tuple[str, ...]) -> list[str]:
    """返回图注命中框架关键词的 figure 的图片引用，按命中数从高到低。"""
    scored: list[tuple[int, list[str]]] = []
    for m in _FIGURE_ENV_RE.finditer(tex):
        env = m.group(1)
        refs = _INCLUDEGRAPHICS_RE.findall(env)
        if not refs:
            continue
        caption = _figure_caption(env).lower()
        score = sum(keyword in caption for keyword in keywords)
        if score > 0:
            scored.append((score, [r.strip().strip('"') for r in refs]))
    scored.sort(key=lambda item: item[0], reverse=True)
    ordered: list[str] = []
    for _, refs in scored:
        ordered.extend(refs)
    return ordered


def _find_member(names: list[str], ref: str) -> str | None:
    candidates = [ref] + [ref + ext for ext in _RESOLVE_EXTS]
    for cand in candidates:
        for name in names:
            if name == cand or name.endswith("/" + cand):
                return name
    base = ref.rsplit("/", 1)[-1]
    for ext in ("",) + _RESOLVE_EXTS:
        target = base + ext
        for name in names:
            if name.rsplit("/", 1)[-1] == target:
                return name
    return None


def _asset_to_png(member: str, data: bytes, zoom: float) -> bytes | None:
    lower = member.lower()
    if lower.endswith(_RASTER_EXTS):
        return data  # 位图直接可嵌入邮件
    if lower.endswith(_RENDER_EXTS):
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            page = doc.load_page(0)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            return pixmap.tobytes("png")
        finally:
            doc.close()
    return None  # eps/ps 等取不到，交由调用方退回 PDF


def extract_framework_figure_from_tar(
    tar_path: str,
    *,
    title: str | None = None,
    keywords: tuple[str, ...] = DEFAULT_FRAMEWORK_KEYWORDS,
    zoom: float = 2.0,
) -> bytes | None:
    try:
        tar = tarfile.open(tar_path)
    except tarfile.ReadError:
        return None
    with tar:
        names = tar.getnames()
        tex = _read_all_tex(tar)
        if not tex:
            return None
        for ref in _ordered_framework_refs(tex, keywords):
            member = _find_member(names, ref)
            if member is None:
                continue
            try:
                data = tar.extractfile(member).read()
                png = _asset_to_png(member, data, zoom)
            except Exception as e:
                logger.debug(f"Failed to read figure asset {member}: {e}")
                continue
            if png is not None:
                return png
    return None


def extract_framework_figure_from_source_url(
    source_url: str,
    *,
    title: str | None = None,
    keywords: tuple[str, ...] = DEFAULT_FRAMEWORK_KEYWORDS,
    zoom: float = 2.0,
) -> bytes | None:
    with TemporaryDirectory() as temp_dir:
        path = f"{temp_dir}/source.tar.gz"
        try:
            urlretrieve(source_url, path)
        except Exception as e:
            logger.debug(f"Failed to download arxiv source {source_url}: {e}")
            return None
        return extract_framework_figure_from_tar(
            path, title=title, keywords=keywords, zoom=zoom
        )
