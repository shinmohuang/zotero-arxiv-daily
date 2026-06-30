import base64
import io
import tarfile

import pytest

pytest.importorskip("pymupdf")

from zotero_arxiv_daily.figure_from_source import (
    extract_framework_figure_from_tar,
    arxiv_source_url,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9/rfUAAAAASUVORK5CYII="
)


def _make_tar(tmp_path, files: dict[str, bytes]) -> str:
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(path)


def test_extracts_framework_figure_asset_from_source(tmp_path):
    tex = (
        r"\documentclass{article}\begin{document}"
        r"\begin{figure}\includegraphics[width=\linewidth]{figures/arch}"
        r"\caption{Overview of the proposed framework.}\end{figure}"
        r"\begin{figure}\includegraphics{figures/loss}"
        r"\caption{Training loss curve.}\end{figure}"
        r"\end{document}"
    )
    tar_path = _make_tar(
        tmp_path,
        {
            "main.tex": tex.encode("utf-8"),
            "figures/arch.png": PNG_BYTES,
            "figures/loss.png": PNG_BYTES,
        },
    )

    figure = extract_framework_figure_from_tar(tar_path, title="A framework")
    # 命中"framework"图注，取到对应的 png 原始字节
    assert figure == PNG_BYTES


def test_returns_none_when_no_framework_caption(tmp_path):
    tex = (
        r"\documentclass{article}\begin{document}"
        r"\begin{figure}\includegraphics{figures/loss}"
        r"\caption{Training loss curve.}\end{figure}"
        r"\end{document}"
    )
    tar_path = _make_tar(
        tmp_path,
        {"main.tex": tex.encode("utf-8"), "figures/loss.png": PNG_BYTES},
    )
    assert extract_framework_figure_from_tar(tar_path) is None


def test_returns_none_for_tikz_only_figure(tmp_path):
    # TikZ 图无 \includegraphics 外部文件 -> 取不到 -> None（由调用方退回 PDF）
    tex = (
        r"\documentclass{article}\begin{document}"
        r"\begin{figure}\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}"
        r"\caption{Overview of the framework architecture.}\end{figure}"
        r"\end{document}"
    )
    tar_path = _make_tar(tmp_path, {"main.tex": tex.encode("utf-8")})
    assert extract_framework_figure_from_tar(tar_path) is None


def test_arxiv_source_url():
    assert arxiv_source_url("https://arxiv.org/pdf/2512.04296") == "https://arxiv.org/e-print/2512.04296"
    assert arxiv_source_url(None, "http://arxiv.org/abs/2512.04296v2") == "https://arxiv.org/e-print/2512.04296v2"
    assert arxiv_source_url("https://example.com/foo.pdf") is None
