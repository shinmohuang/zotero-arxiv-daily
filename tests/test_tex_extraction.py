import io
import tarfile

from zotero_arxiv_daily.utils import extract_tex_code_from_tar, _bm25_pick


def _make_tar(tmp_path, files: dict[str, str]) -> str:
    path = tmp_path / "paper.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(path)


def test_bm25_picks_main_tex_among_multiple_document_blocks(tmp_path):
    # 两个 .tex 都含 \begin{document}（无 bbl 定位）：附录跑题，正文贴合标题
    main = (
        r"\documentclass{article}\begin{document}"
        "Graph neural networks for molecular property prediction. "
        "We propose a graph neural network that predicts molecular properties. "
        r"\end{document}"
    )
    appendix = (
        r"\documentclass{article}\begin{document}"
        "Additional training curves and hyperparameter tables for the experiments. "
        r"\end{document}"
    )
    tar_path = _make_tar(tmp_path, {"main.tex": main, "appendix.tex": appendix})

    result = extract_tex_code_from_tar(
        tar_path, "arxiv:1234.5678",
        paper_title="Graph Neural Networks for Molecular Property Prediction",
    )

    assert result is not None
    assert result["all"] is not None
    assert "molecular properties" in result["all"]
    assert "hyperparameter tables" not in result["all"]


def test_bm25_pick_prefers_title_relevant_candidate():
    candidates = {
        "a": "diffusion model image generation denoising score matching",
        "b": "appendix proofs lemmas notation supplementary derivations",
    }
    assert _bm25_pick("diffusion model for image generation", candidates) == "a"
