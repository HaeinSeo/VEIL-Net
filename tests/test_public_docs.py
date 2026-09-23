import re
import struct
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class DocumentLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src"):
            if key in attrs:
                self.targets.append(attrs[key])
        if tag == "img":
            self.images.append(attrs)


def test_readme_links_and_figures_resolve():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    parser = DocumentLinks()
    parser.feed(text)
    for target in parser.targets + re.findall(r"\]\(([^\s)]+)\)", text):
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        path = (ROOT / unquote(url.path)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), target
    assert all(image.get("alt", "").strip() for image in parser.images)
    images = {image["src"] for image in parser.images if image["src"].startswith("figures/")}
    assert images == {
        "figures/veil-mascot.png",
        "figures/scene-completion.png",
        "figures/veil-net-architecture.drawio.png",
        "figures/qualitative-comparison.png",
    }
    for name in images:
        header = (ROOT / name).read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
        assert min(struct.unpack(">II", header[16:24])) >= 400


def test_project_has_one_implementation_package():
    packages = {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    assert packages == {"veil_net"}


def test_evaluation_package_exposes_cli_entrypoints():
    from veil_net.evaluation import evaluate_pairs, select_pairs
    from veil_net.evaluation.metrics import evaluate_completion_torch

    assert callable(evaluate_pairs) and callable(select_pairs)
    assert callable(evaluate_completion_torch)
