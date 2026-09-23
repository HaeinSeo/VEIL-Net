import csv
import re
import struct
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

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


@pytest.mark.parametrize("name", ["README.md", "docs/RESULTS.md", "docs/TRAINING.md"])
def test_public_document_links_resolve(name):
    path = ROOT / name
    text = path.read_text(encoding="utf-8")
    parser = DocumentLinks()
    parser.feed(text)
    targets = parser.targets + re.findall(r"\]\(([^\s)]+)\)", text)
    for target in targets:
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        resolved = (path.parent / unquote(url.path)).resolve()
        assert resolved.is_relative_to(ROOT), target
        assert resolved.is_file(), target
    for image in parser.images:
        assert image.get("alt", "").strip()


def test_readme_uses_measured_reference_metrics():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    with (ROOT / "benchmarks/validation.csv").open(encoding="utf-8", newline="") as stream:
        reference = next(row for row in csv.DictReader(stream) if row["model"] == "rapc_coverage_balance")
    row = next(line for line in text.splitlines() if line.startswith("| **VEIL-Net** |"))
    metrics = ("cd_l1", "cd_l2", "f_score_003", "f_score_005", "dimension_mae")
    values = [float(cell.strip().strip("*")) for cell in row.split("|")[2:-1]]
    assert values == pytest.approx([float(reference[key]) for key in metrics], abs=0.000005)
    assert "not pretrained weights or datasets" in text
    assert "docs/RESULTS.md" in text


def test_supplied_figures_are_present_and_readable_pngs():
    parser = DocumentLinks()
    parser.feed((ROOT / "README.md").read_text(encoding="utf-8"))
    paths = {image["src"] for image in parser.images if image["src"].startswith("figures/")}
    assert paths == {
        "figures/veil-mascot.png",
        "figures/scene-completion.png",
        "figures/qualitative-comparison.png",
    }
    for name in paths:
        header = (ROOT / name).read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        assert header[12:16] == b"IHDR"
        width, height = struct.unpack(">II", header[16:24])
        assert width >= 400 and height >= 400
