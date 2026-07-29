from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from preprocessing import concat_videos


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


def _write_xml(path: Path, acquisition_date: str) -> Path:
    xml_path = path.with_suffix(".xml")
    xml_path.write_text(
        "\n".join(
            [
                '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">',
                '  <Image ID="Image:0">',
                f"    <AcquisitionDate>{acquisition_date}</AcquisitionDate>",
                "  </Image>",
                "</OME>",
            ]
        ),
        encoding="utf-8",
    )
    return xml_path


def test_section_type_is_canonical():
    assert concat_videos._section_type(Path("2-1.tiff")) == "baseline"
    assert concat_videos._section_type(Path("2-1_LOW_2m.tiff")) == "treatment"
    assert concat_videos._section_type(Path("2-1_HIGH_42m_4m.tiff")) == "treatment"
    assert concat_videos._section_type(Path("2-1_1hr-Recovery.tiff")) == "recovery"
    assert concat_videos._section_type(Path("2-1_RECOVERY.tiff")) == "recovery"


def test_find_sets_orders_members_by_acquisition_date_not_mtime(tmp_path: Path):
    root = tmp_path / "input"
    folder = root / "exp" / "drug" / "day1"

    baseline = _touch(folder / "2-1.tiff")
    member_late = _touch(folder / "2-1_LATE.tiff")
    member_early = _touch(folder / "2-1_EARLY.tiff")
    recovery = _touch(folder / "2-1_1hr-Recovery.tiff")

    _write_xml(member_late, "2026-04-17T19:34:53.454")
    _write_xml(member_early, "2026-04-17T19:11:28.136")
    _write_xml(recovery, "2026-04-17T20:00:00.000")

    # Deliberately invert mtime ordering to prove XML order wins.
    baseline.touch()
    member_late.touch()
    recovery.touch()
    member_early.touch()

    sets = concat_videos.find_sets(root)

    assert len(sets) == 1
    assert sets[0]["baseline"] == baseline
    assert [path.name for path in sets[0]["members"]] == [
        "2-1_EARLY.tiff",
        "2-1_LATE.tiff",
        "2-1_1hr-Recovery.tiff",
    ]


def test_find_sets_raises_when_member_xml_missing(tmp_path: Path):
    root = tmp_path / "input"
    folder = root / "exp" / "drug" / "day1"

    _touch(folder / "2-1.tiff")
    _touch(folder / "2-1_LOW_2m.tiff")

    with pytest.raises(FileNotFoundError, match="Could not find XML metadata"):
        concat_videos.find_sets(root)


def test_read_acquisition_date_raises_for_missing_field(tmp_path: Path):
    tif_path = _touch(tmp_path / "2-1_LOW_2m.tiff")
    tif_path.with_suffix(".xml").write_text(
        "\n".join(
            [
                '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">',
                '  <Image ID="Image:0"></Image>',
                "</OME>",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing AcquisitionDate"):
        concat_videos._read_acquisition_date(tif_path)


def test_concat_set_writes_canonical_section_types_in_metadata_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    baseline = _touch(tmp_path / "2-1.tiff")
    member_early = _touch(tmp_path / "2-1_LOW_2m.tiff")
    member_late = _touch(tmp_path / "2-1_HIGH_42m_4m.tiff")
    recovery = _touch(tmp_path / "2-1_1hr-Recovery.tiff")

    set_dict = {
        "grandparent": Path("exp/drug"),
        "pair_id": "2-1",
        "baseline": baseline,
        "members": [member_early, member_late, recovery],
    }

    fake_frames = {
        baseline: [[[1]]],
        member_early: [[[2]]],
        member_late: [[[3]]],
        recovery: [[[4]]],
    }

    monkeypatch.setattr(
        concat_videos,
        "_read_stack",
        lambda path: __import__("numpy").asarray(fake_frames[path]),
    )
    monkeypatch.setattr(concat_videos.tifffile, "imwrite", lambda *args, **kwargs: None)

    output_paths = concat_videos.ConcatOutputPaths(
        tiff_path=tmp_path / "2-1_concat.tiff",
        order_csv_path=tmp_path / "2-1_concat_order.csv",
        metadata_xml_path=tmp_path / "2-1_concat_metadata.xml",
    )
    concat_videos.concat_set(set_dict, output_paths)

    df = pd.read_csv(output_paths.order_csv_path)
    assert df["source file name"].tolist() == [
        "2-1.tiff",
        "2-1_LOW_2m.tiff",
        "2-1_HIGH_42m_4m.tiff",
        "2-1_1hr-Recovery.tiff",
    ]
    assert df["section type"].tolist() == [
        "baseline",
        "treatment",
        "treatment",
        "recovery",
    ]

    tree = ET.parse(output_paths.metadata_xml_path)
    root = tree.getroot()
    assert root.tag == "ConcatMetadata"
    assert root.attrib["pair_id"] == "2-1"

    source_entries = root.find("sources")
    assert source_entries is not None
    keyed_names = [entry.attrib["source_file_name"] for entry in source_entries]
    assert keyed_names == [
        "2-1.tiff",
        "2-1_LOW_2m.tiff",
        "2-1_HIGH_42m_4m.tiff",
        "2-1_1hr-Recovery.tiff",
    ]
    assert source_entries[0].attrib["section_type"] == "baseline"
    assert source_entries[1].attrib["section_type"] == "treatment"
    assert source_entries[3].attrib["section_type"] == "recovery"


def test_dry_run_prints_metadata_order_details(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    baseline = _touch(tmp_path / "2-1.tiff")
    member = _touch(tmp_path / "2-1_LOW_2m.tiff")
    _write_xml(member, "2026-04-17T19:11:28.136")

    set_dict = {
        "grandparent": Path("exp/drug"),
        "pair_id": "2-1",
        "baseline": baseline,
        "members": [member],
    }

    output_paths = concat_videos.ConcatOutputPaths(
        tiff_path=tmp_path / "out.tiff",
        order_csv_path=tmp_path / "out.csv",
        metadata_xml_path=tmp_path / "out_metadata.xml",
    )
    concat_videos.dry_run_set(set_dict, output_paths)
    output = capsys.readouterr().out

    assert "acquisition: 2026-04-17T19:11:28.136" in output
    assert "section type: treatment" in output
    assert "Would write metadata XML" in output
