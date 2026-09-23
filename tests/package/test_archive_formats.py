"""Archive formats and nested extraction used by real Zenodo submissions."""

import io
import tarfile
import zipfile

import pytest
import responses

pytestmark = [pytest.mark.nodata, pytest.mark.min]


def tar_bytes(name, payload, mode="w:gz", kind=tarfile.REGTYPE):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode=mode) as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.size = len(payload) if kind == tarfile.REGTYPE else 0
        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            member.linkname = "elsewhere"
        archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


@pytest.mark.parametrize("mode", ["w", "w:gz"])
@responses.activate
def test_tar_download_and_readme_redownload(tmp_path, mode):
    import yaml
    from fairmd.lipids import databankio as dio

    # Record 55318 calls a plain TAR sim_files.tar.gz; detect content, not suffix.
    payload = tar_bytes("./rep1/run.xtc", b"trajectory", mode)
    url = "https://zenodo.org/records/123/files/sim_files.tar.gz"
    responses.add(responses.GET, url, body=payload, headers={"Content-Length": str(len(payload))})
    responses.add(responses.GET, "https://zenodo.org/api/records/123", json={"files": [{"key": "sim_files.tar.gz"}]})
    sim = {"DOI": "10.5281/zenodo.123", "TRJ": [["sim_files.tar.gz/rep1/run.xtc"]]}
    sources = dio.prepare_file_sources(sim, ["TRJ"])
    dio.download_resource_from_uri(url, str(tmp_path / "import/run.xtc"), source_path=sources["run.xtc"])
    saved = yaml.safe_load(yaml.safe_dump(sim))
    dio.download_system_file(saved, "run.xtc", str(tmp_path / "reanalysis/run.xtc"))
    assert (tmp_path / "import/run.xtc").read_bytes() == b"trajectory"
    assert (tmp_path / "reanalysis/run.xtc").read_bytes() == b"trajectory"


def test_nested_zip_tar_extracts_exact_destination(tmp_path):
    from fairmd.lipids import databankio as dio

    archive = tmp_path / "outer.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("folder/inner.tar.gz", tar_bytes("rep1/run.xtc", b"trajectory"))
    target = tmp_path / "renamed.xtc"
    dio._extract_zip_member(str(archive), "folder/inner.tar.gz/rep1/run.xtc", str(target))
    assert target.read_bytes() == b"trajectory"


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_tar_nonregular_member_preserves_existing_destination(tmp_path, kind):
    from fairmd.lipids import databankio as dio

    archive = tmp_path / "data.tar.gz"
    archive.write_bytes(tar_bytes("run.xtc", b"", kind=kind))
    target = tmp_path / "run.xtc"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError, match="regular file"):
        dio._extract_zip_member(str(archive), "run.xtc", str(target))
    assert target.read_bytes() == b"existing"


def test_failed_extraction_preserves_existing_destination(tmp_path):
    from fairmd.lipids import databankio as dio

    target = tmp_path / "run.xtc"
    target.write_bytes(b"existing")

    def broken_blocks():
        yield b"partial"
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        dio._write_atomic(target, broken_blocks())
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_colliding_names_do_not_partially_modify_metadata():
    from fairmd.lipids import databankio as dio

    sim = {"TRJ": [["one.tar.gz/run.xtc"], ["two.tar.gz/run.xtc"]]}
    with pytest.raises(ValueError, match="share local filename"):
        dio.prepare_file_sources(sim, ["TRJ"])
    assert sim == {"TRJ": [["one.tar.gz/run.xtc"], ["two.tar.gz/run.xtc"]]}
