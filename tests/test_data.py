"""Daten-Setup: ZIP-Erkennung, Pruefsumme, Bildzaehlung, Dateiliste."""

import zipfile

from cerebrocam.data import count_images, counts_match, identify_zip, list_images, sha256_of_file


def make_zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    return path


def test_identify_zip_by_folder_structure(tmp_path):
    nick = make_zip(tmp_path / "archive.zip", ["Training/glioma/Tr-gl_1.jpg", "Testing/notumor/Te-no_1.jpg"])
    sartaj = make_zip(tmp_path / "archive (1).zip", ["Training/glioma_tumor/gg (1).jpg"])
    other = make_zip(tmp_path / "other.zip", ["images/cat.jpg"])
    assert identify_zip(nick) == "nickparvar"
    assert identify_zip(sartaj) == "sartaj"
    assert identify_zip(other) is None


def test_sha256_of_known_content(tmp_path):
    f = tmp_path / "abc.txt"
    f.write_bytes(b"abc")
    assert sha256_of_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_count_images_ignores_non_images(tmp_path):
    for rel in ["Training/a/1.jpg", "Training/a/2.PNG", "Training/a/notes.txt", "Testing/a/3.jpg"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    counts = count_images(tmp_path)
    assert counts == {"Testing": {"a": 1}, "Training": {"a": 2}}
    assert counts_match(counts, {"Training": {"a": 2}, "Testing": {"a": 1}})
    assert not counts_match(counts, {"Training": {"a": 3}, "Testing": {"a": 1}})


def test_count_images_of_missing_folder(tmp_path):
    assert count_images(tmp_path / "missing") == {}


def test_list_images_is_sorted_and_filtered(tmp_path):
    for rel in ["b.jpg", "a.PNG", "c.txt", "sub/d.jpeg"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    assert [p.name for p in list_images(tmp_path)] == ["a.PNG", "b.jpg"]
    assert [p.relative_to(tmp_path).as_posix() for p in list_images(tmp_path, recursive=True)] == [
        "a.PNG",
        "b.jpg",
        "sub/d.jpeg",
    ]
