from __future__ import annotations

import hashlib
import importlib.metadata
import io
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from cvxgenrust import config
from scripts.verify_release import canonical_stable_version, verify_distributions, write_checksums

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
ACTION_PATTERN = re.compile(r"^\s*(?:-\s*)?uses:\s+([^\s#]+)", re.MULTILINE)
# JavaScript actions use Node.js 24; dtolnay/rust-toolchain is a composite action.
REVIEWED_ACTION_PINS = frozenset(
    {
        "Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6",  # v2.9.2
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",  # v6.1.0
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",  # v8.0.1
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",  # v6.3.0
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",  # v7.0.1
        "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b",  # v8.1.0
        "dtolnay/rust-toolchain@6bed0761d98439e5a578e2877258200ad565ba87",  # stable
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",  # v1.14.2
    }
)


def _source_tree(root: Path) -> Path:
    source = root / "cvxgenrust"
    templates = source / "templates"
    templates.mkdir(parents=True)
    (source / "__init__.py").write_text("from .cgr import generate_code\n", encoding="utf-8")
    (source / "cgr.py").write_text("def generate_code(): ...\n", encoding="utf-8")
    (templates / "runtime.rs.tmpl").write_text("pub struct Runtime;\n", encoding="utf-8")
    return source


def _package_files(source: Path) -> list[Path]:
    return sorted(
        [*source.rglob("*.py"), *(source / "templates").glob("*.tmpl")],
        key=lambda path: path.as_posix(),
    )


def _wheel(
    dist: Path,
    source: Path,
    *,
    version: str = "0.1.0",
    metadata_version: str | None = None,
    tag: str = "py3-none-any",
    include_templates: bool = True,
) -> Path:
    path = dist / f"cvxgenrust-{version}-{tag}.whl"
    metadata = metadata_version or version
    with zipfile.ZipFile(path, "w") as archive:
        for package_file in _package_files(source):
            if not include_templates and package_file.suffix == ".tmpl":
                continue
            relative = package_file.relative_to(source).as_posix()
            archive.writestr(f"cvxgenrust/{relative}", package_file.read_bytes())
        archive.writestr(
            f"cvxgenrust-{version}.dist-info/METADATA",
            (
                "Metadata-Version: 2.4\n"
                "Name: cvxgenrust\n"
                f"Version: {metadata}\n"
                "Requires-Python: >=3.12\n"
                "License-Expression: Apache-2.0\n\n"
            ),
        )
        archive.writestr(
            f"cvxgenrust-{version}.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n\n",
        )
        archive.writestr(f"cvxgenrust-{version}.dist-info/licenses/LICENSE", "Apache License\n")
    return path


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _sdist(
    dist: Path,
    source: Path,
    *,
    version: str = "0.1.0",
    metadata_version: str | None = None,
    project_version: str | None = None,
    extra: str | None = None,
) -> Path:
    path = dist / f"cvxgenrust-{version}.tar.gz"
    root = f"cvxgenrust-{version}"
    metadata = metadata_version or version
    project = project_version or version
    pyproject = (
        "[project]\n"
        'name = "cvxgenrust"\n'
        f'version = "{project}"\n'
        'requires-python = ">=3.12"\n'
        'license = "Apache-2.0"\n'
    )
    with tarfile.open(path, "w:gz") as archive:
        _add_tar_bytes(archive, f"{root}/.gitignore", b"dist/\n")
        _add_tar_bytes(archive, f"{root}/LICENSE", b"Apache License\n")
        _add_tar_bytes(
            archive,
            f"{root}/PKG-INFO",
            (
                "Metadata-Version: 2.4\n"
                "Name: cvxgenrust\n"
                f"Version: {metadata}\n"
                "Requires-Python: >=3.12\n"
                "License-Expression: Apache-2.0\n\n"
            ).encode(),
        )
        _add_tar_bytes(archive, f"{root}/README.md", b"# cvxgenrust\n")
        _add_tar_bytes(archive, f"{root}/pyproject.toml", pyproject.encode())
        for package_file in _package_files(source):
            relative = package_file.relative_to(source).as_posix()
            _add_tar_bytes(archive, f"{root}/cvxgenrust/{relative}", package_file.read_bytes())
        if extra is not None:
            _add_tar_bytes(archive, f"{root}/{extra}", b"unexpected\n")
    return path


@pytest.mark.parametrize("version", ["0.0.0", "0.1.0", "1.2.3", "10.20.30"])
def test_canonical_stable_release_versions(version: str) -> None:
    assert str(canonical_stable_version(version)) == version


@pytest.mark.parametrize(
    "version",
    [
        "v0.1.0",
        "0.1",
        "01.2.3",
        "1!0.2.0",
        "0.2.0rc1",
        "0.2.0.post1",
        "0.2.0.dev1",
        "0.2.0+local",
    ],
)
def test_noncanonical_or_unstable_release_versions_are_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="canonical stable x.y.z"):
        canonical_stable_version(version)


def test_generator_version_comes_from_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_distributions: list[str] = []

    def distribution_version(distribution_name: str) -> str:
        requested_distributions.append(distribution_name)
        return "9.8.7"

    monkeypatch.setattr(config.importlib.metadata, "version", distribution_version)

    assert config._load_generator_version() == "9.8.7"
    assert requested_distributions == ["cvxgenrust"]


def test_generator_version_falls_back_when_distribution_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(config.importlib.metadata, "version", missing_distribution)

    assert config._load_generator_version() == "0.0.0.dev0"


def test_repository_release_metadata_is_consistent() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    project_version = canonical_stable_version(project["version"])
    installed_version = importlib.metadata.version(project["name"])

    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))
    root_packages = [
        package
        for package in lock["package"]
        if package["name"] == project["name"] and package.get("source") == {"editable": "."}
    ]
    assert len(root_packages) == 1, "uv.lock must contain exactly one editable root package."
    assert root_packages[0]["version"] == str(project_version), (
        "Run `uv lock` after changing the project version."
    )
    assert installed_version == str(project_version)
    assert config.GENERATOR_VERSION == installed_version


def test_release_distributions_and_checksums(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = _wheel(dist, source)
    sdist = _sdist(dist, source)

    assert verify_distributions(dist, "0.1.0", source) == (wheel, sdist)

    checksums = dist / "SHA256SUMS"
    write_checksums((wheel, sdist), checksums)
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (wheel, sdist)}
    actual = {
        name: digest
        for digest, name in (
            line.split("  ", 1) for line in checksums.read_text(encoding="utf-8").splitlines()
        )
    }
    assert actual == expected


def test_release_verifier_rejects_inconsistent_wheel_metadata(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, source, metadata_version="0.2.0")
    _sdist(dist, source)

    with pytest.raises(ValueError, match="wrong project version"):
        verify_distributions(dist, "0.1.0", source)


def test_release_verifier_rejects_inconsistent_sdist_metadata(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, source)
    _sdist(dist, source, project_version="0.2.0")

    with pytest.raises(ValueError, match="inconsistent project metadata"):
        verify_distributions(dist, "0.1.0", source)


def test_release_verifier_rejects_nonuniversal_wheel(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, source, tag="cp312-cp312-manylinux_2_17_x86_64")
    _sdist(dist, source)

    with pytest.raises(ValueError, match="py3-none-any"):
        verify_distributions(dist, "0.1.0", source)


def test_release_verifier_requires_templates_in_wheel(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, source, include_templates=False)
    _sdist(dist, source)

    with pytest.raises(ValueError, match="missing package sources.*runtime.rs.tmpl"):
        verify_distributions(dist, "0.1.0", source)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("tests/test_release.py", "development-only path 'tests'"),
        ("cvxgenrust/generated/solver.rs", "Generated file"),
        ("cvxgenrust/__pycache__/cgr.pyc", "Generated file"),
        ("../escape.py", "Unsafe path"),
    ],
)
def test_release_verifier_rejects_development_generated_or_unsafe_sdist_files(
    tmp_path: Path,
    extra: str,
    message: str,
) -> None:
    source = _source_tree(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist, source)
    _sdist(dist, source, extra=extra)

    with pytest.raises(ValueError, match=message):
        verify_distributions(dist, "0.1.0", source)


def test_hatch_sdist_manifest_is_minimal() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["hatch"]["build"]["targets"]["sdist"] == {
        "only-include": ["cvxgenrust", "README.md", "LICENSE", "pyproject.toml"]
    }


def test_external_workflow_action_pins_are_immutable() -> None:
    workflows = sorted([*WORKFLOWS_DIRECTORY.glob("*.yml"), *WORKFLOWS_DIRECTORY.glob("*.yaml")])
    assert workflows
    external_actions = [
        action
        for workflow in workflows
        for action in ACTION_PATTERN.findall(workflow.read_text(encoding="utf-8"))
        if not action.startswith("./")
    ]
    assert external_actions
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action in external_actions)
    assert set(external_actions) == REVIEWED_ACTION_PINS
