"""Validate cvxgenrust release distributions and write SHA-256 checksums."""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
import tarfile
import tomllib
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

_PROJECT_NAME = "cvxgenrust"
_REQUIRES_PYTHON = ">=3.12"
_LICENSE_EXPRESSION = "Apache-2.0"
_RELEASE_VERSION_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
)
_PROHIBITED_SDIST_PATHS = {
    ".github",
    "Makefile",
    "docs",
    "examples",
    "scripts",
    "tests",
    "uv.lock",
}
_GENERATED_PARTS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "generated",
    "target",
}


def canonical_stable_version(value: str) -> Version:
    """Parse a canonical stable ``major.minor.patch`` release or raise ``ValueError``."""
    if _RELEASE_VERSION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Release version must be a canonical stable x.y.z version: {value!r}.")
    version = Version(value)
    if str(version) != value or version.epoch != 0 or len(version.release) != 3:
        raise ValueError(f"Release version must be a canonical stable x.y.z version: {value!r}.")
    return version


def _distribution_files(dist_dir: Path) -> tuple[Path, Path]:
    distributions = sorted(
        path
        for path in dist_dir.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    wheels = [path for path in distributions if path.suffix == ".whl"]
    sdists = [path for path in distributions if path.name.endswith(".tar.gz")]
    if len(distributions) != 2 or len(wheels) != 1 or len(sdists) != 1:
        names = ", ".join(path.name for path in distributions) or "none"
        raise ValueError(f"Expected exactly one wheel and one sdist; found: {names}.")
    return wheels[0], sdists[0]


def _validate_archive_path(name: str, *, archive: Path) -> PurePosixPath:
    if not name or "\\" in name:
        raise ValueError(f"Unsafe path in {archive.name}: {name!r}.")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in {archive.name}: {name!r}.")
    if any(part in _GENERATED_PARTS or part.endswith((".pyc", ".pyo")) for part in path.parts):
        raise ValueError(f"Generated file in {archive.name}: {name!r}.")
    return path


def _metadata_value(raw: bytes, field: str, *, archive: Path) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    value = message.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{archive.name} metadata is missing {field!r}.")
    return value


def _validate_metadata(raw: bytes, expected_version: Version, *, archive: Path) -> None:
    if canonicalize_name(_metadata_value(raw, "Name", archive=archive)) != _PROJECT_NAME:
        raise ValueError(f"{archive.name} contains the wrong project name.")
    if _metadata_value(raw, "Version", archive=archive) != str(expected_version):
        raise ValueError(f"{archive.name} contains the wrong project version.")
    if _metadata_value(raw, "Requires-Python", archive=archive) != _REQUIRES_PYTHON:
        raise ValueError(f"{archive.name} contains an unexpected Python requirement.")
    if _metadata_value(raw, "License-Expression", archive=archive) != _LICENSE_EXPRESSION:
        raise ValueError(f"{archive.name} contains an unexpected license expression.")


def _required_package_sources(source_root: Path, *, prefix: str) -> set[str]:
    python_sources = {
        f"{prefix}/{path.relative_to(source_root).as_posix()}"
        for path in source_root.rglob("*.py")
        if path.is_file()
    }
    templates_root = source_root / "templates"
    template_sources = {
        f"{prefix}/{path.relative_to(source_root).as_posix()}"
        for path in templates_root.glob("*.tmpl")
        if path.is_file()
    }
    if not python_sources:
        raise ValueError(f"Package source directory contains no Python files: {source_root}.")
    if not template_sources:
        raise ValueError(f"Package source directory contains no templates: {templates_root}.")
    return python_sources | template_sources


def _validate_wheel(wheel: Path, expected_version: Version, source_root: Path) -> None:
    name, version, build, tags = parse_wheel_filename(wheel.name)
    if canonicalize_name(name) != _PROJECT_NAME or version != expected_version or build:
        raise ValueError(f"Unexpected wheel filename: {wheel.name!r}.")
    if {str(tag) for tag in tags} != {"py3-none-any"}:
        raise ValueError(
            f"cvxgenrust must publish one pure-Python py3-none-any wheel, not {wheel.name!r}."
        )

    expected_dist_info = f"{_PROJECT_NAME}-{expected_version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        paths = [_validate_archive_path(entry.filename, archive=wheel) for entry in entries]
        if any(stat.S_ISLNK(entry.external_attr >> 16) for entry in entries):
            raise ValueError(f"{wheel.name} contains unsupported symbolic links.")
        for prohibited in _PROHIBITED_SDIST_PATHS:
            if any(
                path.as_posix() == prohibited
                or path.as_posix().startswith(f"{prohibited}/")
                for path in paths
            ):
                raise ValueError(f"{wheel.name} contains development-only path {prohibited!r}.")

        archived = {path.as_posix() for path in paths}
        metadata_name = f"{expected_dist_info}/METADATA"
        wheel_name = f"{expected_dist_info}/WHEEL"
        if metadata_name not in archived or wheel_name not in archived:
            raise ValueError(f"{wheel.name} must contain its expected METADATA and WHEEL files.")

        _validate_metadata(archive.read(metadata_name), expected_version, archive=wheel)
        wheel_metadata = archive.read(wheel_name)
        if _metadata_value(wheel_metadata, "Root-Is-Purelib", archive=wheel).lower() != "true":
            raise ValueError(f"{wheel.name} is not marked as a pure-Python wheel.")
        if _metadata_value(wheel_metadata, "Tag", archive=wheel) != "py3-none-any":
            raise ValueError(f"{wheel.name} contains an unexpected compatibility tag.")

        if f"{expected_dist_info}/licenses/LICENSE" not in archived:
            raise ValueError(f"{wheel.name} is missing its license file.")
        required_sources = _required_package_sources(source_root, prefix=_PROJECT_NAME)
        missing = sorted(required_sources - archived)
        if missing:
            raise ValueError(f"{wheel.name} is missing package sources: {', '.join(missing)}.")


def _validate_sdist(sdist: Path, expected_version: Version, source_root: Path) -> None:
    name, version = parse_sdist_filename(sdist.name)
    if canonicalize_name(name) != _PROJECT_NAME or version != expected_version:
        raise ValueError(f"Unexpected sdist filename: {sdist.name!r}.")

    expected_root = f"{_PROJECT_NAME}-{expected_version}"
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError(f"{sdist.name} is empty.")
        paths = [_validate_archive_path(member.name, archive=sdist) for member in members]
        if any(
            member.issym() or member.islnk() or not (member.isfile() or member.isdir())
            for member in members
        ):
            raise ValueError(f"{sdist.name} contains unsupported filesystem entries.")
        if any(not path.parts or path.parts[0] != expected_root for path in paths):
            raise ValueError(f"{sdist.name} must contain only the root directory {expected_root!r}.")

        relative_paths = {
            PurePosixPath(*path.parts[1:]).as_posix()
            for path in paths
            if len(path.parts) > 1
        }
        relative_files = {
            PurePosixPath(*path.parts[1:]).as_posix()
            for path, member in zip(paths, members, strict=True)
            if len(path.parts) > 1 and member.isfile()
        }
        for prohibited in _PROHIBITED_SDIST_PATHS:
            if any(path == prohibited or path.startswith(f"{prohibited}/") for path in relative_paths):
                raise ValueError(f"{sdist.name} contains development-only path {prohibited!r}.")

        required_root_files = {"LICENSE", "PKG-INFO", "README.md", "pyproject.toml"}
        required_sources = _required_package_sources(source_root, prefix=_PROJECT_NAME)
        # Hatch always includes the repository's VCS ignore file in an sdist.
        optional_build_metadata = {".gitignore"}
        expected_files = required_root_files | required_sources
        missing = sorted(expected_files - relative_files)
        if missing:
            raise ValueError(f"{sdist.name} is missing required files: {', '.join(missing)}.")
        allowed_files = expected_files | optional_build_metadata
        allowed_directories = {
            parent.as_posix()
            for file_name in allowed_files
            for parent in PurePosixPath(file_name).parents
            if parent.parts
        }
        unexpected = sorted(relative_paths - allowed_files - allowed_directories)
        if unexpected:
            raise ValueError(f"{sdist.name} contains unexpected paths: {', '.join(unexpected)}.")

        pyproject_file = archive.extractfile(f"{expected_root}/pyproject.toml")
        if pyproject_file is None:
            raise ValueError(f"Could not read pyproject.toml from {sdist.name}.")
        try:
            project = tomllib.loads(pyproject_file.read().decode("utf-8"))["project"]
            project_name = canonicalize_name(project["name"])
            project_version = project["version"]
            requires_python = project["requires-python"]
            license_expression = project["license"]
        except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"{sdist.name} contains invalid project metadata.") from exc
        if (
            project_name != _PROJECT_NAME
            or project_version != str(expected_version)
            or requires_python != _REQUIRES_PYTHON
            or license_expression != _LICENSE_EXPRESSION
        ):
            raise ValueError(f"{sdist.name} contains inconsistent project metadata.")

        package_info = archive.extractfile(f"{expected_root}/PKG-INFO")
        if package_info is None:
            raise ValueError(f"Could not read PKG-INFO from {sdist.name}.")
        _validate_metadata(package_info.read(), expected_version, archive=sdist)


def verify_distributions(dist_dir: Path, version: str, source_root: Path) -> tuple[Path, Path]:
    """Validate and return the wheel and sdist for one cvxgenrust release."""
    expected_version = canonical_stable_version(version)
    directory = dist_dir.resolve()
    sources = source_root.resolve()
    if not directory.is_dir():
        raise ValueError(f"Distribution directory does not exist: {directory}.")
    if not sources.is_dir():
        raise ValueError(f"Package source directory does not exist: {sources}.")

    wheel, sdist = _distribution_files(directory)
    _validate_wheel(wheel, expected_version, sources)
    _validate_sdist(sdist, expected_version, sources)
    return wheel, sdist


def write_checksums(distributions: tuple[Path, Path], output: Path) -> None:
    """Write stable SHA-256 checksum lines for release distributions."""
    lines = []
    for path in sorted(distributions, key=lambda item: item.name):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{digest}  {path.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="directory containing release distributions",
    )
    parser.add_argument("--version", required=True, help="canonical stable x.y.z package version")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("cvxgenrust"),
        help="cvxgenrust package source directory",
    )
    parser.add_argument(
        "--checksums",
        type=Path,
        default=Path("dist/SHA256SUMS"),
        help="checksum output file",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        distributions = verify_distributions(args.dist_dir, args.version, args.source_root)
        write_checksums(distributions, args.checksums)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
