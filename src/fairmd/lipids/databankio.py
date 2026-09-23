"""
Input/Output auxiliary functions.

Input/Output module with some small usefull functions. It includes:
- Downloading files.
- Resolving URLs.
- Calculating file hash for fingerprinting.
"""

import hashlib
import logging
import math
import os
import tempfile
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import libarchive
import requests
import requests.exceptions as rexp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fairmd.lipids import __version__, progress

__all__ = [
    "MAX_BYTES_DEFAULT",
    "_fmdl_chunk_size",
    "_get_file_size_with_retry",
    "_open_url_with_retry",
    "calc_file_sha1_hash",
    "create_simulation_directories",
    "download_resource_from_uri",
    "download_system_file",
    "download_with_progress_with_retry",
    "prepare_file_sources",
    "resolve_file_url",
]

_fmdl_chunk_size = 8192  # 8 KB, chunk size for downloading files in pieces
logger = logging.getLogger(__name__)
MAX_BYTES_DEFAULT = 50 * 1024 * 1024  # 50 MB, default max size for download_resource_from_uri(..., max_bytes=True)

# --- Helper Functions for Network Requests ---


def _requests_session_with_retry(
    retries: int = 5,
    backoff: float = 10,
) -> requests.Session:
    """Session gererator for using in with-constructs.

    :param retries: Max num of retries, defaults to 5
    :param backoff: Starting time before first retry, defaults to 10
    :return: requests session
    """
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@contextmanager
def _open_url_with_retry(
    uri: str,
    backoff: float = 10,
    *,
    stream: bool = True,
    update_headers: dict | None = None,
) -> Generator[requests.Response, None, None]:
    """Open a URL with a timeout and retry logic (aprivate helper).

    :param uri: The URL to open.
    :param backoff: The backoff timeout for the request in seconds.
    :param stream: Whether to stream the response content.
    :param update_headers: Additional headers to include in the request.

    :return: The response object.
    """
    headers = {"User-Agent": f"fairmd-lipids/{__version__}"}
    if update_headers is not None:
        headers.update(update_headers)
    with _requests_session_with_retry(retries=5, backoff=backoff) as session:
        response = session.get(uri, stream=stream, headers=headers)
        response.raise_for_status()
        try:
            yield response
        finally:
            response.close()


def _get_file_size_with_retry(uri: str) -> int:
    """Fetch the size of a file from a URI with retry logic.

    :param uri: (str) The URL of the file.

    :returns: The size of the file in bytes, or 0 if the 'Content-Length'
              header is not present (int).
    """
    with _open_url_with_retry(uri) as response:
        content_length = response.headers.get("Content-Length")
        return int(content_length) if content_length else 0


def download_with_progress_with_retry(
    uri: str,
    dest: str,
    *,
    tqdm_title: str = "Downloading",
    stop_after: int | None = None,
    total_size: int | None = None,
) -> None:
    """Download a file with a progress bar and retry logic.

    Uses tqdm to display a progress bar during the download.

    Args:
        uri (str): The URL of the file to download.
        dest (str): The local destination path to save the file.
        tqdm_title (str): The title used for the progress bar description.
        stop_after (int): Download max num of bytes
        total_size (int): Total size of the file to download (for resuming).
    """

    class RetrieveProgressBar:
        def __init__(self, **kwargs):
            self._pbar = progress(**kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._pbar.close()

        def update_retrieve(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self._pbar.total = tsize
            self._pbar.update(b * bsize - self._pbar.n)

    with RetrieveProgressBar(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=tqdm_title,
    ) as u:
        # go
        if os.path.isfile(dest):
            # Resuming download
            dl_size = os.path.getsize(dest)

            n_chunks = dl_size // _fmdl_chunk_size
            downloaded = _fmdl_chunk_size * n_chunks
            if dl_size % _fmdl_chunk_size != 0:
                print(f"Applying truncation to nearest chunk size [{downloaded}].")
                with open(dest, "a") as f:
                    f.truncate(downloaded)
            u.update_retrieve(b=downloaded, bsize=1, tsize=total_size)
            mode = "ab"
            headers = {"Range": f"bytes={downloaded}-"}  # request only missing part
        else:
            mode = "wb"
            headers = {}
            downloaded = 0
        # open connection
        with open(dest, mode) as f, _open_url_with_retry(uri, update_headers=headers) as resp:
            if total_size is not None:
                total = total_size
            else:
                content_length = int(resp.headers.get("Content-Length", 0))
                total = content_length + downloaded if mode == "ab" else content_length
            if mode == "ab" and resp.status_code != requests.status_codes.codes.PARTIAL_CONTENT:
                msg = (
                    "Server doesn't return PARTIAL CONTENT 206 status.",
                    f"Cannot resume {dest}. Please delete it and restart.",
                )
                raise requests.exceptions.HTTPError(msg)
            if mode == "wb" and resp.status_code != requests.status_codes.codes.OK:
                msg = f"Failed to download {dest}. Server returned status code {resp.status_code}."
                raise requests.exceptions.HTTPError(msg)
            if stop_after is not None:
                total = min(total, stop_after)

            for chunk in resp.iter_content(chunk_size=_fmdl_chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                u.update_retrieve(b=downloaded, bsize=1, tsize=total)
                if downloaded >= total:
                    break

        if downloaded > total:
            with open(dest, "rb+") as f:
                f.truncate(total)


ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")

# --- Archive support ---
# A repository file name may address one member inside a ZIP, TAR, TAR.GZ, TGZ, or 7z archive,
# e.g. "simulation.zip/folder/run.xtc": only "simulation.zip" is a real file
# on the repository, the rest is a path inside it once downloaded.


def _validate_source_path(source: str) -> PurePosixPath:
    """Reject archive member paths that escape the archive, e.g. "../x"."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("A nonempty repository filename is required")
    path = PurePosixPath(source)
    if path.is_absolute() or ".." in path.parts or "\\" in source or path.name in ("", "."):
        msg = f"Invalid archive member path: {source!r}"
        raise ValueError(msg)
    return path


def find_archive_in_path(file_path):
    """Split at the first archive with a member following it."""
    parts = Path(file_path).parts
    for i, part in enumerate(parts[:-1]):
        if part.lower().endswith(ARCHIVE_EXTENSIONS):
            return Path(*parts[: i + 1]), str(Path(*parts[i + 1 :]))
    return None, None


def _write_atomic(destination, blocks):
    """Keep incomplete downloads/extractions out of the cache."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            for block in blocks:
                output.write(block)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_file_from_archive(archive_path, target_path, dest_dir):
    """Extract one regular member to dest_dir using its basename."""
    target = _validate_source_path(target_path)
    with libarchive.file_reader(str(archive_path)) as entries:
        for entry in entries:
            if PurePosixPath(entry.pathname) == target:
                if not entry.isfile or entry.islnk:
                    raise ValueError(f"Archive member is not a regular file: {target}")
                destination = Path(dest_dir) / target.name
                _write_atomic(destination, entry.get_blocks())
                return destination
    raise FileNotFoundError(f"'{target}' not found in archive {archive_path}")


def extract_nested_file_from_archives(archive_path, nested_path, dest_path):
    """Extract arbitrary archive nesting, resetting member paths at each layer."""
    remaining = str(_validate_source_path(nested_path))
    current = Path(archive_path)
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest_path) as temporary:
        depth = 0
        while True:
            inner, member = find_archive_in_path(remaining)
            if inner is None:
                return extract_file_from_archive(current, remaining, dest_path)
            current = extract_file_from_archive(current, str(inner), Path(temporary) / str(depth))
            remaining = member
            depth += 1


def _extract_zip_member(archive_path: str, member: str, dest: str) -> None:
    """Compatibility wrapper for extracting one archive member to an exact path."""
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        extracted = extract_nested_file_from_archives(archive_path, member, Path(temporary))
        os.replace(extracted, destination)


def prepare_file_sources(sim: Mapping, file_keys) -> dict[str, str]:
    """Normalize file fields in place to local basenames.

    :param sim: mutable simulation mapping (a System instance or dict).
    :param file_keys: names of the fields that hold file lists (each entry
        being e.g. `[filename]` or `[filename, hash]`).
    :return: {local_basename: original_repository_path}, so the caller can
        resolve/download each file from where it actually comes from.
    """
    sources: dict[str, str] = {}
    replacements = []
    for key in file_keys:
        for entry in sim.get(key) or []:
            original = entry[0]
            source = str(_validate_source_path(original))
            local = _validate_source_path(original).name
            if local in sources and sources[local] != source:
                msg = f"Files {sources[local]!r} and {source!r} share local filename {local!r}"
                raise ValueError(msg)
            sources[local] = source
            replacements.append((entry, local))
    for entry, local in replacements:
        entry[0] = local
    archived = {local: source for local, source in sources.items() if source != local}
    if archived:
        sim["SOURCE_FILES"] = archived
    return sources


def download_system_file(
    sim: Mapping,
    local_name: str,
    dest: str,
    *,
    override_if_exists: bool = False,
    max_restarts: int = 0,
) -> int:
    """Download a simulation file using SOURCE_FILES to resolve archive paths.

    Falls back to local_name for standalone files or older metadata.
    Use for re-downloads after the original working directory is gone.
    """
    source = (sim.get("SOURCE_FILES") or {}).get(local_name, local_name)
    uri = resolve_file_url(sim["DOI"], source)
    return download_resource_from_uri(
        uri,
        dest,
        override_if_exists=override_if_exists,
        max_restarts=max_restarts,
        source_path=source if source != local_name else None,
    )


# --- Main Functions ---


def download_resource_from_uri(
    uri: str,
    dest: str,
    *,
    override_if_exists: bool = False,
    max_bytes: bool = False,
    max_restarts: int = 0,
    source_path: str | None = None,
) -> int:
    """Download file resource from a URI to a local destination.

    Checks if the file already exists and has the same size before downloading.
    Can also perform a partial "dry-run" download.

    Args:
        uri (str): The URL of the file resource.
        dest (str): The local destination path to save the file.
        override_if_exists (bool): If True, the file will be re-downloaded
            even if it already exists. Defaults to False.
        max_bytes (bool): If True, only a partial download is performed
            (up to MAX_DRYRUN_SIZE). Defaults to False.
        source_path (str | None): Set this when the file lives inside a
            archive on the repository, e.g. "sim.zip/folder/run.xtc"
            (`uri` must then point at the archive's own record/name, as
            resolve_file_url() does for such paths). The archive is
            cached under `.archives` next to `dest` and the requested member
            (possibly nested in another archive) is extracted to `dest`.
            Archives are downloaded completely, even with `max_bytes=True`.
            Defaults to None.

    Returns
    -------
        int: A status code indicating the result.
            0: Download was successful.
            1: Download was skipped because the file already exists.
            2: File was re-downloaded due to a size mismatch.

    Raises
    ------
        ConnectionError: An error occurred after multiple download attempts.
        OSError: The downloaded file size does not match the expected size.
    """
    if source_path is not None:
        source = _validate_source_path(source_path)
        archive, member = find_archive_in_path(str(source))
        if archive is not None:
            cache_key = hashlib.sha256(uri.encode()).hexdigest()
            archive_dest = Path(dest).parent / ".archives" / cache_key / archive.name
            archive_dest.parent.mkdir(parents=True, exist_ok=True)
            status = download_resource_from_uri(
                uri,
                str(archive_dest),
                override_if_exists=override_if_exists,
                max_restarts=max_restarts,
            )
            _extract_zip_member(str(archive_dest), member, dest)
            return status

    fi_name = uri.split("/")[-1]
    return_code = 0

    if os.path.isdir(dest):
        msg = f"Destination '{dest}' is a directory, not a file."
        raise IsADirectoryError(msg)

    # Check if dest path already exists and compare file size
    if not override_if_exists and os.path.isfile(dest):
        try:
            fi_size = _get_file_size_with_retry(uri)
            if fi_size == os.path.getsize(dest):
                logger.info(f"{dest}: file already exists, skipping")
                return 1
            logger.warning(
                f"{fi_name} filesize mismatch of local file '{fi_name}', redownloading ...",
            )
            return_code = 2
        except (rexp.HTTPError, rexp.ConnectionError):
            logger.exception(
                f"Failed to verify file size for {fi_name}. Proceeding with redownload.",
            )
            return_code = 2

    url_size = _get_file_size_with_retry(uri)
    if max_bytes:
        # Download file in dry run mode
        download_with_progress_with_retry(uri, dest, tqdm_title=fi_name, stop_after=MAX_BYTES_DEFAULT)
        url_size = min(url_size, MAX_BYTES_DEFAULT)
        return_code = 0
    else:
        # Download with progress bar and check for final size match
        dest_part = dest + ".part"
        re = 0
        while True:
            try:
                download_with_progress_with_retry(uri, dest_part, tqdm_title=fi_name, total_size=url_size)
            except (rexp.ReadTimeout, rexp.ChunkedEncodingError) as e:  # noqa: PERF203
                re += 1
                if re <= max_restarts:
                    logger.warning("Download timed out. Attempting restart...")
                    time.sleep(5)
                else:
                    msg = "Maximum download attempts exceeded."
                    raise ConnectionError(msg) from e
            else:
                break
        os.replace(dest_part, dest)

    size = os.path.getsize(dest)
    if url_size != 0 and url_size != size:
        msg = f"Downloaded filesize mismatch ({size}/{url_size} B)"
        raise OSError(msg)

    return return_code


def resolve_file_url(doi: str, fi_name: str, *, validate_uri: bool = True) -> str:
    """
    Resolve a download file URI from a Zenodo DOI and filename.

    `fi_name` may address a file inside a ZIP, TAR, TAR.GZ, TGZ, or 7z archive using a
    repository-relative path, e.g. "simulation.zip/folder/run.xtc"

    :param doi (str): The DOI identifier for the repository (e.g., "10.5281/zenodo.1234").
    :param fi_name (str): The name of the file within the repository, optionally
                          prefixed by the archive that contains it.
    :param validate_uri (bool): If True, checks if the resolved URL is a valid and
                                reachable address. Defaults to True.

    :return str: The full, direct download URL for the archive/file.

    :raises HTTPError or other connection errors: If the URL cannot be opened after multiple retries.
    :raises NotImplementedError: If the DOI provider is not supported.
    """
    archive_name = fi_name.split("/")[0]

    if "zenodo" in doi.lower():
        zenodo_entry_number = doi.split(".")[2]
        uri = f"https://zenodo.org/records/{zenodo_entry_number}/files/{archive_name}"
        if validate_uri:
            api_uri = f"https://zenodo.org/api/records/{zenodo_entry_number}"
            with _open_url_with_retry(api_uri) as resp:
                jsresp = resp.json()
            if not any(t["key"] == archive_name for t in jsresp["files"]):
                msg = f"File '{archive_name}' not found in zenodo record '{doi}'"
                raise rexp.HTTPError(msg)
        return uri

    msg = "Repository not supported. Please upload the data to zenodo.org"
    raise NotImplementedError(msg)


def calc_file_sha1_hash(fi: str, step: int = 67108864, *, one_block: bool = True) -> str:
    """Calculate the SHA1 hash of a file.

    Reads the file in chunks to handle large files efficiently if specified.

    Args:
        fi (str): The path to the file.
        step (int): The chunk size in bytes for reading the file.
            Defaults to 64MB. Only used if `one_block` is False.
        one_block (bool): If True, reads the first `step` bytes of the file.
            If False, reads the entire file in chunks of `step` bytes.
            Defaults to True.

    Returns
    -------
        str: The hexadecimal SHA1 hash of the file content.
    """
    sha1_hash = hashlib.sha1()  # noqa: S324
    fsize = os.path.getsize(fi)
    if fsize == 0:
        msg = "File should be non-empty for hash fingerprinting!"
        raise ValueError(msg)
    n_tot_steps = math.ceil(fsize / step)
    with open(fi, "rb") as f:
        if one_block:
            block = f.read(step)
            sha1_hash.update(block)
        else:
            with progress(total=n_tot_steps, desc="Calculating SHA1") as pbar:
                for byte_block in iter(lambda: f.read(step), b""):
                    sha1_hash.update(byte_block)
                    pbar.update(1)
    return sha1_hash.hexdigest()


_SOFTWARE_CONFIG = {
    "gromacs": {"primary": "TPR", "secondary": "TRJ"},
    "openMM": {"primary": "TRJ", "secondary": "TRJ"},
    "NAMD": {"primary": "TRJ", "secondary": "TRJ"},
}  # dictionary describing how the hash is formed depending on MD engine (not exported)
# is used right down in :func:`create_simulation_directories`


def create_simulation_directories(
    software: str,
    sim_hashes: Mapping,
    out: str,
    *,
    dry_run_mode: bool = False,
) -> str:
    """Create a nested output directory structure to save simulation results.

    The directory structure is generated based on the hashes of the simulation
    input files.

    Args:
        software: MD engine software (from simulation metadata)
        sim_hashes (Mapping): A dictionary mapping file types (e.g., "TPR",
            "TRJ") to their hash information. The structure is expected to be
            `{'TYPE': [('filename', 'hash')]}`.
        out (str): The root output directory where the nested structure
            will be created.
        dry_run_mode (bool): If True, the directory path is resolved but
            not created. Defaults to False.

    Returns
    -------
        str: The full path to the created output directory.

    Raises
    ------
        FileExistsError: If the target output directory already exists and is
            not empty.
        NotImplementedError: If the simulation software is not supported.
        RuntimeError: If the target output directory could not be created.
    """
    config = _SOFTWARE_CONFIG.get(software)
    if not config:
        msg = f"Sim software '{software}' not supported"
        raise NotImplementedError(msg)

    primary_hash = sim_hashes.get(config["primary"])[0][1]
    secondary_hash = sim_hashes.get(config["secondary"])[0][1]

    head_dir = primary_hash[:3]
    sub_dir1 = primary_hash[3:6]
    sub_dir2 = primary_hash
    sub_dir3 = secondary_hash

    directory_path = os.path.join(out, head_dir, sub_dir1, sub_dir2, sub_dir3)

    logger.debug(f"output_dir = {directory_path}")

    if os.path.exists(directory_path) and os.listdir(directory_path):
        msg = f"Output directory '{directory_path}' is not empty. Delete it if you wish."
        raise FileExistsError(msg)

    if not dry_run_mode:
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception as e:
            msg = f"Could not create the output directory at {directory_path}"
            raise RuntimeError(msg) from e

    return directory_path
