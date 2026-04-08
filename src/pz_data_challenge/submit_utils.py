import os
import tarfile
import tempfile
from typing import Any
import urllib.request
from pathlib import Path

import qp
import tables_io


def download_and_extract_tar(url: str, extract_to: str | Path = ".") -> None:
    """
    Download a tar file from a URL and extract its contents.

    Parameters
    ----------
    url : str
        URL of the tar file to download. Supports .tar, .tar.gz, .tgz,
        .tar.bz2, and .tar.xz formats.
    extract_to : str or Path, optional
        Directory path where the contents will be extracted.
        Default is the current directory ('.').

    Returns
    -------
    None

    Raises
    ------
    urllib.error.URLError
        If the download fails due to network issues or invalid URL.
    tarfile.TarError
        If the file is not a valid tar archive or extraction fails.
    PermissionError
        If there are insufficient permissions to write to the extraction
        directory or create temporary files.

    Notes
    -----
    The function automatically detects the compression format of the tar
    file. The downloaded tar file is stored in a temporary location and
    automatically deleted after extraction.
    """
    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp_file:
        tmp_path: str = tmp_file.name
        urllib.request.urlretrieve(url, tmp_path)

    try:
        # Extract with automatic format detection
        with tarfile.open(tmp_path, "r:*") as tar:
            tar.extractall(path=extract_to, filter="data")
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def check_pz_submission_file(
    submit_file: str | Path,
    test_file: str | Path,
) -> list[int]:
    """
    Validate a photo-z submission file against test data requirements.

    This function checks that a submission file exists, is in valid qp format,
    contains required ancillary data (zmode and object_id), and that the
    object IDs match those in the test file.

    Parameters
    ----------
    submit_file : str or Path
        Path to the submission file to validate. Must be in qp-readable format.
    test_file : str or Path
        Path to the test file containing reference object IDs. Must be readable
        by tables_io.

    Returns
    -------
    List with flags to mark if submission file exists and is well formatted

    Raises
    ------
    FileNotFoundError
        If test_file does not exist.

    Notes
    -----
    The function performs the following checks in order, and assigns
    corresponding flags to the output list for each
    1. File existence
    2. Valid qp ensemble format
    3. Presence of ancillary dictionary
    4. Presence of 'zmode' in ancillary data
    5. Presence of 'object_id' in ancillary data
    6. Matching object IDs between submission and test files

    """
    # build the output list
    out_list: list[int] = []

    # Convert to Path objects for easier handling
    submit_path: Path = Path(submit_file)
    test_path: Path = Path(test_file)

    # Check that test_file exists
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Check that submit_file exists
    if not submit_path.exists():
        return out_list

    out_list += [1]

    # Open and validate qp format
    try:
        ensemble = qp.read(submit_file)
    except Exception:
        return out_list

    out_list += [2]

    # Check that ancillary dict exists
    try:
        ancil = ensemble.ancil
    except AttributeError:
        return out_list

    out_list += [3]

    if ancil is None:
        return out_list

    out_list += [4]

    # Check for zmode entry
    if "zmode" in ancil:
        out_list += [5]

    # Check for object_id entry
    if "object_id" in ancil:
        out_list += [6]

    # Get object IDs from submission file
    submit_ids = set(ancil["object_id"])

    # Get object IDs from test file
    try:
        test_data = tables_io.read(test_file)
        test_ids = set(test_data["object_id"])
    except Exception as e:
        raise Exception(f"Failed to read test file: {e}")

    # Check that object IDs match
    if submit_ids == test_ids:
        out_list += [7]

    return out_list


def pretty_print_manifest_dict(manifest_dict: dict[str, Any]) -> None:
    print("Key                            Status")
    print("-------------------------------------------")

    for key, checks in manifest_dict.items():
        if key.find("time") >= 0:
            continue
        outst = ""
        for i in range(1, 8):
            if i in checks:
                outst += "+ "
            else:
                outst += "- "

        print(f"{key:<30} {outst}")
        print("")


def pretty_print_time_dict(manifest_dict: dict[str, Any]) -> None:

    print("Key                            Time")
    print("-------------------------------------------")
    for key, time_ in manifest_dict.items():
        if key.find("time") < 0:
            continue
        print(f"{key:<30} {time_:.2f}")
        print("")
