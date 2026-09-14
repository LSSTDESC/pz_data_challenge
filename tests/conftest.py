import os

# Force JAX XLA to use standard OS platform allocator (malloc) and disable 23GB BFC preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"

import pytest
import numpy as np

from pz_data_challenge import submit_utils

# Primary public URL points to GitHub v2.0.0 release asset; NERSC portal acts as secondary fallback.
PUBLIC_URL: str = os.environ.get(
    "PZDC_PUBLIC_URL",
    "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz"
)
FALLBACK_PUBLIC_URL: str = "https://github.com/mardom/pz_data_challenge/releases/download/v2.0.0/public.tgz"


@pytest.fixture(name="setup_public_area", scope="package")
def setup_public_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the public data with multi-tier resilience.
    """
    submit_utils._DOWNLOAD_TIMEOUT = 30
    submit_utils._DOWNLOAD_RETRY_DELAY = 5
    if not os.path.exists("tests/public"):
        download_success = False
        for url in [PUBLIC_URL, FALLBACK_PUBLIC_URL]:
            try:
                print(f"[setup_public_area] Attempting download from {url}...")
                submit_utils.download_and_extract_tar(url, "tests")
                download_success = True
                break
            except Exception as e:
                print(f"[setup_public_area] Could not download from {url}: {e}")
                raise

    def teardown_public_area() -> None:
        pass

    request.addfinalizer(teardown_public_area)

    return 0
