import os
import pytest

from pz_data_challenge import submit_utils

# don't change these
PUBLIC_URL: str = "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz"


@pytest.fixture(name="setup_public_area", scope="package")
def setup_public_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the public data
    """

    if not os.path.exists("tests/public"):
        # Note that the tar file has "public" as top level directory
        # so we if we extract to "tests" the files actually end
        # up in "tests/public"
        submit_utils.download_and_extract_tar(PUBLIC_URL, "tests")

    def teardown_public_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            os.system("\\rm tests/public")

    request.addfinalizer(teardown_public_area)

    return 0
