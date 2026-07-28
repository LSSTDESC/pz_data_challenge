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
    #
    submit_utils._DOWNLOAD_TIMEOUT = 120
    submit_utils._DOWNLOAD_RETRY_DELAY = 30
    if not os.path.exists("tests/public"):
        # Note that the tar file has "public" as top level directory
        # so if we extract to "tests" the files actually end up in "tests/public"
        try:
            submit_utils.download_and_extract_tar(PUBLIC_URL, "tests")
        except Exception as e:
            # Failsafe fallback method: try reading/extracting corresponding dataset files
            # from local release archives or fallback tarballs if NERSC is unreachable.
            import tarfile
            import shutil
            
            fallback_extracted = False
            tar_candidates = [
                "public.tgz",
                "tests/public.tgz",
                "submissions/graysmoke/public.tgz",
                "submissions/graysmoke_submission.tgz",
                "submissions/rail_aion_submission.tgz",
            ]
            for tar_path in tar_candidates:
                if os.path.exists(tar_path):
                    try:
                        with tarfile.open(tar_path, "r:*") as tar:
                            tar.extractall(path="tests", filter="data")
                        fallback_extracted = True
                        break
                    except Exception:
                        continue

            if not fallback_extracted:
                for sub_dir in ["submissions/graysmoke/public", "submissions/rail_aion/public"]:
                    if os.path.exists(sub_dir):
                        shutil.copytree(sub_dir, "tests/public", dirs_exist_ok=True)
                        fallback_extracted = True
                        break

            if not fallback_extracted:
                raise e

    def teardown_public_area() -> None:
        pass

    request.addfinalizer(teardown_public_area)

    return 0
