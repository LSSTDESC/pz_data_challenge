import os
from pz_data_challenge import submit_utils

# don't change these
PUBLIC_URL: str = "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz"


def setup_public_area() -> None:
    """
    A function download the public data
    """

    if not os.path.exists("public"):
        # Note that the tar file has "public" as top level directory
        # so we if we extract to "tests" the files actually end
        # up in "tests/public"
        submit_utils.download_and_extract_tar(PUBLIC_URL, ".")


if __name__ == '__main__':

    setup_public_area()
