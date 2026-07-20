import glob
import os
import sys
import subprocess
import yaml

from pz_data_challenge import admin_utils, evaluation

PZ_DATA_PATH = os.environ["PZ_DATA_PATH"]
PZ_RESERVED_DATA_PATH = os.path.join(PZ_DATA_PATH, "reserved")

SUBMISSION_TOP_DIR = "submissions"
RESULTS_TOP_DIR = "results"
ACCEPTED_SUBMISSIONS_DIR = "accepted"


if __name__ == "__main__":

    # sys.argv[0] is the script name, sys.argv[1:] are the arguments
    all_submissions = evaluation.get_submissions(ACCEPTED_SUBMISSIONS_DIR)

    if len(sys.argv) == 1:
        submissions = []
    elif sys.argv[1] == "all":
        submissions = all_submissions
    else:
        submissions = sys.argv[1:]

    for submission_name in submissions:

        submission_dir = os.path.join(SUBMISSION_TOP_DIR, submission_name)
        results_dir = os.path.join(RESULTS_TOP_DIR, submission_name)

        try:
            admin_utils.evaluate_submission(
                ACCEPTED_SUBMISSIONS_DIR,
                submission_name,
                submission_dir,
                results_dir,
                RESULTS_TOP_DIR,
                PZ_RESERVED_DATA_PATH,
            )
        except Exception as exc:
            print(f"Failed to evaluate {submission_name} because {exc}")
            raise

    if not os.environ.get('SKIP_SUMMARY'):
        admin_utils.make_all_summary_plots_and_files(RESULTS_TOP_DIR, all_submissions)
