import glob
import os
import sys
import subprocess
import yaml

from pz_data_challenge import admin_utils

PZ_DATA_PATH = os.environ['PZ_DATA_PATH']
PZ_RESERVED_DATA_PATH = os.path.join(PZ_DATA_PATH, 'reserved')

SUBMISSION_TOP_DIR = 'submissions'
RESULTS_TOP_DIR = 'results'


if __name__ == '__main__':

    # sys.argv[0] is the script name, sys.argv[1:] are the arguments
    if len(sys.argv) == 1:
        script_name = sys.argv[0]
        num_args = len(sys.argv) - 1
        
        print(f"Error: Expected > 1 argument", file=sys.stderr)
        print(f"Usage: python {script_name} <arguments>", file=sys.stderr)        
        sys.exit(1)


    for arg_ in sys.argv[1:]:
        
        # Get and use the argument
        submission_name = arg_
        submission_dir = os.path.join(SUBMISSION_TOP_DIR, submission_name)
        results_dir = os.path.join(RESULTS_TOP_DIR, submission_name)
        
        admin_utils.evaluate_submission(
            submission_name,
            submission_dir,
            results_dir,
            RESULTS_TOP_DIR,
            PZ_RESERVED_DATA_PATH,
        )
        
    
    
