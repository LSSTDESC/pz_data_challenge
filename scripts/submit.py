import os
import sys


if __name__ == '__main__':

    # sys.argv[0] is the script name, sys.argv[1:] are the arguments
    if len(sys.argv) != 2:
        script_name = sys.argv[0]
        num_args = len(sys.argv) - 1
        
        print(f"Error: Expected exactly 1 argument, but got {num_args}", file=sys.stderr)
        print(f"Usage: python {script_name} <argument>", file=sys.stderr)
        
        if num_args > 1:
            print(f"Arguments received: {sys.argv[1:]}", file=sys.stderr)
        
        sys.exit(1)
   
    # Get and print the argument
    submission_name = sys.argv[1]

    print(f"\n This script will help you submit your pz_data_challenge entry called {submission_name}\n")

    print(f"First you verify that your entry is properly formatted by running 'py.test'\n")

    print("Next you should check the status of your local git clone by running 'git status'")
    print(f"  Make sure that you are on the branch submit/{submission_name}")
    print("  and do not have any files added or modified\n")
    
    print("Next you add your files to git by running:")
    print(f"  git add .github/workflows/submit_{submission_name}.yaml requirements_{submission_name}.txt tests/test_{submission_name}.py\n")

    print("Next commit your files to your branch by running:")
    print(f'  git commit -m "Submitting {submission_name}" .github/workflows/submit_{submission_name}.yaml requirements_{submission_name}.txt tests/test_{submission_name}.py\n')

    print("Next push your files to git by running:")
    print(f"  git push --set-upstream origin submit/{submission_name}\n")
    
    print("Pushing to git should give you a URL that you can visit to create a pull request, for example")
    print(f"  https://github.com/LSSTDESC/pz_data_challenge/pull/new/submit/{submission_name}\n")
    print("Visit that URL and create a pull request, then add the 'submission' label to the PR")

    print("Finally, make sure that the github action validating your submission succeeds and fix any issues\n")
 
    
