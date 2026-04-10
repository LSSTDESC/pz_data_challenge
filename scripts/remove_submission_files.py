import os
import sys

def remove_file(path) -> None:
    """
    Remove a file.
    
    Parameters:
    -----------
    path : str
        Path to the file
    Returns:
    --------
    None
    """
    os.unlink(path)
    print(f"Remove {path}\n")


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

    print(f"\n Removing files from pz_data_challenge for a submission called {submission_name}\n")

    remove_file(f".github/workflows/submit_{submission_name}.yaml")

    remove_file(f"requirements_{submission_name}.txt")

    remove_file(f"tests/test_{submission_name}.py")


    
    
    
