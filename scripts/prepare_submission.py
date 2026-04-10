import os
import sys
from pz_data_challenge import submit_utils

# don't change these
PUBLIC_URL: str = "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz"


def setup_public_area() -> None:
    """
    A function download the public data
    """
    print(f"copying data from {PUBLIC_URL}\n")
    
    if not os.path.exists("public"):
        # Note that the tar file has "public" as top level directory
        # so we if we extract to "tests" the files actually end
        # up in "tests/public"
        submit_utils.download_and_extract_tar(PUBLIC_URL, ".")


def copy_txt_with_replacement(template_path, output_path, old_string, new_string):
    """
    Copy a text file to an output file and replace occurrences of a string.
    
    Parameters:
    -----------
    template_path : str
        Path to the template file
    output_path : str
        Path where the modified file will be saved
    old_string : str
        The string to be replaced
    new_string : str
        The string to replace with
    
    Returns:
    --------
    None
    """
    try:
        # Read the template file
        with open(template_path, 'r', encoding='utf-8') as template_file:
            content = template_file.read()
        
        # Replace the string
        modified_content = content.replace(old_string, new_string)
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(modified_content)
        
        print(f"Successfully created {output_path} from {template_path}\n")
        
    except FileNotFoundError:
        print(f"Error: Template file '{template_path}' not found", file=sys.stderr)
    except PermissionError:
        print(f"Error: Permission denied when accessing files", file=sys.stderr)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)


def copy_file(template_path, output_path):
    """
    Copy a text file to an output file
    
    Parameters:
    -----------
    template_path : str
        Path to the template file
    output_path : str
        Path where the modified file will be saved
        
    Returns:
    --------
    None
    """
    try:
        # Read the template file
        with open(template_path, 'r', encoding='utf-8') as template_file:
            content = template_file.read()
                
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(content)
        
        print(f"Successfully created {output_path} from {template_path}\n")

    except FileNotFoundError:
        print(f"Error: Template file '{template_path}' not found", file=sys.stderr)
    except PermissionError:
        print(f"Error: Permission denied when accessing files", file=sys.stderr)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
    

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

    print(f"\nSetting up pz_data_challenge for a submission called {submission_name}\n")
    
    copy_txt_with_replacement(
        "templates/submit_template.yaml",
        f".github/workflows/submit_{submission_name}.yaml",
        "__SUBMISSION_NAME__",
        submission_name
    )

    copy_file(
        "templates/requirements_template.txt",
        f"requirements_{submission_name}.txt",
    )

    copy_txt_with_replacement(
        "templates/test_template.py",
        f"tests/test_{submission_name}.py",
        "__SUBMISSION_NAME__",
        submission_name        
    )

    setup_public_area()

    print(f"Successfully set up pz_data_challenge for a submission called {submission_name}\n")
    print(f"You should probably run the command 'git checkout -b submit/{submission_name}'")
    print(f"This will create a git branch for your submission")

    
    
    
