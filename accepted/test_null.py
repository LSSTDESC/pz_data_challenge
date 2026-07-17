import os
from pathlib import Path
import pytest

# Put needed import here

def test_null(
    setup_public_area: int,
) -> None:
    """Make sure that the public area is set up
    """
    
    assert setup_public_area == 0
