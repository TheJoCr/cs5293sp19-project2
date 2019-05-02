import os
import shutil
from project2 import redactor

# Test case that is broad
TO_OBFUSCATE = """
On March 5th, 1995, Mary called from 405-555-5555, and said that
she was going to be in town for a few weeks, until May 10th. She
"""
EXPECTED_TEXT = """
On March 5th, 1995, ████ called from 405-555-5555, and said that
she was going to be in town for a few weeks, until May 10th. She
"""
EXPECTED_FILE = """1
Mary

On March 5th, 1995, ████ called from 405-555-5555, and said that
she was going to be in town for a few weeks, until May 10th. She
"""

def test_obfuscate_text():
    names, clean = redactor.obfuscate_text(TO_OBFUSCATE)
    assert clean == EXPECTED_TEXT

def test_obfuscate_file():
    # Make a directory that will be cleaned up, and stick the above file in it.
    d = 'temp'
    if os.path.exists(d):
        # Clean up
        shutil.rmtree(d)
    os.mkdir(d)
    f = d + '/input.txt'
    with open(f, 'w') as o:
        o.write(TO_OBFUSCATE)
    removed_names = redactor.obfuscate_file(f, d)

    # now, temp/input.txt.redacted should have EXPECTED as contents
    with open(f + '.redacted', 'r') as o:
        result = o.read()
    
    # Cleanup
    shutil.rmtree(d)

    # Assertion time!
    assert result == EXPECTED_FILE
