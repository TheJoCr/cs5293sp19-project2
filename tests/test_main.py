
from project2 import redactor


def test_argparse():
    # Very complicated args:
    args = [
        '--input', 'test.txt',
        '--input', 'other/test.txt',
        '--output', 'output_test', 
        '--stats', 'test_stats.txt',
        # Pick a random subset of these three
        '--names', '--genders', '--addresses',
        '--concept', 'fancy test',
        '--concept', 'other test',
    ]
    
    args = main.parser.parse_args(args)
    assert args.input == ['test.txt', 'other/test.txt']
    assert args.output == 'output_test'
    assert args.stats == 'test_stats.txt'
    assert args.names 
    assert args.genders 
    assert not args.dates
    assert args.addresses
    assert not args.phones
    assert args.concept == ['fancy test', 'other test']

# Test case that is broad
TO_OBFUSCATE = """
On March 5th, 1995, Mary called from 405-555-5555, and said that
she was going to be in town for a few weeks, until May 10th. She
was staying at 121 W. Bobson St., though I wasn't sure if that
was in Denver or Boulder. 
"""
EXPECTED = """
On████████████████, ████ called from ████████████, and said that
███ was going to be in town for a few weeks, until███████th. ███
was staying at █████████████████, though I wasn't sure if that
was in Denver or Boulder. 
"""

def test_obfuscate_text():
    # Tricky way to create a new type with arbitrary attributes
    args = type('', (), {})()
    args.names = True
    args.genders = True
    args.dates = True
    args.addresses = True
    args.phones = True
    args.concept = []

    clean, stats = main.obfuscate_text(args, TO_OBFUSCATE)
    print(clean)
    assert clean == EXPECTED
    assert stats['addresses'] == 1

import os
import shutil
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
    args = type('', (), {})()
    args.names = True
    args.genders = True
    args.dates = True
    args.addresses = True
    args.phones = True
    args.concept = []
    stats = main.obfuscate_file(args, f, d)

    # now, temp/input.txt.redacted should have EXPECTED as contents
    with open(f + '.redacted', 'r') as o:
        result = o.read()
    
    # Cleanup
    shutil.rmtree(d)

    # Assertion time!
    assert result == EXPECTED
    assert stats['dates'] == 2

    
def test_execute():
    # Make a directory that will be cleaned up, and stick example files in it
    d = 'temp'
    if os.path.exists(d):
        # Clean up
        shutil.rmtree(d)
    os.mkdir(d)
    f1 = d + '/input1.txt'
    with open(f1, 'w') as o:
        o.write(TO_OBFUSCATE)
    f2 = d + '/input2.txt'
    with open(f2, 'w') as o:
        o.write(TO_OBFUSCATE)

    args = type('', (), {})()
    args.input = ['temp/*.txt']
    args.output = 'temp'
    args.stats = 'temp/stats.txt'
    args.concept = ['cat', 'dog']
    args.names = True
    args.genders = True
    args.dates = True
    args.addresses = True
    args.phones = True
    
    stats = main.execute(args)

    # now, temp/input.txt.redacted should have EXPECTED as contents
    with open(f1 + '.redacted', 'r') as o:
        result1 = o.read()
    with open(f2 + '.redacted', 'r') as o:
        result2 = o.read()
    with open('temp/stats.txt', 'r') as o:
        stats = o.read()
    # Cleanup
    shutil.rmtree(d)
    assert result1 == EXPECTED
    assert result2 == EXPECTED

