# Main file which supports all of the orchestration and handling
# user input.
import glob
import argparse
import os
import os.path
import nltk


def mark_names(text):
    sentences = nltk.sent_tokenize(text)
    name_indexes = []
    i = 0
    for s in sentences:
        words = nltk.word_tokenize(s)
        tags = nltk.pos_tag(words)
        chunked = nltk.ne_chunk(tags)
        for chunk in chunked:
            if not (hasattr(chunk, 'label') and chunk.label() == 'PERSON'):
                continue
            # We've got ourselves a person!
            for word in chunk:
                remove_from = text.index(word[0], i)
                name_indexes.append(
                    (remove_from, remove_from + len(word[0]))
                )
        i += len(s) 
    return name_indexes

def obfuscate_text(text):
    """
    Handles logic for the actual obfuscation of
    a block of text, finding everything that needs
    to be obfuscated and then replacing it with
    the full block character. Returns stats about
    what got removed.
    """
    redact = []
    names = mark_names(text)
    redact += names
    # Copy text into a mutable buffer:
    names_removed = []
    text = list(text)
    for start, end in redact:
        names_removed.append(''.join(text[start:end]))
        for i in range(start, end):
            text[i] = '\u2588'
    return names_removed, ''.join(text) 

def obfuscate_file(f, output_dir):
    """
    Obfuscates the file at f per args and stores
    the result in output_dir.
    """
    with open(f, 'r') as input_file:
        text = input_file.read()
    removed_names, ob_text = obfuscate_text(text)
    file_name = os.path.basename(f) + '.redacted'
    output_file_path = output_dir + '/' + file_name
    with open(output_file_path, 'w') as output_file:
        output_file.write('%d\n' % len(removed_names))
        for name in removed_names:
            output_file.write('%s\n' % name)
        output_file.write(ob_text)
    return removed_names

def execute(args):
    input_files = [f for input_glob in args.input for f in glob.glob(input_glob)]
    if len(input_files) == 0:
        print("No input files found! Exiting...")
        exit(1)
    # setup_stats(args.stats)
    output_dir = args.output
    # Try to make the dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    removed_names = []
    for f in input_files:
        removed_names += obfuscate_file(f, output_dir)
    print("Removed names: %s" % sorted(list(set(removed_names))))

parser = argparse.ArgumentParser(
    description='Obfuscate files for testing deobfuscater'
)
# Input and Output
parser.add_argument('--input', action='append', required=True)
parser.add_argument('--output', default='out')

def main():
    """
    Executes the program, parsing args, locating input files,
    running the obfuscation logic, and printing out stats.
    """
    # parse args
    args = parser.parse_args()
    execute(args)

if __name__ == "__main__":
    main()

