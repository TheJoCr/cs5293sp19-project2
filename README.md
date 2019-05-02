# CS 5293 Project 2 Jordan Crawford

## Overview

This is a tool designed to redact comment files as supplied by the imdb
comment dataset, as well as to attempt to unredact those same files. 

There are two modules, aptly named the 'redactor' and the 'unredactor'.
The first is in charge of preprocessing the comment files by removing
all names that are identified by the nltk named entity recognition
tools, as well as rewrite them in an easy to use format. The second is
a simple tool designed to undo the work of the first. Given redacted
files, it attempts to guess what names should replace redacted fields.

## Usage

To run the redactor, execute: 

```bash 
pipenv run python -m project2.redactor [-h] --input INPUT [--output OUTPUT]
```

Here, `--input` is a glob of files and may be specified multiple times
to include multiple input globs. `--output` specifies a single directory
for storing the results. For example, to create the structure used in
testing, the following commands were used:

```bash 
pipenv run python -m project2.redactor \
    --input 'aclImdb/train/pos/*.txt' \
    --input 'aclImdb/train/neg/*.txt' \
    --input 'aclImdb/train/unsup/*.txt' \
    --output redacted_data

pipenv run python -m project2.redactor \
    --input 'aclImdb/test/pos/*.txt' \
    --input 'aclImdb/test/neg/*.txt' \
    --output redacted_data_test
```

This creates 2 directories. The first, at `redacted_data`, is a training
directory with 75,000 total training examples. The second, at
`redacted_data_test` is a test directory with just 25,000 test cases for
evaluating the accuracy of the unredactor tool. Note that the unredactor
expects a file called `dict.txt` to exist in both of these directories. 
This file must be copied manually from the `aclImdb/imdb.vocab` folder.


To run the unredactor, execute:

```bash
pipenv run python -m project2.unredactor [-h] --input-dir INPUT_DIR --test-dir TEST_DIR
```

For example, given the setup created by the above commands, the
following command will process inputs, train a model, and then calculate
the accuracy of that model against the test set.

```bash
pipenv run python -m project2.unredactor --input-dir redacted_data --test-dir redacted_data_test
```

## Unredactor Methodology

The unredactor uses the Naive Bayes algorithm to determine what name
should be used to replace any given redacted word. The procedure for
preprocessing, training, and analyzing results proceeds roughly like the
following:

First, filter the list of names. We look at all redacted names in the
input set and order them by frequency. Because we are considering names
as the 'classes' in Naive Bayes, the memory/time used is proportional to
the number of names used in training: we therefore made the decision to
use only the top 254 most popular names in the training dataset. 

Second, extract features. For each input comment, we extract the full
set of redacted names, as well as the full text of the comment. For each
redacted name that appears in the top 254 most common, we store a
feature vector consisting of the number of redacted letters in the name
and a bag-of-words representation of the comment. This has the upside of
being extremely simple, but the downside that many redacted words may
have very similar feature sets if the come from the same comment. These
feature vectors are aggregated together as a sparse matrix and cached to
disk to support quick runtimes.

Third, run the `ComplementNB` algorithm from sklearn, that implements
Naive Bayes in a way that works particularly well for text
classification task. This produces the classifier.

Finally, repeat the second step on the test data to produce data with
identical feature vectors, and apply the classifier to those vectors.
Compute the portion of test samples which were unredacted (i.e.
classified by the naive bayes algorithm) correctly, and report that to
the user. 

## Results and Discussion

The methodology described above produces an accuracy rate of around
8.5%. While this isn't phenomenal, it's a dramatic improvement over the
random classifier which achieves an accuracy rate of just (1/255 = .3%).

The data also suggest that some over fitting is likely occurring in the
training process, since the accuracy of the unredactor on the test data 
is 33.9%. This suggests that at least one of our problems is the lack of
data - hopefully, a larger training dataset would help combat
over fitting on the test set. It also points to the need for a more
sophisticated feature vector that is better able to distinguish between
different obfuscated words in the same comment. 

The process is surprisingly performant. While the data-preprocessing and
feature vector extraction steps take, on my machine, upwards of 20
minutes, once the features vectors are cached, total runtime drops to
less than 30 seconds for training and comparing against the test set. 

This means that rapid experimentation was possible with the
training-time tasks, while iteration on the feature vectors had
extremely slow development turn around. 

## Future Research

Future research would have to start by analyzing the effects of adding
additional information to the feature vectors. In particular,
information about the size of and distance to adjacent obfuscated fields
would add a wealth of contextual information that would facilitate
increasing the number of differentiators between two obfuscated names in
the same comment. This could dramatically decrease over fitting and
increase accuracy rates. 


## Tests

To run tests execute: ``` pipenv run python setup.py test ``` This
prints the test result (generated via pytest) to standard out. 
