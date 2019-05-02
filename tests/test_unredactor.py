import numpy as np
from project2 import unredactor as un
import tempfile

test_dict = {
    'great': 0,
    'really': 1,
    'some': 2,
    'text': 3,
}
def test_bag_of_words():
    test_str = "Some Really great text with text in it twice"
    words = un.bag_of_words(test_str, test_dict)
    assert len(words) == 4
    assert words[test_dict['text']] == 2


dict_file_contents = \
'''great
really
some
text '''

def test_get_dictionary():
    with tempfile.SpooledTemporaryFile(mode='rw') as dict_file:
        dict_file.write(dict_file_contents)
        dict_file.seek(0)
        d = un.get_dictionary(dict_file)
        assert len(d) == 4


training_file1_contents = \
'''2
great
some
test some great I mean really great text
'''

training_file2_contents = \
'''1
great
test I mean really great text some some some
'''

def test_read_training_files():
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(tmpdirname + '/file1.txt', 'w') as f1:
            f1.write(training_file1_contents)
        with open(tmpdirname + '/file2.txt', 'w') as f2:
            f2.write(training_file2_contents)
        data, answers, name_ids = un.read_training_files( [
                tmpdirname + '/file1.txt', 
                tmpdirname + '/file2.txt', 
            ],  
            test_dict 
        ) 
        assert (answers == np.array([0, 1, 0])).all()
        assert data.shape == (3,5)
        assert 'great' in name_ids

def test_train():
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(tmpdirname + '/dict.txt', 'w') as d:
            d.write(dict_file_contents)
        with open(tmpdirname + '/file1.redacted', 'w') as f1:
            f1.write(training_file1_contents)
        with open(tmpdirname + '/file2.redacted', 'w') as f2:
            f2.write(training_file2_contents)
        clf, _, _= un.train(tmpdirname)
        assert clf.predict(np.array([[4,0,2,0,0]])) is not None
