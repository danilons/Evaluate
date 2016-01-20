# coding: utf-8
from __future__ import unicode_literals
from __future__ import division
import os
import nltk
import flask
import logging
import unicodedata
import string
import numpy as np
import pandas as pd
import skimage.data
import cStringIO as StringIO
from PIL import Image
from nltk.corpus import stopwords

app = flask.Flask(__name__)

@app.route("/")
def main():
    return flask.render_template('index.html', has_result=False)


@app.route("/query", methods=['GET'])
def query():
    term = flask.request.args.get('term', '')
    logging.info('searching term: %s', term)
    try:
        class1, verb, class2 = tokenize(term)
    except Exception as err:
        logging.info('Tokenization error: %s', err)
        return flask.render_template('index.html', has_result=True, result=(False, 'Cannot tokenize term.'))

    index1 = app.classes.index(class1)
    index2 = app.classes.index(class2)

    loc1 = np.where(app.objs[:, index1] > 0)[0]
    loc2 = np.where(app.objs[:, index2] > 0)[0]
    loc_ = np.intersect1d(loc1, loc2)

    index3 = app.mereo.index(app.synset[verb])
    loc3 = np.where(app.rels[:, index3] > 0)[0]
    indices = np.intersect1d(loc_, loc3)

    result = []
    for position in indices:
        # probability of choosing a specific column
        score1 = (app.objs[position, index1] + app.objs[position, index2]) / (2 * app.objs[position].sum())
        score2 = app.rels[position, index3] / app.rels[position].sum()
        index_ = '{0:06d}'.format(app.index[position])

        # reading image from local path
        im_path = os.path.join('static', 'imgs', index_ + '.jpg')
        result.append((True, im_path, '{:.2f}'.format(score1 * score2)))

    result = sorted(result, key=lambda x: x[2], reverse=True)

    for n, idx in enumerate(app.index):
        if n not in indices:
            index_ = '{0:06d}'.format(idx)
            im_path = os.path.join('static', 'imgs', index_ + '.jpg')
            result.append((False, im_path, 0.0))

    return flask.render_template('list.html', has_result=True, result=result)


def embed_image_html(image):
    """
    convert an image array to base64 image
    :param image: image as array
    :return: base64 image
    """
    string_buf = StringIO.StringIO()
    image = image.resize((256, 256))
    image.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def flat(in_):
    """
    flatten an array read from csv file

    :param in_: csv-like array
    :return: flattened
    """
    out = np.array([eval(data) for data in in_[0].replace('[', '').replace(']','').split()])
    for x in in_[1:]:
        arr = np.array([eval(data) for data in x.replace('[', '').replace(']','').split()])
        out = np.vstack((out, arr))
    return out

def _cleaned_text(text):
    """
    lowercases, remove punctuation, replaces \ and / by ' ' and - by '' from a unicode text

    :param text: [string] input text
    :return: [string] cleaned text
    """
    if not isinstance(text, unicode):
        text = unicode(text)

    lowers = text.lower()
    lowers = lowers.replace('-', '').replace('\\', ' ').replace('/', ' ')

    # remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    return lowers.translate(remove_punctuation_map)


def _asciifolding(_str):
    """
    replace non-ascii characteres to its ascii equivalent.
    Ex: é -> e, ã -> a, and so forth.

    We use the NFKD unicode normalization method. See more on:
      http://www.unicode.org/reports/tr15/

    :param _str: [string] input string
    :return
    """
    try:
        unicode_str = unicode(_str, 'utf-8')
    except:
        unicode_str = _str

    if unicode_str:
        return unicode(unicodedata.normalize('NFKD', unicode_str).encode('ascii', 'ignore'))
    else:
        return u''


def _get_tokens(text):
    """
    Turn text into a list of word tokens

    :params text: [string] input text
    :return: [list] list of tokens
    """
    cleaned = _cleaned_text(text)
    tokens = nltk.word_tokenize(cleaned)
    return tokens


def _filter_stopwords(tokens):
    """
    Remove stop words.
    We use a enchanced stopwords list, defined on the STOPWORDS global var.

      https://en.wikipedia.org/wiki/Stop_words

    :param tokens: [list] tokens list
    :return: [list] filtered tokens list
    """
    return [w for w in tokens if w not in stopwords.words()]


def _stem_tokens(tokens):
    """
    Apply stemming to each token.

    The stemming process reduces every word to a common morphological root form.
    See more about stemming here:

      https://en.wikipedia.org/wiki/Stemming

    We use the RSLPStemmer, which is a stemming algorithm developed specifically
    for brazilian portuguese. See more on the link below:

      http://www.inf.ufrgs.br/~viviane/rslp/

    :param tokens: [list] tokens list
    :return: [list] list of stems
    """
    stemmer = RSLPStemmer()
    return [stemmer.stem(item) for item in tokens]


def tokenize(text, with_stemming=False):
    """
    Common tokenizer. Receives a unicode generic text and returns a filtered list of
    word tokens.

    :param text: [unicode] input text
    :param with_stemming: [bool] check if should apply stemming to the tokens
    :return: [list] tokens list
    """
    # apply NKFD ascii folding
    text = _asciifolding(text)

    # get word tokens
    tokens = _get_tokens(text)

    # remove stopwords
    tokens = _filter_stopwords(tokens)

    if with_stemming:
        # apply RSLP stemming
        tokens = _stem_tokens(tokens)

    return tokens


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    app.dataset_path = '/Users/danilonunes/workspace/datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages'
    # app.dataset_path = 'static/imgs'

    app.classes = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    app.mereo = ('EQ', 'PO', 'PP', 'PPi', 'DC')

    app.synset = {'riding': 'PO',
                  'near': 'DC',
                  'inside': 'PP',
                  'behind': 'PO',
                  'far': 'DC',
                  'within': 'PP',
                  'outside': 'PPi',
                  'equal': 'EQ',
                  'same': 'EQ'}

    dataframe = pd.DataFrame.from_csv('dataset.csv')
    app.index = dataframe.index
    app.objs = flat(dataframe.classes.as_matrix())
    app.rels = flat(dataframe.relations.as_matrix())
    # run the app in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)