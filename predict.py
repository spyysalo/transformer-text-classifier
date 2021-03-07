#!/usr/bin/env python3

import os
import sys
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras.models import load_model
from transformers import AutoConfig, AutoTokenizer

from readers import READERS, get_reader
from common import timed, load_data, make_tokenization_function


# Parameter defaults
DEFAULT_BATCH_SIZE = 8


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('model_dir', metavar='DIR')
    ap.add_argument('test_data')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=DEFAULT_BATCH_SIZE,
                    help='batch size for predictions')
    ap.add_argument('--input_format', choices=READERS.keys(),
                    default=list(READERS.keys())[0],
                    help='input file format')
    return ap


def get_custom_objects():
    """Return dictionary of custom objects required for load_model."""
    from transformers import AdamWeightDecay
    from tensorflow_addons.metrics import F1Score
    return {
        'AdamWeightDecay': AdamWeightDecay,
        'F1Score': F1Score,
    }


def load_trained_model(directory):
    config = AutoConfig.from_pretrained(directory)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        config=config
    )
    model = load_model(
        os.path.join(directory, 'model.hdf5'),
        custom_objects=get_custom_objects()
    )
    labels = []
    with open(os.path.join(directory, 'labels.txt')) as f:
        for ln, l in enumerate(f, start=1):
            labels.append(l.rstrip('\n'))
    return model, tokenizer, labels, config


def main(argv):
    options= argparser().parse_args()

    model, tokenizer, labels, config = load_trained_model(options.model_dir)
    test_texts, test_labels = load_data(
        options.test_data,
        options.input_format,
        config.multiclass
    )

    label_encoder = MultiLabelBinarizer(classes=labels)

    tokenize = make_tokenization_function(tokenizer, config.seq_len)

    metrics_values = model.evaluate(
        tokenize(test_texts),
        label_encoder.fit_transform(test_labels),
        batch_size=options.batch_size
    )    
    for name, value in zip(model.metrics_names, metrics_values):
        print(f'{name}\t{value}')

    predictions = model.predict(
        tokenize(test_texts),
        verbose=1,
        batch_size=options.batch_size
    )

    assert len(test_texts) == len(predictions)
    for text, gold, preds in zip(test_texts, test_labels, predictions):
        if config.multiclass:
            pred_labels = [labels[preds.argmax()]]
        else:
            pred_labels = [labels[i] for i, v in enumerate(preds) if v > 0.5]
        print('{}\t{}'.format(','.join(pred_labels), text))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
