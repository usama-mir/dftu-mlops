# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    train_content = []
    for i in range(5):
        train_content.append(np.load(f"{input_filepath}/train_{i}.npz"))

    train_data = torch.tensor(np.concatenate([c['images'] for c in train_content])).reshape(-1, 1, 28, 28)
    train_labels = torch.tensor(np.concatenate([c['labels'] for c in train_content]))
    test_content = (np.load(f"{input_filepath}/test.npz"))
    test_data = torch.tensor(test_content['images']).reshape(-1, 1, 28, 28)
    test_laels = torch.tensor(test_content['labels'])
    torch.save(train_data, f"{output_filepath}/train_data.pt")
    torch.save(train_labels, f"{output_filepath}/train_labels.pt")
    torch.save(test_data, f"{output_filepath}/test_data.pt")
    torch.save(test_laels, f"{output_filepath}/test_labels.pt")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
