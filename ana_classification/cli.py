""" Command line interface module. """
import os
from collections import Counter
from contextlib import contextmanager
from timeit import default_timer as timer
from typing import Optional, List, Tuple

import click
import numpy as np
import cv2 as cv

from ana_classification import preprocess, segmentate, ConvNetCellClassifier, BaseCellClassifier, overlay, SegmentationResult

# load classifier once
classifier: BaseCellClassifier = ConvNetCellClassifier()

# cli options
cli_verbose: bool = False
cli_overlays: Optional[str] = None


@click.command(
    context_settings=dict(help_option_names=['-h', '--help'])
)
@click.argument(
    'images',
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    help='Display more information about classification process.'
)
@click.option(
    '-o', '--overlays',
    type=click.Path(file_okay=False, writable=True),
    help='Save images (in provided directory) with an overlay about segmentation and classification'
)
def cli(images, verbose, overlays):
    """ Classifies given IMAGES returning detected class for each one of them. """

    # set cli options globally
    global cli_verbose, cli_overlays
    cli_verbose = verbose
    cli_overlays = overlays

    # handle each image
    for path in images:
        handle_image(path)


def handle_image(path: str):
    """ Classifies image from given path and displays information if requested. """

    # load image
    image = cv.imread(path)

    # print info about image
    echo_verbose(f'\n{click.format_filename(path)}', fg='blue')
    echo_verbose(f'\tLoaded image of size {image.shape[0]}x{image.shape[1]} with {image.shape[2]} channels')

    # preprocessing
    with timeit('Pre-processing'):
        preprocessed = preprocess(image)

    # segmentation
    with timeit('Segmentation'):
        segmentation_result = segmentate(preprocessed)

    # classification
    with timeit('Classification'):
        cells_results = classifier.classify(list(segmentation_result.cells))
        image_result = classifier.merge_results(cells_results)

    # get stats
    counter = Counter(cells_results.argmax(axis=1))
    results = sorted([
        (image_result[i]*100, counter[i], classifier.classes[i])
        for i in range(len(image_result))
    ], reverse=True)

    # print info about classification
    echo_verbose(f'\tClassified {len(cells_results)} cells:\t', nl=False)
    echo_verbose(', '.join(f'{name}: {count}' for _, count, name in results))
    echo_verbose(f'\tImage classification:\t', nl=False)
    echo_verbose(', '.join(f'{name}: {proc:4.2f}%' for proc, _, name in results))

    # create overlay
    if cli_overlays is not None:
        create_overlay(path, preprocessed, segmentation_result, cells_results, results)

    # print results in verbose mode
    echo_verbose(f'\tImage classified as ', nl=False)
    echo_verbose(results[0][2], bold=True, fg='green')

    # print results in normal mode
    if not cli_verbose:
        click.secho(results[0][2], bold=True, fg='green', nl=False)
        click.echo(f' ({results[0][0]:5.2f}%) ', nl=False)
        click.echo(click.format_filename(path))


def create_overlay(path: str, preprocessed: np.ndarray,
                   segmentation_result: SegmentationResult, cells_results: np.ndarray,
                   results: List[Tuple[float, int, str]]):
    """ Creates and saves an overlaid image. """

    # create path if needed
    if not os.path.exists(cli_overlays):
        os.makedirs(cli_overlays)

    # get out path
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(cli_overlays, name + '-overlay.png')

    # create boxes data
    boxes = [
        (
            (seg.y, seg.x),
            (seg.y + seg.mask.shape[1], seg.x + seg.mask.shape[0]),
            f'{classifier.classes[cell_result.argmax()]} {cell_result.max() * 100:4.2f}'
        )
        for seg, cell_result in zip(segmentation_result.segments, cells_results)
    ]

    # create image
    img = overlay.draw_overlay(preprocessed, boxes, results)

    # save image
    cv.imwrite(out_path, img)

    # print info
    echo_verbose(f'\tSaved overlaid image to {out_path}')


def echo_verbose(*args, **kwargs):
    """ Works just like click.secho() but respects cli_verbose flag. """
    if cli_verbose:
        click.secho(*args, **kwargs)


@contextmanager
def timeit(name: str):
    """ Context manager that prints execution time if requested. """
    echo_verbose(f'\t{name}... ', nl=False)

    start = timer()
    yield
    end = timer()

    echo_verbose(f'\tDone in {end-start:.4f}s')

