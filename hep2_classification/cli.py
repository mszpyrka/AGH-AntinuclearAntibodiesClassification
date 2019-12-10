""" Command line interface module. """
from collections import Counter
from timeit import default_timer as timer

import click
import numpy as np
import cv2 as cv

from hep2_classification import preprocess, segmentate
from hep2_classification.classification import ConvNetClassifier


# load classifier once
classifier = ConvNetClassifier()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('images', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', is_flag=True, help='Display more information about classification process.')
def cli(images, verbose):
    """ Classifies given IMAGES returning detected class for each one of them. """

    # collects results for each image while displaying information if needed
    results = [classify(path, verbose) for path in images]

    if verbose:
        click.secho('\n\n---------- SUMMARY ----------', fg='blue')

    # print results
    for img_path, result in zip(images, results):

        # sorted indices
        ind = (-result).argsort()

        # print result
        click.secho(classifier.classes[ind[0]], bold=True, fg='green', nl=False)
        click.echo(f' ({result[ind[0]]*100:5.2f}%) ', nl=False)
        click.echo(click.format_filename(img_path))

        # print additional data in verbose mode
        if verbose:
            click.echo(''.join(f'\t{classifier.classes[i]}: {result[i]*100:7.4f}%\n' for i in ind))


def classify(path: str, verbose: bool) -> np.ndarray:
    """ Classifies image from given path and displays information if requested. """

    # load image
    image = cv.imread(path)

    # print info
    if verbose:
        click.secho(f'\n{click.format_filename(path)}', fg='blue')
        click.echo(f'\tLoaded image of size {image.shape[0]}x{image.shape[1]} with {image.shape[2]} channels')
        click.echo(f'\tPre-processing... ', nl=False)

    # pre-processing
    start = timer()
    image = preprocess(image)
    end = timer()

    # print info
    if verbose:
        click.echo(f'\tDone in {end-start:.4f}s')
        click.echo(f'\tSegmentation... ', nl=False)

    # segmentation
    start = timer()
    segmentation_result = segmentate(image)
    end = timer()

    # print info
    if verbose:
        click.echo(f'\tDone in {end-start:.4f}s')
        click.echo(f'\tClassification... ', nl=False)

    # classification
    start = timer()
    results = classifier.classify(list(segmentation_result.cells))
    result = classifier.merge_results(results)
    end = timer()

    # print info
    if verbose:
        click.echo(f'\tDone in {end - start:.4f}s')

        # get stats
        counter = Counter(results.argmax(axis=1))

        click.echo(f'\tClassified {len(results)} cells:\t', nl=False)
        click.echo(', '.join(f'{classifier.classes[i]}: {count}' for i, count in counter.items()))
        click.echo(f'\tImage classification:\t', nl=False)
        click.echo(', '.join(f'{classifier.classes[i]}: {proc*100:4.2f}%' for i, proc in enumerate(result)))

    return result
