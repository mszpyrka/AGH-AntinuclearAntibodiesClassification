import click


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('images', nargs=-1, type=click.File('rb', lazy=True))
@click.option('-v', '--verbose', is_flag=True, help='Display more information about classification process.')
@click.option('-c', '--config', type=click.File('rb'), help='Load model parameters from given file.')
def cli(images, verbose, config):
    """ Classifies given IMAGES returning detected class for each one of them. """
    raise NotImplementedError()
