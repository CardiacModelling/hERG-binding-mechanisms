#
# Fitting I/O methods
#

from . import models, parameters


def cmd(title=''):
    """
    Parses command line args, returns a tuple ``()``
    """
    # Get model
    import argparse
    parser = argparse.ArgumentParser(title)
    parser.add_argument(
        '-m', '--model',
        nargs='?',
        choices=models._model_files.keys(),
        default='m12',
        help='The model to use')
    parser.add_argument(
        '--ic50',
        nargs='?',
        choices=parameters.ic50.keys(),
        default='li',
        help='IC50 values to use')
    parser.add_argument(
        '-p', '--protocol',
        nargs='+',
        choices=['staircase'],
        default=['staircase'],
        help='The protocol to use')
    parser.add_argument(
        '-pc', '--percent_current',
        action='store_true',
        default=False,
        help='The model to use')
    args = parser.parse_args()

    return args

