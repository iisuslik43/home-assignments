#! /usr/bin/env python3

__all__ = [
    'read_rgb_f32',
    'read_grayscale_f32'
]

import click
import cv2
import numpy as np
import pims


@pims.pipeline
def _to_float32(rgb):
    return (rgb / 255.0).astype(np.float32)


@pims.pipeline
def _to_grayscale(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def read_rgb_f32(path_to_sequence: str) -> pims.FramesSequence:
    return _to_float32(pims.open(path_to_sequence))


def read_grayscale_f32(path_to_sequence: str) -> pims.FramesSequence:
    return _to_grayscale(read_rgb_f32(path_to_sequence))


@click.command()
@click.argument('frame_sequence')
def _cli(frame_sequence):
    sequence = read_grayscale_f32(frame_sequence)
    click.echo("Press 'q' to stop")
    for i, image in enumerate(sequence):
        cv2.imwrite('{}.png'.format(i), (image * 255).astype(np.uint8))


if __name__ == '__main__':
    _cli()  # pylint:disable=no-value-for-parameter
