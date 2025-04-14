#!/usr/bin/python
import argparse

SOLVER = "CBS"


def create_parser():
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')

    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s), e.g., "instances/exp0.txt"')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS, PBS, Independent, Prioritized}), defaults to ' + str(
                            SOLVER))
    return parser
