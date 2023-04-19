import logging
import importlib

import click
import yaml


@click.group()
def _dispatch(): pass


@_dispatch.command()
@click.argument('config')
def train(config):
    logger = _create_logger()
    config = _load_config(config)
    entry = config['entry']
    entry = _import_object(entry)
    entry(config=config, logger=logger)


def _create_logger():
    logger = logging.getLogger('tinynlp')
    logger.setLevel('INFO')
    logging.basicConfig()
    return logger


def _load_config(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def _import_object(target):
    mod, name = target.split(':')
    mod = importlib.import_module(mod)
    ob = getattr(mod, name)
    return ob


if __name__ == '__main__':
    _dispatch()
