import click
import yaml

from .predictor import Predictor


@click.command()
@click.argument('config')
@click.argument('text')
def main(config, text):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    predictor = Predictor.from_config(config)
    for token in predictor.run(text):
        print(token, end=' ')


if __name__ == '__main__':
    main()
