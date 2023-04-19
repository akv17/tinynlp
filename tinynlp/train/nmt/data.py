import json
import dataclasses


@dataclasses.dataclass
class Sentence:
    lang: str
    text: str


@dataclasses.dataclass
class Sample:
    src: Sentence
    dst: Sentence


class SampleCollection:

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples = [
            Sample(
                src=Sentence(**rec['src']),
                dst=Sentence(**rec['dst']),
            )
            for rec in data
        ]
        return samples
