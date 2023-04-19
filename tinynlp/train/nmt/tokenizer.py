import os

from tokenizers import (
    Tokenizer as _Tokenizer,
    models,
    trainers,
    decoders,
    normalizers,
    pre_tokenizers
)


class Tokenizer:
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    START_TOKEN = '[START]'
    END_TOKEN = '[END]'
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]

    @classmethod
    def load(cls, path, postfix=None, vocab_size=8192):
        name = f'tokenizer_{postfix}.json' if postfix is not None else 'tokenizer.json'
        fp = os.path.join(path, name)
        tokenizer = _Tokenizer.from_file(fp)
        tokenizer.decoder = decoders.WordPiece()
        ob = cls(vocab_size=vocab_size)
        ob.tokenizer = tokenizer
        return ob

    def __init__(self, vocab_size=8912):
        self.vocab_size = vocab_size

        self.tokenizer = _Tokenizer(models.WordPiece(unk_token=self.UNK_TOKEN, max_input_chars_per_word=512))
        self.normalizer = normalizers.Sequence([
            normalizers.NFC(),
        ])
        self.tokenizer.normalizer = self.normalizer
        self.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.Punctuation(),
        ])
        self.tokenizer.decoder = decoders.WordPiece()

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id(self.PAD_TOKEN)
    
    @property
    def start_id(self):
        return self.tokenizer.token_to_id(self.START_TOKEN)
    
    @property
    def end_id(self):
        return self.tokenizer.token_to_id(self.END_TOKEN)

    def encode(self, text):
        ids = self.tokenizer.encode(text).ids
        return ids
    
    def decode(self, ids, skip_special=True):
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special)
        return text

    def decode_one(self, i):
        text = self.tokenizer.decode([i], skip_special_tokens=False)
        return text

    def train(self, texts):
        trainer = trainers.WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=self.SPECIAL_TOKENS)
        self.tokenizer.train_from_iterator(texts, trainer)

    def save(self, path, postfix=None):
        os.makedirs(path, exist_ok=True)
        name = f'tokenizer_{postfix}.json' if postfix is not None else 'tokenizer.json'
        fp = os.path.join(path, name)
        self.tokenizer.save(fp)
