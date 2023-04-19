import os

import torch
import numpy as np

from .tokenizer import Tokenizer
from .model import Model


class Encoder:
    
    def __init__(self, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seq_len_real = self.seq_len - 2

    def run(self, text):
        src_ids = self.tokenizer.encode(text)
        src_ids = src_ids[:self.seq_len_real]
        src_ids = [self.tokenizer.start_id] + src_ids + [self.tokenizer.end_id]
        src_pad_size = self.seq_len - len(src_ids)
        src_ids += [self.tokenizer.pad_id] * src_pad_size
        attn_mask = np.array(src_ids) == self.tokenizer.pad_id
        src_ids = torch.as_tensor(src_ids)
        attn_mask = torch.from_numpy(attn_mask)
        enc = {'ids': src_ids, 'attn_mask': attn_mask}
        return enc


class Decoder:
    
    def __init__(self, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len


class Predictor:

    @classmethod
    def from_config(cls, config):
        path = config['global']['path']
        vocab_size = config['tokenizer']['vocab_size']
        src_tokenizer = Tokenizer.load(path=path, postfix='src', vocab_size=vocab_size)
        dst_tokenizer = Tokenizer.load(path=path, postfix='dst', vocab_size=vocab_size)
        weights = os.path.join(path, 'model.pth')
        weights = torch.load(weights, map_location='cpu')
        model = Model(vocab_size)
        model.load_state_dict(weights)
        model.eval()
        encoder = Encoder(src_tokenizer)
        decoder = Decoder(dst_tokenizer)
        ob = cls(encoder=encoder, decoder=decoder, model=model)
        return ob

    def __init__(self, encoder, decoder, model):
        self.encoder = encoder
        self.decoder = decoder
        self.model = model
    
    def run(self, text):
        with torch.no_grad():
            enc = self.encoder.run(text)
            _attn_mask = enc.pop('attn_mask')
            enc = {k: v.unsqueeze(0) for k, v in enc.items()}
            enc = self.model.encode(**enc)
            dst_list = [self.decoder.tokenizer.start_id]
            state = None
            stop = False
            text = [self.decoder.tokenizer.START_TOKEN]
            yield text[-1]
            while not stop:
                dst = torch.as_tensor(dst_list).unsqueeze(0)
                attn_mask = _attn_mask.view(1, 1, -1)
                attn_mask = torch.tile(attn_mask, [1, dst.size(1), 1])
                logits, state, attn = self.model.decode(enc, dst, attn_mask, state)
                logits = logits[0, -1].view(-1)
                logits = torch.softmax(logits, -1)
                # next_ix = torch.multinomial(logits, 1).item()
                next_ix = torch.argmax(logits).item()
                next_text = self.decoder.tokenizer.decode([next_ix])
                next_text = next_text.replace('#', '')
                text.append(next_text)
                yield text[-1]
                dst_list.append(next_ix)
                if next_ix == self.decoder.tokenizer.end_id or len(dst_list) == self.decoder.seq_len:
                    stop = True
        print()
