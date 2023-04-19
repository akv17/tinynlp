import numpy as np


class Display:

    def __init__(self, decoder):
        self.decoder = decoder

    def run(self, encoding):
        enc = encoding
        src_ids = enc['src'].cpu().numpy()
        dst_ids = enc['dst'].cpu().numpy()
        dst_targets = enc['targets'].cpu().numpy()
        dst_mask = enc['targets_masks'].cpu().numpy()
        attn_mask = enc['attn_masks'].cpu().numpy()
        src_mask = ~attn_mask[0]

        print(f'src:      {src_ids.shape}')
        print(f'dst:      {dst_ids.shape}')
        print(f'targets:  {dst_targets.shape}')
        print(f'mask:     {dst_mask.shape}')
        print()

        src_text = self.decoder.decode_src(src_ids)
        dst_text = self.decoder.decode_dst(dst_ids)
        targets_text = self.decoder.decode_dst(dst_targets)
        print(f'src:        {repr(src_text)}')
        print(f'dst:        {repr(dst_text)}')
        print(f'targets:    {repr(targets_text)}')
        print()

        src_dec = np.array([self.decoder.src_tokenizer.decode_one(i) for i in src_ids])
        dst_dec = np.array([self.decoder.dst_tokenizer.decode_one(i) for i in dst_ids])
        targets_dec = np.array([self.decoder.dst_tokenizer.decode_one(i) for i in dst_targets])

        print(f'src:      {src_dec[src_mask]}')
        print(f'dst:      {dst_dec[dst_mask]}')
        print(f'targets:  {targets_dec[dst_mask]}')
        print()

        # breakpoint()
        # src_text = self.decoder.decode_src(src_ids)
        # dst_text = self.decoder.decode_dst(dst_ids)
        # dst_targets = self.decoder.decode_dst(dst_targets)

        # print(f'src: {src_text}')
        # print(f'dst: {dst_text}')
        # print(f'targets: {dst_targets}')
        # print(f'mask: {dst_targets}')
