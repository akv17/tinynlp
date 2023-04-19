import torch
import numpy as np


class Encoder:

    def __init__(self, src_tokenizer, dst_tokenizer, src_seq_len=64, dst_seq_len=64):
        self.src_tokenizer = src_tokenizer
        self.dst_tokenizer = dst_tokenizer
        self.src_seq_len = src_seq_len
        self.src_seq_len_real = self.src_seq_len - 2
        self.dst_seq_len = dst_seq_len
        self.dst_seq_len_real = self.dst_seq_len - 2

    def run(self, sample):
        src_ids = self.src_tokenizer.encode(sample.src.text)
        src_ids = src_ids[:self.src_seq_len_real]
        src_ids = [self.src_tokenizer.start_id] + src_ids + [self.src_tokenizer.end_id]
        src_pad_size = self.src_seq_len - len(src_ids)
        src_ids += [self.src_tokenizer.pad_id] * src_pad_size
        attn_mask = np.array(src_ids) == self.src_tokenizer.pad_id
        
        dst_ids = self.dst_tokenizer.encode(sample.dst.text)
        dst_ids = dst_ids[:self.dst_seq_len_real]
        dst_ids = [self.dst_tokenizer.start_id] + dst_ids + [self.dst_tokenizer.end_id]
        dst_pad_size = self.dst_seq_len - len(dst_ids)
        dst_ids += [self.dst_tokenizer.pad_id] * dst_pad_size
        dst_targets = np.roll(dst_ids, -1).tolist()
        dst_mask = np.bitwise_and(
            np.array(dst_ids) != self.dst_tokenizer.pad_id,
            np.array(dst_ids) != self.dst_tokenizer.end_id,
        )
        attn_mask = np.tile(attn_mask.reshape(1, -1), [len(dst_ids), 1]).tolist()
        
        src_ids = torch.as_tensor(src_ids)
        dst_ids = torch.as_tensor(dst_ids)
        dst_targets = torch.as_tensor(dst_targets)
        dst_mask = torch.as_tensor(dst_mask)
        attn_mask = torch.as_tensor(attn_mask)
        enc = {
            'src': src_ids,
            'dst': dst_ids,
            'targets': dst_targets,
            'targets_masks': dst_mask,
            'attn_masks': attn_mask
        }
        return enc

    def infer(self, text):
        src_ids = self.src_tokenizer.encode(text)
        src_ids = src_ids[:self.src_seq_len_real]
        src_ids = [self.src_tokenizer.start_id] + src_ids + [self.src_tokenizer.end_id]
        src_pad_size = self.src_seq_len - len(src_ids)
        src_ids += [self.src_tokenizer.pad_id] * src_pad_size
        attn_mask = np.array(src_ids) == self.src_tokenizer.pad_id
