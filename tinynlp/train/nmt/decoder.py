class Decoder:

    def __init__(self, src_tokenizer, dst_tokenizer):
        self.src_tokenizer = src_tokenizer
        self.dst_tokenizer = dst_tokenizer
    
    def decode_src(self, ids):
        text = self.src_tokenizer.decode(ids)
        return text
    
    def decode_dst(self, ids):
        text = self.dst_tokenizer.decode(ids)
        return text
