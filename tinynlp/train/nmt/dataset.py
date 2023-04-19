import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, encoder, display):
        self.samples = samples
        self.encoder = encoder
        self.display = display
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, ix):
        sample = self.samples[ix]
        enc = self.encoder.run(sample)
        return enc

    def explain(self, ix):
        enc = self[ix]
        self.display.run(enc)
