import torch
import numpy as np


class LossEvaluator:

    def __init__(
        self,
        dataset,
        model,
        loss_fn,
        device='cpu',
        batch_size=4,
        workers=0,
        **kwargs
    ):
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.workers = workers

        self.dataloader = None

    def setup(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.workers
        )

    def run(self):
        losses = []
        with torch.no_grad():
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop('targets')
                targets_masks = batch.pop('targets_masks')
                outputs = self.model(**batch)
                loss = self.loss_fn(outputs, targets, targets_masks)
                losses.append(loss.item())
        score = float(np.mean(losses))
        metrics = {'loss': score}
        return score, metrics
