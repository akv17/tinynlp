import os
import json
import time

import torch
import numpy as np


class Trainer:

    def __init__(
        self,
        logger,
        dataset,
        model,
        loss_fn,
        path,
        device='cpu',
        batch_size=1,
        epochs=1,
        workers=0,
        prefetch=2,
        eval_every=1000,
        eval_params=None,
        evaluator=None,
        checkpoint=None,
        lr=0.001,
    ):
        self.logger = logger
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.path = path
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.prefetch = prefetch
        self.eval_every = eval_every
        self.eval_params = eval_params or {}
        self.evaluator = evaluator
        self.checkpoint = checkpoint
        self.lr = lr

        self.dataloader = None
        self.state = None
        self.optimizer = None

    def setup(self):
        if self.checkpoint is not None:
            self.logger.info(f'Loading model checkpoint: {self.checkpoint}')
            weights = torch.load(self.checkpoint, map_location='cpu')
            self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            prefetch_factor=self.prefetch
        )
        self.logger.info(f'Learning rate: {self.lr}')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.evaluator.setup()
        self.state = {}
        os.makedirs(self.path, exist_ok=True)

    def run(self):
        self.setup()
        step = 0
        losses = []
        for epoch in range(self.epochs):
            epoch += 1
            self.state['epoch'] = epoch
            st = time.time()
            for batch in self.dataloader:
                step += 1
                self.state['step'] = step
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop('targets')
                targets_masks = batch.pop('targets_masks')
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = self.loss_fn(outputs, targets, targets_masks)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                self.state['losses'] = losses
                rt = time.time() - st
                self.state['rt'] = round(rt, 2)
                st = time.time()
                self.on_step_end()

    def on_step_end(self):
        loss = np.mean(self.state['losses'])
        msg = f'[step] epoch: {self.state["epoch"]} step: {self.state["step"]} loss: {loss} time: {self.state["rt"]}'
        self.logger.info(msg)
        if self.state['step'] % self.eval_every != 0:
            return
        self.model.eval()
        self.on_eval()
        self.model.train()

    def on_eval(self):
        self.logger.info(f'[evaluation] ...')
        score, report = self.evaluator.run()
        self.logger.info(f'[score]: {score}')
        self.logger.info(f'[report]:')
        print(report)
        if 'best_score' not in self.state or score <= self.state['best_score']:
            self.state['best_score'] = float(score)
            self.state['best_report'] = report
            model_dst = os.path.join(self.path, 'model.pth')
            torch.save(self.model.state_dict(), model_dst)
            # report_fp = os.path.join(self.path, 'report.txt')
            # with open(report_fp, 'w', encoding='utf-8') as f:
            #     f.write(report)
            state_dst = os.path.join(self.path, 'state.json')
            state = self.state.copy()
            state.pop('losses')
            state.pop('best_report', None)
            with open(state_dst, 'w') as f:
                json.dump(state, f)
            msg = f'[checkpoint]: {score}'
            self.logger.info(msg)
        model_dst = os.path.join(self.path, 'model_last.pth')
        torch.save(self.model.state_dict(), model_dst)
