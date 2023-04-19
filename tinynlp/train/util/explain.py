import os
import random


class Explain:
    
    def __init__(self):
        self.value = os.getenv('EXPLAIN', '-1').strip()
        self.flag = self.value != '-1'
    
    def __bool__(self):
        return self.flag
    
    def maybe(self, dataset):
        if not self:
            return
        ix = random.randint(0, len(dataset) - 1) if self.value == 'rand' else int(self.value)
        print(ix)
        dataset.explain(ix)
        exit()


EXPLAIN = Explain()
