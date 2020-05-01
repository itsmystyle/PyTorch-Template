class MultipleOptimizer(object):
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
