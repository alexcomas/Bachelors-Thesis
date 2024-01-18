class MyDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.generator = loader.dataset.generator
        self.n = 0
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # print("Reset iterator.")
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def __len__(self):
        return len(self.loader)