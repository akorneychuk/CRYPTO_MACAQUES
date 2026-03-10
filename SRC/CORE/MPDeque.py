from multiprocessing import Manager

class MPDeque:
    def __init__(self, maxlen):
        self.manager = Manager()
        self.data = self.manager.list()
        self.maxlen = maxlen

    def append(self, item):
        self.data.append(item)
        if len(self.data) > self.maxlen:
            del self.data[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(list(self.data))
