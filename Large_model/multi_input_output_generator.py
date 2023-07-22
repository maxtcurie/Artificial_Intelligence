class DataGenerator(keras.utils.Sequence):
    def __init__(self, x1, x2, y1, y2, batch_size=32, shuffle=True):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.y1))
        self.on_epoch_end()

    def __len__(self):
        return len(self.y1) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        x1_temp = self.x1[indexes]
        x2_temp = self.x2[indexes]
        y1_temp = self.y1[indexes]
        y2_temp = self.y2[indexes]

        return {'input1': x1_temp, 'input2': x2_temp}, {'output1': y1_temp, 'output2': y2_temp}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
