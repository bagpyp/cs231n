import numpy as np
from tqdm import tqdm

from cifar import get_data


class NearestNeighbor:
    def __init__(self, Xtr, ytr):
        self.Xtr = Xtr  # (50_000 x 3_072) 50_000 images, each image is a point in N^3072
        self.ytr = ytr  # (50_000 x 1) An integer label for each image point [0,1,...,10]

    def predict(self, Xtest):
        num_images = Xtest.shape[0]  # 10_000 number of images in my test dataset
        y_pred = np.zeros(num_images, dtype=self.ytr.dtype)  # array of zeros, one for each test image
        for i in tqdm(range(num_images)):
            distances = np.sum(np.abs(self.Xtr - Xtest[i]), axis=1)
            min_index = np.argmin(distances)
            y_pred[i] = self.ytr[min_index]
        return y_pred


if __name__ == "__main__":
    train_data, test_data = get_data()
    Xtr = np.array(train_data['data'])
    ytr = np.array(train_data['labels'])
    Xtest = np.array(test_data['data'])
    ytest = np.array(test_data['labels'])

    nn = NearestNeighbor(Xtr, ytr)
    y_pred = nn.predict(Xtest)
    print(np.sum(np.abs(ytest-y_pred)))
