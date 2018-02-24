import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
fn1 = "params.pkl"

class FaceRecognition:

    def __init__(self):
        [meta, weights, expected] = pickle.load(open(fn1, 'rb'), encoding='latin1')
        self.CNN = CNN()
        self.CNN.load_weights(weights)

        self.CNN.load_expected(expected)

        self.persons = meta["Persons"]

    def predict(self, X):
        predicted_i = self.CNN.predict(X)
        return self.persons[predicted_i]

    def train(self, X, expected_person):
        self.CNN.back_propagation(X, expected_person)



class CNN():
    def __init__(self):
        self.layers = []
        self.expected = {}
        self.pool_size = 2

    def load_weights(self, weights):
        assert not self.layers, "Weights can only be loaded once!"
        for k in range(len(weights.keys())):
            self.layers.append(weights['layer_{}'.format(k)])

    def load_expected(self, _expected):
        assert not self.expected, "Expected values can be load once!"
        self.expected = _expected


    def error(self, cnn_out, expected_person):
        y = self.expected[expected_person]
        return self.cross_entropy(cnn_out, y)

    def back_propagation(self, X, expected_person):
        cnn_out = self.forward_propagation(X)
        cnn_error = self.error(cnn_out, expected_person)
        # L_delta = cnn_error * self.relu_layer(cnn_out, deriv=True)
        print(cnn_error)
        # print("Error:" + str(np.mean(np.abs(cnn_error))))

    def forward_propagation(self, X):
        print(X.shape)
        h = self.cnn_layer(X)
        X = h
        print(X.shape)
        h = self.relu_layer(X)
        X = h
        h = self.cnn_layer(X, layer_i=2, border_mode="valid")
        X = h
        print(X.shape)
        h = self.relu_layer(X)
        X = h
        h = self.maxpooling_layer(X)
        X = h
        print(X.shape)
        h = self.flatten_layer(X)
        X = h
        print("flat")
        print(X.shape)
        print(np.min(X))
        print(np.max(X))
        h = self.dense_layer(X, layer_i=7)
        X = h # have to prepare better weight dimenstions for that
        h = self.relu_layer(X)
        X = h
        # h = self.dropout_layer(X, .5)
        # X = h
        h = self.dense_layer(X, layer_i=10)
        X = h
        # print(X.shape)
        h = self.normalize(X, -10, 10)
        X = h
        h = self.softmax_layer2D(X)
        X = h
        plt.plot(X)
        plt.show()
        return X

    def predict(self, X):
        h = self.forward_propagation(X)
        X = h
        max_i = self.classify(X)
        return max_i

    def cnn_layer(self, X, layer_i=0, border_mode="full"):
        features = self.layers[layer_i]["param_0"]
        bias = self.layers[layer_i]["param_1"]
        patch_dim = features[0].shape[-1]  # ???
        nb_features = features.shape[0]
        image_dim = X.shape[1]  # assume image square

        image_channels = X.shape[0]
        if border_mode == "full":
            conv_dim = image_dim + patch_dim - 1
        elif border_mode == "valid":
            conv_dim = image_dim - patch_dim + 1
        convolved_features = np.zeros((nb_features, conv_dim, conv_dim))
        for feature_i in range(nb_features):
            convolved_image = np.zeros((conv_dim, conv_dim))
            for channel in range(image_channels):
                feature = features[feature_i, 0, :, :]
                image = X[channel, :, :]
                # print("XXX")
                # print(X.shape)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
                convolved_image += self.convolve2d(image, feature, border_mode)
            convolved_image = convolved_image/image_channels + bias[feature_i]
            # cv2.imshow('image', convolved_image)
            # cv2.waitKey(0)
            convolved_features[feature_i, :, :] = convolved_image
        return convolved_features

    def dense_layer(self, X, layer_i=0):
        W = self.layers[layer_i]["param_0"]
        b = self.layers[layer_i]["param_1"]
        output = np.dot(X, W) + b
        return output


    @staticmethod
    def normalize(x, r_1, r_2):
        m = min(x)
        range = max(x) - m
        x = (x - m) / range

        range2 = r_2 - r_1
        return (x * range2) + r_1

    @staticmethod
    def convolve2d(image, feature, border_mode="full"):
        image_dim = np.array(image.shape)
        # print("image_dim")
        # print(image_dim)
        feature_dim = np.array(feature.shape)
        # print("feature_dim")
        # print(feature_dim)
        # print("image_dim")
        # print(image_dim)
        target_dim = image_dim + feature_dim - 1
        fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
        target = np.fft.ifft2(fft_result).real


        if border_mode == "valid":
            # To compute a valid shape, either np.all(x_shape >= y_shape) or
            # np.all(y_shape >= x_shape).
            valid_dim = image_dim - feature_dim + 1
            if np.any(valid_dim < 1):
                valid_dim = feature_dim - image_dim + 1
            start_i = (target_dim - valid_dim) // 2
            end_i = start_i + valid_dim
            target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]
        return target

    def maxpooling_layer(self, convolved_features):
        nb_features = convolved_features.shape[0]
        conv_dim = convolved_features.shape[1]
        res_dim = int(conv_dim / self.pool_size)  # assumed square shape

        pooled_features = np.zeros((nb_features, res_dim, res_dim))
        for feature_i in range(nb_features):
            for pool_row in range(res_dim):
                row_start = pool_row * self.pool_size
                row_end = row_start + self.pool_size

                for pool_col in range(res_dim):
                    col_start = pool_col * self.pool_size
                    col_end = col_start + self.pool_size

                    patch = convolved_features[feature_i, row_start: row_end, col_start: col_end]
                    pooled_features[feature_i, pool_row, pool_col] = np.max(patch)
        return pooled_features

    @staticmethod
    def cross_entropy(X, y, deriv = False):
        if deriv == True:
            return X - y
        y = y.astype(int)
        m = y.shape[0]
        log_likelihood = []
        for i in range(m):
            log = -np.log(X[i])*y[i]
            log_likelihood.append(log)
        loss = np.sum(log_likelihood) / m
        return loss


    @staticmethod
    def relu_layer(x, deriv = False):
        z = np.zeros_like(x)
        if deriv == True:
            x[x > 0] = 1
            x[x <= 0] = 0
            return x
        return np.where(x > z, x, z)

    @staticmethod
    def softmax_layer2D(w, deriv = False):
        e = np.exp(w)
        softmax = e / np.sum(e, axis=0, keepdims=True)
        if deriv == True:
            return (softmax * (1 - softmax))
        return softmax

    @staticmethod
    def repeat_vector(X, n):
        y = np.ones((X.shape[0], n, X.shape[2])) * X
        return y

    @staticmethod
    def dropout_layer(X, p):
        retain_prob = 1. - p
        X *= retain_prob
        return X

    @staticmethod
    def classify(X):
        return np.argmax(X)

    @staticmethod
    def flatten_layer(X):
        flatX = X.flatten(order='C')
        return flatX



x = cv2.imread("test_image.jpg")
x_swap = np.einsum('kli->ilk', x)
fr = FaceRecognition()
# print(fr.predict(np.asarray(x_swap)))
print(fr.train(np.asarray(x_swap), 'Aaron_Eckhart'))