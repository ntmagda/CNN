import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
fn1 = "params.pkl"
from scipy import signal


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
        for i in range(20):
            self.CNN.back_propagation(X, expected_person)
            self.CNN.weights_update(0.2)


class CNN():
    def __init__(self):
        self.layer_outputs = {}
        self.layers = []
        self.layers_deltas = {}
        self.expected = {}
        self.pool_size = 2

    def load_weights(self, weights):
        assert not self.layers, "Weights can only be loaded once!"
        for k in range(len(weights.keys())):
            self.layers.append(weights['layer_{}'.format(k)])

    def load_expected(self, _expected):
        assert not self.expected, "Expected values can be load once!"
        self.expected = _expected


    def cross_entropy_error(self, cnn_out, expected_person, func):
        # TODO - function passed - there are two versions of cross entropy - which one should I use?
        y = self.expected[expected_person]
        return func(cnn_out, y)

    def back_propagation(self, X, expected_person):
        nb_features = self.layers[0]["param_0"].shape[0]
        nb_channels = 3  # TODO _ not hardcode

        y = self.expected[expected_person]
        cnn_out = self.forward_propagation(X)

        print("ERROR: ")

        cross_entropy_error = self.cross_entropy_error(cnn_out, expected_person, self.cross_entropy)
        print(cross_entropy_error)
        layer_10_error = self.cross_entropy_softmax_delta(cnn_out, y) # predicted - expected
        layer_10_delta = np.dot(np.asmatrix(self.layer_outputs["layer_8"]).T, np.asmatrix(layer_10_error))
        #x.T *(y-y')
        relu7_deriv = self.relu_layer(self.layer_outputs["layer_7"], deriv=True)
        layer_7_error = np.asarray(np.dot(layer_10_error, np.asmatrix(self.layers[10]['param_0']).T)) * relu7_deriv
        layer_7_delta = np.dot(np.asmatrix(self.layer_outputs["layer_5"]).T, layer_7_error)

        # linear backpropagating just changing shapes
        flatten_error = np.asarray(layer_7_error.dot(self.layers[7]["param_0"].T)[0,:]).reshape((32, 38, 38))
        # przypisanie wartosci bledu na miejsce w ktorych zostala pobrana wartosc do max poolingu
        (_, maxpool_full_dim) = self.layer_outputs['layer_4']
        maxpool_error = self.maxpool_back(flatten_error, maxpool_full_dim)

        relu2_deriv = self.relu_layer(self.layer_outputs["layer_2"], deriv=True)
        layer_2_error = maxpool_error * relu2_deriv


        layer_2_delta = np.zeros((nb_features, 3, 3))
        for feature_i in range(nb_features):
            # TODO - flip the error martix or not - to be figured out
            layer_2_delta_i = self.convolve2d(np.asmatrix(self.layer_outputs["layer_1"][feature_i ,:,:]).T,
                                              layer_2_error[feature_i ,:,:], border_mode='valid')
            layer_2_delta[feature_i] = layer_2_delta_i

        relu0_deriv = self.relu_layer(self.layer_outputs["layer_0"], deriv=True)
        layer_0_error = np.zeros((nb_features, 78, 78))
        for feature_i in range(nb_features):
            feature = np.asmatrix(self.layers[2]['param_0'][feature_i, :, :]).T
            feature_rot = self.rot180(feature)
            layer_0_error[feature_i] = self.convolve2d(layer_2_error[feature_i, :, :], feature_rot, border_mode="full")
        layer_0_error = layer_0_error * relu0_deriv

        layer_0_delta = np.zeros((nb_features, nb_channels, 3, 3))
        for feature_i in range(nb_features):
            for channel_i in range(nb_channels):
                layer_0_delta_i = self.convolve2d(np.asmatrix(self.layer_outputs["input_image"][channel_i, :, :]).T,
                                                  layer_0_error[feature_i, :, :], border_mode='valid')
                layer_0_delta[feature_i][channel_i] = layer_0_delta_i

        # for now I do not tak einto consideration channels so:
        layer_0_delta = layer_0_delta[:, 0, :, :]
        self.layers_deltas["layer_10"] = layer_10_delta
        self.layers_deltas["layer_7"] = layer_7_delta
        self.layers_deltas["layer_2"] = layer_2_delta
        self.layers_deltas["layer_0"] = layer_0_delta

    def weights_update(self, learning_rate):
        # self.layers[0]["param_0"] -= learning_rate * self.layers_deltas["layer_0"]
        # self.layers[2]["param_0"] -= learning_rate * self.layers_deltas["layer_2"]
        self.layers[7]["param_0"] -= learning_rate * self.layers_deltas["layer_7"]
        self.layers[10]["param_0"] -= learning_rate * self.layers_deltas["layer_10"]




    def forward_propagation(self, X):
        self.layer_outputs["input_image"] = X
        h = self.cnn_layer(X)
        self.layer_outputs["layer_0"] = h

        h = self.relu_layer(h)
        self.layer_outputs["layer_1"] = h

        h = self.cnn_layer(h, layer_i=2, border_mode="valid") # backpropagated
        self.layer_outputs["layer_2"] = h

        h = self.relu_layer(h)
        self.layer_outputs["layer_3"] = h

        (h, h_full_dim) = self.maxpooling_layer(h)
        self.layer_outputs["layer_4"] = (h, h_full_dim)

        h = self.flatten_layer(h)
        self.layer_outputs["layer_5"] = h

        h = self.dense_layer(h, layer_i=7)
        self.layer_outputs["layer_7"] = h #bacpropagated

        h = self.relu_layer(h)
        self.layer_outputs["layer_8"] = h

        # h = self.dropout_layer(X, .5)
        # X = h

        h = self.dense_layer(h, layer_i=10)
        self.layer_outputs["layer_10"] = h #backpropagated

        h = self.softmax_layer2D(h)
        X = h
        return X

    def predict(self, X):
        h = self.forward_propagation(X)
        X = h
        max_i = self.classify(X)
        return max_i

    def cnn_layer(self, X, layer_i=0, border_mode="full"):
        features = self.layers[layer_i]["param_0"]
        bias = self.layers[layer_i]["param_1"]
        patch_dim = features[0].shape[-1]
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
                feature = features[feature_i, :, :]
                image = X[channel, :, :]
                convolved_image += self.convolve2d(image, feature, border_mode)
            convolved_image = convolved_image/image_channels + bias[feature_i]
            convolved_features[feature_i, :, :] = convolved_image
        return convolved_features

    @staticmethod
    def convolve_backprop(x, y):
        return signal.convolve2d(x,y)


    def dense_layer(self, X, layer_i=0):
        W = self.layers[layer_i]["param_0"]
        b = self.layers[layer_i]["param_1"]
        output = np.dot(X, W) + b
        # plt.plot(output, '*')
        # plt.show()
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
        feature_dim = np.array(feature.shape)
        target_dim = image_dim + feature_dim - 1
        fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
        target = np.fft.ifft2(fft_result).real

        if border_mode == "valid":
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
        pooled_features_full_dim = np.zeros((nb_features, conv_dim, conv_dim)) # just for backpropagation
        for feature_i in range(nb_features):
            for pool_row in range(res_dim):
                row_start = pool_row * self.pool_size
                row_end = row_start + self.pool_size

                for pool_col in range(res_dim):
                    col_start = pool_col * self.pool_size
                    col_end = col_start + self.pool_size

                    patch = convolved_features[feature_i, row_start: row_end, col_start: col_end]
                    pooled_arg_max = np.argmax(patch)
                    # calculate index from original image that was taken further by maxpool layer
                    if pooled_arg_max == 0:
                        max_index = (row_start, col_start)
                    if pooled_arg_max == 1:
                        max_index = (row_start, col_start+1)
                    if pooled_arg_max == 2:
                        max_index = (row_start+1, col_start)
                    if pooled_arg_max == 3:
                        max_index = (row_start+1, col_start+1)
                    pooled_features[feature_i, pool_row, pool_col] = np.max(patch)
                    pooled_features_full_dim[feature_i, max_index[0], max_index[1]] = np.max(patch)
        return (pooled_features, pooled_features_full_dim)

    def maxpool_back(self, pooled_features, pooled_features_full_dim):
        nb_features = pooled_features_full_dim.shape[0]
        conv_dim = pooled_features_full_dim.shape[1]
        res_dim = int(conv_dim / self.pool_size)  # assumed square shape

        for feature_i in range(nb_features):
            for pool_row in range(res_dim):
                row_start = pool_row * self.pool_size
                row_end = row_start + self.pool_size

                for pool_col in range(res_dim):
                    col_start = pool_col * self.pool_size
                    col_end = col_start + self.pool_size

                    # patch = pooled_features_full_dim[feature_i, row_start: row_end, col_start: col_end]
                    max_patch = pooled_features[feature_i, pool_row, pool_col]
                    pooled_features_full_dim[feature_i, row_start: row_end, col_start: col_end][pooled_features_full_dim[feature_i, row_start: row_end, col_start: col_end]!= 0] = max_patch
        return pooled_features_full_dim

    @staticmethod
    def cross_entropy(X, y):
        y = y.astype(int)
        m = y.shape[0]
        log_likelihood = []
        for i in range(m):
            log = -np.log(X[i])*y[i]
            log_likelihood.append(log)
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def cross_entropy_softmax_delta(X, y):
        return X - y

    # @staticmethod
    # def cross_entropy_v2(X, y, deriv = False):
    #     if deriv == True:
    #         return -1 * (( y * (1/X)+ ( 1 - y)* (1/(1 - X))))
    #     y = y.astype(int)
    #     m = y.shape[0]
    #     log_likelihood = []
    #     for i in range(m):
    #         factor = - (y[i] * np.log(X[i]) + (1 - y[i])*np.log(1-X[i]))
    #         log_likelihood.append(factor)
    #     loss = np.sum(log_likelihood) / m
    #     return loss

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
        # max = np.max(w)
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

    @staticmethod
    def rot180(m):
        temp = np.rot90(m)
        return np.rot90(temp)



x = cv2.imread("test_image.jpg")
x_swap = np.einsum('kli->ilk', x)
fr = FaceRecognition()
# print(fr.predict(np.asarray(x_swap)))
fr.train(np.asarray(x_swap), 'Lena_Olin')