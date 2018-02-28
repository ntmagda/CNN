import pickle
import numpy as np
import os
database_path = "lwf"

meta = {"Persons" : os.listdir(database_path)}

weights = {}
weights['layer_0'] = {}
weights['layer_1'] = {}
weights['layer_2'] = {}
weights['layer_3'] = {}
weights['layer_4'] = {}
weights['layer_5'] = {}
weights['layer_6'] = {}
weights['layer_7'] = {}
weights['layer_8'] = {}
weights['layer_9'] = {}
weights['layer_10'] = {}



weights['layer_0']['param_1'] = (2 * np.random.rand(32) - 1)/100
weights['layer_0']['param_0'] = (2 * np.random.rand(32, 1, 3, 3) - 1)/100

weights['layer_2']['param_1'] = (2 * np.random.rand(32) - 1)/100
weights['layer_2']['param_0'] = (2 * np.random.rand(32, 32, 3, 3) - 1)/100


weights['layer_7']['param_1'] = (2* np.random.rand(6000) - 1)/100
weights['layer_7']['param_0'] = (2* np.random.rand(38*38*32, 6000) - 1)/100

weights['layer_10']['param_1'] = (2 * np.random.rand(4835) - 1)/100
weights['layer_10']['param_0'] = (2 * np.random.rand(6000, 4835) - 1)/100

ppl = meta["Persons"]
a = np.zeros((len(ppl), len(ppl)), float)
np.fill_diagonal(a, 1.0)
expected = {i:x for i,x in zip(ppl, a)}

pickle.dump( [meta, weights, expected], open("params.pkl", "wb"))


