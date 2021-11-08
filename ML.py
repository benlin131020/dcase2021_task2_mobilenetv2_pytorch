import os
from tqdm import tqdm
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from joblib import dump, parallel_backend
import gc
import common as com

param = com.yaml_load()
im_path = os.path.join(param["exp_directory"], param["im_directory"])
if not os.path.exists(im_path):
    os.makedirs(im_path)

emb_path = os.path.join(param["exp_directory"], param["emb_directory"])
for machine_type in os.listdir(emb_path):

    # data = np.empty((0, 1280), float)
    data = np.empty((0, 128), float)
    counter = 0
    print("{} data generating:".format(machine_type))
    for emb_name in tqdm(os.listdir(os.path.join(emb_path, machine_type))):
        emb = np.load(os.path.join(emb_path, machine_type, emb_name))
        data = np.append(data, emb, axis=0)
        # if counter >= 500:
        #     break
        # counter += 1

    print(data.shape)
    
    # clf = OCSVM(verbose=True)
    # knn = KNN(n_jobs=-1)
    lof = LOF(n_neighbors=4, n_jobs=-1)

    print("fitting....")
    # knn.fit(data)
    # clf.fit(data)
    lof.fit(data)
    # dump(clf, "ocsvm/{}.joblib".format(machine_type))
    # dump(knn, "knn/{}.joblib".format(machine_type))
    dump(lof, os.path.join(im_path, "{}.joblib".format(machine_type)))
    # dump(lof, "{}/{}/{}.joblib".format(param["exp_directory"], param["im_directory"], machine_type))

    del data
    del lof
    gc.collect()
