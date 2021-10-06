import os
from tqdm import tqdm
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from joblib import dump, parallel_backend
import gc

for machine_type in os.listdir("emb"):

    # data = np.empty((0, 1280), float)
    data = np.empty((0, 128), float)
    counter = 0
    print("{} data generating:".format(machine_type))
    for emb_name in tqdm(os.listdir(os.path.join("emb", machine_type))):
        emb = np.load(os.path.join("emb", machine_type, emb_name))
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
    dump(lof, "lof/{}.joblib".format(machine_type))

    del data
    del lof
    gc.collect()
