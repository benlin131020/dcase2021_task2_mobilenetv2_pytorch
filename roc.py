import numpy as np
from sklearn import metrics
import os
import pandas as pd
import sys

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

y = []
scores = []
cnt = 0
prec_list = []
recall_list = []
f1_list = []
for f in sorted(os.listdir("result_lof")):
    if "score" in f:
        machine_type = f.split('_')[2]
        
        csv = open(os.path.join("result_lof", f))
        for line in csv:
            if "normal" in line.split(',')[0]:
                y.append(0)
            elif "anomaly" in line.split(',')[0]:
                y.append(1)
            else:
                print(line)
                print("error")
                break
        
            scores.append(float(line.split(',')[1]))
            print(line.split(',')[1])
            
        cnt += 1
        if cnt == 6:
            print(cnt)
            fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
            # print(len(thresholds))
            optimal_idx = np.argmax(tpr - fpr)
            decision_threshold = thresholds[optimal_idx]
            print(decision_threshold)
            tn, fp, fn, tp = metrics.confusion_matrix(y, [1 if x > decision_threshold else 0 for x in scores]).ravel()
            prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
            recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
            f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
            prec_list.append(prec)
            recall_list.append(recall)
            f1_list.append(f1)
            # print(decision_threshold)
            # print(prec, recall, f1)

            y = []
            scores = []
            cnt = 0

print(np.mean(np.array(prec_list, dtype=float), axis=0))
print(np.mean(np.array(recall_list, dtype=float), axis=0))
print(np.mean(np.array(f1_list, dtype=float), axis=0))