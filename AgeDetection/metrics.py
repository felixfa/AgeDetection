import numpy as np
import pandas as pd
from math import modf

def weighted_accuracy(y_pred):
    weighted_pred = 0
    for i, element in enumerate(y_pred):
        num_1 = int(np.argsort(y_pred[i])[-1])
        num_2 = int(np.argsort(y_pred[i])[-2])
        num_3 = int(np.argsort(y_pred[i])[-3])
        prop_1 = y_pred[i][num_1]
        prop_2 = y_pred[i][num_2]
        prop_3 = y_pred[i][num_3]
        prop_sum = prop_1+prop_2+prop_3
        pred = num_1*prop_1/prop_sum+num_2*prop_2/prop_sum+num_3*prop_3/prop_sum

        weighted_pred = round(modf(pred)[1]*5+1 + modf(pred)[0] * 5, 2)
        # pred=int(round(pred,0))
        # pred_list.append(pred)
    # return np.array(pred_list)
    return weighted_pred


def one_off_accuracy(y_test, y_pred):

    try:
        # Categorical y_test
        i_ind = []
        accurate = 0
        size = y_test.shape[1]

        hit_miss =\
            pd.DataFrame(data=0, dtype="int16",
                         index=np.linspace(0, size-1,
                                           size, dtype="int8"),
                         columns=['Hit', 'Miss'])

        conf =\
            pd.DataFrame(data=0, dtype="int16",
                         index=np.linspace(0, size-1,
                                           size, dtype="int8"),
                         columns=np.linspace(0, size-1,
                                             size, dtype="int8"))

        for i in range(len(y_pred)):

            # 1-Off Accuracy
            i_ind = [y_pred[i].argmax()-1,
                     y_pred[i].argmax(),
                     y_pred[i].argmax()+1]
            if y_test[i].argmax() in i_ind:
                accurate += 1

            # Hit/Miss Matrix
                hit_miss.at[y_test[i].argmax(), "Hit"] += 1
            else:
                hit_miss.at[y_test[i].argmax(), "Miss"] += 1

            # Confusion Matrix
            conf.at[y_test[i].argmax(), y_pred[i].argmax()] += 1

        conf["Total"] = np.sum(y_test, axis=0, dtype="int16")
        for k in range(0, y_test.shape[1]):
            if (conf.iloc[k, k] != 0) and (conf.at[k, "Total"] != 0):
                conf.at[k, "Hit Rate"] =\
                    round(100*np.divide(conf.at[k, k], conf.at[k, "Total"]), 1)
        print(f"The 1-Off Accuracy is {round(100*accurate/len(y_test), 2)}%")

    except IndexError:
        # Non-categorical y_test -> for sparse y's
        i_ind = []
        accurate = 0
        size = len(np.unique(y_test))

        hit_miss =\
            pd.DataFrame(data=0, dtype="int16",
                         index=np.linspace(0, size-1,
                                           size, dtype="int8"),
                         columns=['Hit', 'Miss'])

        conf =\
            pd.DataFrame(data=0, dtype="int16",
                         index=np.linspace(0, size-1,
                                           size, dtype="int8"),
                         columns=np.linspace(0, size-1,
                                             size, dtype="int8"))

        for i in range(len(y_pred)):

            # 1-Off Accuracy
            i_ind = [y_pred[i].argmax()-1,
                     y_pred[i].argmax(),
                     y_pred[i].argmax()+1]
            if y_test[i] in i_ind:
                accurate += 1

            # Hit/Miss Matrix
                hit_miss.at[y_test[i], "Hit"] += 1
            else:
                hit_miss.at[y_test[i], "Miss"] += 1

            # Confusion Matrix
            conf.at[y_test[i], y_pred[i].argmax()] += 1

        conf["Total"] = np.sum(y_test, axis=0, dtype="int16")
        for k in range(0, len(np.unique(y_test))):
            if (conf.iloc[k, k] != 0) and (conf.at[k, "Total"] != 0):
                conf.at[k, "Hit Rate"] =\
                    round(100*np.divide(conf.at[k, k], conf.at[k, "Total"]), 1)
        print(f"The 1-Off Accuracy is {round(100*accurate/len(y_test), 2)}%")

    return hit_miss, conf
