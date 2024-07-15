import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from keras.models import load_model
from .const import class_names

#plot confusion matrix
def plt_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_mark = np.arange(len(classes))
    plt.xticks(tick_mark, classes, rotation=45)
    plt.yticks(tick_mark, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.axis]
        print("normalized confusion matrix")

    else:
        print("confusion matrix without normalization")

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel("predicted label")
        plt.ylabel("True label")



def get_test_data_class(test_path):
    names = []
    for i in test_path:
        name = i.split("/")[-2]
        name_idx = class_names.index(name)
        names.append(name_idx)
    names = np.array(names, dtype=np.int32)
    return names


def batch_prediction(model_path:str, test_df, test):
    cm = load_model(model_path)
    prediction = cm.predict(test_df, verbose=0)
    np.around(prediction)
    y_pred_classes = np.argmax(prediction, axis=1)

    classes = get_test_data_class(test)

    cm = confusion_matrix(y_true=classes, y_pred=y_pred_classes)
    plt_confusion_matrix(cm=cm, classes=class_names, title="confusion matrix", )

    print(classification_report(y_true=classes, y_pred=y_pred_classes))
