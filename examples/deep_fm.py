'''

@Author : fushuai

@Email : fushuai@qutoutiao.net

@IDE : PyCharm

@Time : 2020/8/6 10:15

@Desc :

'''
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

if __name__ == '__main__':
    data = pd.read_csv("criteo_sample.txt")
    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)
    target = ['label']
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # unique()是以 数组形式（numpy.ndarray）返回列的所有唯一值（特征的所有唯一值）
    # nunique() Return number of unique elements in the object.即返回的是唯一值的个数

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for i, feat in
                              enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_name = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_name}
    test_model_input = {name: test[name] for name in feature_name}
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy"])
    history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print(pred_ans)
    print("Test LogLoss:", round(log_loss(test[target], pred_ans), 4))
    print("test AUC:", round(roc_auc_score(test[target], pred_ans), 4))
