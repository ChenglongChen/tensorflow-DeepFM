import tensorflow as tf
from sklearn.metrics import roc_auc_score
from DeepFM import DeepFM

# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 512*4,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2018
}
# prepare training and validation data in the required format
Xi_train, Xv_train, y_train = prepare(...)
Xi_valid, Xv_valid, y_valid = prepare(...)
# init a DeepFM model
dfm = DeepFM(feature_size, field_size, **dfm_params)
# fit a DeepFM model
dfm.fit(Xi_train, Xv_train, y_train)
# export model
dfm.export_model()
# make prediction
print dfm.predict(Xi_valid, Xv_valid)
# evaluate a trained model
print dfm.evaluate(Xi_valid, Xv_valid, y_valid)

 
