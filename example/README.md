An example usage of DeepFM/FM/DNN models for [Porto Seguro's Safe Driver Prediction competition on Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction).

Please download the data from the above website and put them into the proper directory (see `config.py`).

To train DeepFM model for this dataset, run

```
$ python main.py
```

# Some tips
- [ ] You can compare DeepFM and FM (or DNN) to see what works best for this dataset.
- [ ] You should tune the parameters for the models in order to get reasonable performance (gini around 0.28).
