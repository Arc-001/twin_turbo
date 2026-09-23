"""Published C-MAPSS results used for the benchmark comparison.

All numbers are RMSE on the official test sets (last cycle per engine, RUL
clipped at 125 or 130 depending on the paper), transcribed from Table III of
Ragab et al., "Attention Sequence to Sequence Model for Machine Remaining
Useful Life Prediction", arXiv:2007.09868, which compiles them from the
original papers cited per row.
"""

LITERATURE_RMSE = [
    # (method, citation key, FD001, FD002, FD003, FD004)
    ("Gradient boosting", "zhang2016modbne", 15.67, 29.09, 16.84, 29.01),
    ("2D CNN", "babu2016cnn", 18.45, 30.29, 19.82, 29.16),
    ("Deep LSTM", "zheng2017lstm", 16.14, 24.49, 16.81, 28.17),
    ("1D deep CNN", "li2018dcnn", 12.61, 22.36, 12.64, 23.31),
    ("MODBNE ensemble", "zhang2016modbne", 15.04, 25.05, 12.51, 28.66),
    ("BiLSTM encoder-decoder", "yu2019bilstmed", 14.74, 22.07, 17.48, 23.49),
    ("Hybrid DNN (HDNN)", "aldulaimi2019hdnn", 13.02, 15.24, 12.22, 18.16),
    ("Attention seq2seq (ATS2S)", "ragab2020ats2s", 12.63, 14.65, 11.44, 16.66),
]

LITERATURE_SOURCE = "ragab2020ats2s"
