# Nishika - Hate Speech Detection

LB: 0.763（2nd）, PB: 0.750（5th）

## Directory Structure

```
./nishika-hate-speech-detection
├── LICENCE
├── Makefile
├── README.md
├── config
│   └── config.yaml
├── data
│   ├── final
│   ├── processed
│   └── raw
│       ├── data_explanation.xlsx
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── docs
├── environment.yml
├── models
├── notebooks
├── reports
│   └── figures
└── src
    ├── __init__.py
    ├── data
    │   └── make_dataset.py
    ├── features
    │   └── build_features.py
    └── models
    　   ├── predict_model.py
    　   └── train_model.py
```

## Requirements

- Anaconda 1.7.2
- Python 3.8.5
- Pytorch 1.13.0

## Usage

```
# 仮想環境の構築
```

## 12/13までにやること

- Argparseでconfigを指定，ハンドラの追加
- predictionを作成
- generate_labelをbuild_features.pyに追加
- リファクタリング
- READMEを書く
