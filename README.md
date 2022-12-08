# Nishika - Hate Speech Detection

## Project Structure

```
./nishika-hate-speech-detection
├── LICENCE
├── Makefile
├── README.md
├── config
│   ├── main.yaml
│   └── settings.json
├── data
│   ├── external
│   ├── interim
│   │   └── soft_label.npy
│   ├── processed
│   │   └── sub.csv
│   └── raw
│       ├── data_explanation.xlsx
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── docs
├── environment.yml
├── models
├── notebooks
├── references
├── reports
│   └── figures
└── src
    ├── __init__.py
    ├── data
    │   └── make_dataset.py
    ├── features
    │   └── build_features.py
    ├── models
    │   ├── predict_model.py
    │   └── train_model.py
    └── visualization
        └── visualize.py
```

## 12/10までにやること

- predictionを作成
- generate_labelをbuild_features.pyに追加
- ハンドラの追加
- READMEを書く
- 公開
