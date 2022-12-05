# Nishika - ヘイトスピーチ検出

## Project Structure

```
./
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
├── models
├── requirements.txt
└── src
    ├── __init__.py
    ├── make_dataset.py
    ├── predict_model.py
    └── train_model.py
```

## 12/4までにやること

- ハンドラの追加
- predictionを作成
- generate_labelの追加
- READMEを書く
- 公開
