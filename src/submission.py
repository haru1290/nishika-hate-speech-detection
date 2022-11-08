import pandas as pd


def main():
    preds = []
    for i in range(5):
        preds.append(pd.read_csv(f"./data/outputs/sub_{str(i)}.csv"))

    result = (preds[0]["label"] + preds[1]["label"] + preds[2]["label"] + preds[3]["label"] + preds[4]["label"]) / 5
    preds[0]["label"] = round(result).astype(int)

    preds[0].to_csv("./data/outputs/sub.csv", index=None)


if __name__ == "__main__":
    main()