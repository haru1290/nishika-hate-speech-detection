import pandas as pd


def main():
    k = 20
    preds = []
    for i in range(k):
        preds.append(pd.read_csv(f"./data/outputs/sub_{str(i)}.csv"))

    total = preds[0]["label"]
    for i in range(1, k):
        total += preds[i]["label"]

    preds[0]["label"] = round(total/k).astype(int)
    preds[0].to_csv("./data/outputs/sub.csv", index=None)


if __name__ == "__main__":
    main()