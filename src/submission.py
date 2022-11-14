import pandas as pd


def main():
    k = 26
    preds = []
    for i in range(k):
        if i == 5:
            continue
        preds.append(pd.read_csv(f"./data/submission/sub_{str(i)}.csv"))

    total = preds[0]["label"]
    for i in range(1, k - 1):
        total += preds[i]["label"]

    preds[0]["label"] = round(total/k).astype(int)
    preds[0].to_csv("./data/submission/sub.csv", index=None)


if __name__ == "__main__":
    main()