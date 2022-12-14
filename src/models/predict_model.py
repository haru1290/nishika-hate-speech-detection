# 予測

import hydra
import pandas as pd


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    df = pd.read_csv(cfg.path.train)
    print(df)


if __name__ == "__main__":
    main()