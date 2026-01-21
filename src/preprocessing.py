
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Any


class PreProcessing:

    def cleaning(self)-> Tuple[pd.DataFrame, pd.DataFrame]:

    # ----------------- Config paths -----------------
    INP_RAW  = "../../data/processed/df_nivel_cliente.csv"
    OUT_DIR  = Path("../reports/vulnerability"); OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CLEAN = "../../data/processed/vulnerabilidad/df_clean_for_vuln.csv"
    OUT_FEAT  = "../../data/processed/vulnerabilidad/df_features_final.csv"

    # Cargar el DataFrame desde un archivo CSV

    df_training = pd.read_parquet("../data/processed/train_raw.parquet")
    df_testing = pd.read_parquet("../data/processed/test_raw.parquet")

    # mostrar todas las columnas
    pd.set_option('display.max_columns', None)

    print(df_training.columns)
    print(df_training.shape)
    print(df_training.head())
    print(df_training.info())

    return df_training, df_testing







