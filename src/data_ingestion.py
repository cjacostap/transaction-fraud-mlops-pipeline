
import pandas as pd
from typing import Tuple

class DataIngestion:
     
    def load_files(self)-> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Load training and testing datasets from local CSV files.
        '''

        df_training = pd.read_csv(
            "../data/raw/customer_churn_dataset-training-master.csv"
        )
        df_testing = pd.read_csv(
            "../data/raw/customer_churn_dataset-testing-master.csv"
        )

        return df_training, df_testing
