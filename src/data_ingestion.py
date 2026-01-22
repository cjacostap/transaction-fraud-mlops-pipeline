
import pandas as pd

class DataIngestion:
     
    def load_files(self)-> pd.DataFrame:
        '''
        Load training and testing datasets from local CSV files.
        '''

        df_data = pd.read_csv(
            "../data/raw/onlinefraud.csv"
        )

        return df_data
