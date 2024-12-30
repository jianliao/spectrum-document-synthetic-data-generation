from datasets import Dataset
import pandas as pd

new_data = pd.read_csv("./synthetic_data_12_29.csv")
new_dataset = Dataset.from_pandas(new_data)
new_dataset.push_to_hub("JianLiao/spectrum-design-docs")