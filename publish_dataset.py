from datasets import Dataset
import pandas as pd

new_data = pd.read_csv("./synthesis_qa_output_12_21.csv")
new_dataset = Dataset.from_pandas(new_data)
new_dataset.push_to_hub("JianLiao/spectrum-design-docs")