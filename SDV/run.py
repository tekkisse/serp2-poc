import pandas as pd
import os
from sdv.metadata import SingleTableMetadata


directory_path = './data'
sample_size=10000

def process_parquet_file(file_path):
    print(f"Processing {file_path}")

    real_data = pd.read_parquet(file_path, engine='pyarrow').sample(n=sample_size)
    num_rows = len(real_data)
    print(f"Processing {file_path} - Number of Rows (NOW): {num_rows}")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    metadata
    real_data.head()

    print("read file")

    from sdv.lite import SingleTablePreset
    synthesizer = SingleTablePreset(metadata,name='FAST_ML')
    synthesizer.fit(data=real_data)
    synthesizer.save(f'output/{file_path}.pkl')

    print("model done")

    # use to generate some data
    synthetic_data = synthesizer.sample(num_rows=1500)
    synthetic_data.head()
    synthetic_data.to_csv(f'output/{file_path}.csv')

    print("sample created")


# Iterate over the files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".parquet"):
        file_path = os.path.join(directory_path, filename)
        process_parquet_file(file_path)