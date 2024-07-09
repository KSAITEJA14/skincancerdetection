import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_engineering import feature_engineering


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

if __name__ == '__main__':
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent
    
    train_metadata_path = home_dir.as_posix() + '/data/raw/isic-2024-challenge/train-metadata.csv'
    test_metadata_path = home_dir.as_posix() + '/data/raw/isic-2024-challenge/test-metadata.csv'
    
    train_metadata = load_data(train_metadata_path)
    test_metadata = load_data(test_metadata_path)
    
    train_interim = feature_engineering(pd.DataFrame(train_metadata))
    numeric_columns = train_interim.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove("target")

    scaler = StandardScaler()
    scaler.fit(train_interim[numeric_columns])

    train_interim[numeric_columns] = scaler.transform(train_interim[numeric_columns])
    test_interim = feature_engineering(test_metadata)
    test_interim[numeric_columns] = scaler.transform(test_interim[numeric_columns])
    
    output_path = home_dir.as_posix() + '/data/processed'
    save_data(pd.DataFrame(train_interim), pd.DataFrame(test_interim), output_path)
    
    del train_metadata, test_metadata, train_interim, test_interim, numeric_columns, scaler
    