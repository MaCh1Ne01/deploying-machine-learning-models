import warnings

import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_raw_dataset


@pytest.fixture(autouse=True)
def suppress_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            message="is_categorical_dtype is deprecated")
    warnings.filterwarnings("ignore", category=FutureWarning,
                            message="Downcasting object dtype arrays on .fillna")


@pytest.fixture()
def sample_input_data():
    data = load_raw_dataset(file_name=config.app_config.raw_data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.model_config.target, axis=1),  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    return X_test, y_test
