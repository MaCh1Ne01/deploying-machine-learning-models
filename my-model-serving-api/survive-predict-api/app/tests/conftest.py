from typing import Generator, Tuple

import pandas as pd
import pytest
from classification_model.config.core import config
from classification_model.processing.data_manager import load_raw_dataset
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from app.main import app


def test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
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


@pytest.fixture(scope="module")
def X_test_data() -> pd.DataFrame:
    return test_data()[0]


@pytest.fixture(scope="module")
def y_test_data() -> pd.DataFrame:
    return test_data()[1]


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
