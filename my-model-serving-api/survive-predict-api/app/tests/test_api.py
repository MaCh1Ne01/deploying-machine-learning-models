import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score


def test_make_prediction(
    client: TestClient, X_test_data: pd.DataFrame, y_test_data: pd.DataFrame
) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": X_test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"]
    assert prediction_data["errors"] is None
    _predictions = list(prediction_data["predictions"])
    assert len(_predictions) == 262
    accuracy = accuracy_score(_predictions, y_test_data)
    assert accuracy > 0.6
