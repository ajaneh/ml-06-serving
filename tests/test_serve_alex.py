from unittest.mock import patch

from fastapi.testclient import TestClient

from mlstudio.serve_alex import app

client = TestClient(app)


def test_predict_returns_200_for_valid_payload() -> None:
    response = client.post(
        "/predict",
        json={
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
        },
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
    assert "probabilities" in response.json()


def test_predict_returns_422_for_missing_or_invalid_features() -> None:
    response = client.post(
        "/predict",
        json={
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
        },
    )

    assert response.status_code == 422
    assert "Missing required feature" in response.json()["detail"]

    response = client.post(
        "/predict",
        json={
            "bill_length_mm": 39.1,
            "bill_depth_mm": "not-a-number",
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
        },
    )

    assert response.status_code == 422
    assert "must be numeric" in response.json()["detail"]


def test_predict_returns_500_for_unexpected_errors() -> None:
    with patch("mlstudio.serve_alex.MODEL.predict", side_effect=RuntimeError("boom")):
        response = client.post(
            "/predict",
            json={
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
