from __future__ import annotations

import numpy as np
import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.pilot.model import LocalQwenEncoder, normalize_embeddings_float32


def test_float32_normalization_repairs_bfloat16_scale_error() -> None:
    vectors = np.asarray(
        [
            [0.9972502, 0.0, 0.0],
            [0.0, 1.0028310, 0.0],
            [0.0, 0.0, -0.9998542],
        ],
        dtype=np.float32,
    )

    normalized = normalize_embeddings_float32(vectors)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(normalized.astype(np.float64), axis=1),
        np.ones(3),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "vectors",
    [
        np.asarray([[np.nan, 1.0]], dtype=np.float32),
        np.asarray([[np.inf, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([], dtype=np.float32),
    ],
)
def test_float32_normalization_rejects_invalid_embeddings(vectors: np.ndarray) -> None:
    with pytest.raises(ContractError):
        normalize_embeddings_float32(vectors)


def test_local_encoder_requests_raw_vectors_before_float32_normalization() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            self.kwargs = kwargs
            assert texts == ["synthetic"]
            return np.asarray([[0.0, 2.0]], dtype=np.float32)

    encoder = object.__new__(LocalQwenEncoder)
    encoder.model = FakeModel()

    vectors = encoder.encode(["synthetic"], batch_size=4)

    assert encoder.model.kwargs is not None
    assert encoder.model.kwargs["normalize_embeddings"] is False
    assert encoder.model.kwargs["convert_to_numpy"] is True
    np.testing.assert_array_equal(vectors, np.asarray([[0.0, 1.0]], dtype=np.float32))
