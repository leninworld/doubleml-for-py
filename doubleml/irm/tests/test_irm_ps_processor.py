"""Tests for propensity score processing in the interactive regression model."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLData, DoubleMLIRM
from doubleml.utils.propensity_score_processing import PSProcessorConfig


@pytest.mark.ci
@pytest.mark.parametrize(
    "ps_config",
    [
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=0.05, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=True),
    ],
)
def test_irm_ml_m_predictions_ps_processor(generate_data_irm, ps_config):
    x, y, d = generate_data_irm
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    np.random.seed(3141)
    dml_irm = DoubleMLIRM(
        obj_dml_data=dml_data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        ps_processor_config=ps_config,
        n_rep=1,
    )
    dml_irm.fit(store_predictions=True)
    ml_m_preds = dml_irm.predictions["ml_m"][:, 0, 0]
    # Just check that predictions are within [clipping_threshold, 1-clipping_threshold]
    assert np.all(ml_m_preds >= ps_config.clipping_threshold)
    assert np.all(ml_m_preds <= 1 - ps_config.clipping_threshold)


@pytest.mark.ci
def test_irm_ml_m_predictions_ps_processor_differences(generate_data_irm):
    x, y, d = generate_data_irm
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    np.random.seed(3141)
    configs = [
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=0.05, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=True),
    ]
    preds = []
    for cfg in configs:
        dml_irm = DoubleMLIRM(
            obj_dml_data=dml_data,
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            ps_processor_config=cfg,
            n_rep=1,
        )
        dml_irm.fit(store_predictions=True)
        preds.append(dml_irm.predictions["ml_m"][:, 0, 0])
    # Check that at least two configurations yield different predictions (element-wise)
    diffs = [not np.allclose(preds[i], preds[j], atol=1e-6) for i in range(len(preds)) for j in range(i + 1, len(preds))]
    assert any(diffs)


@pytest.mark.ci
def test_irm_retains_raw_propensity_from_external_predictions(generate_data_irm):
    """Retain external propensity predictions before clipping without changing prediction keys."""
    x, y, d = generate_data_irm
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    n_rep = 2
    raw_propensity = np.tile(np.linspace(0.001, 0.999, dml_data.n_obs)[:, np.newaxis], (1, n_rep))
    external_predictions = {"d": {"ml_m": raw_propensity}}
    dml_irm = DoubleMLIRM(
        obj_dml_data=dml_data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        ps_processor_config=PSProcessorConfig(clipping_threshold=0.1),
        n_rep=n_rep,
    )

    dml_irm.fit(store_predictions=True, external_predictions=external_predictions)

    expected_raw = raw_propensity[:, :, np.newaxis]
    np.testing.assert_array_equal(dml_irm.raw_propensity, expected_raw)
    np.testing.assert_array_equal(dml_irm.predictions["ml_m"], np.clip(expected_raw, 0.1, 0.9))
    assert dml_irm.params_names == ["ml_g0", "ml_g1", "ml_m"]
    assert set(dml_irm.predictions) == {"ml_g0", "ml_g1", "ml_m"}

    returned_raw = dml_irm.raw_propensity
    returned_raw[0, 0, 0] = 0.5
    assert dml_irm.raw_propensity[0, 0, 0] == raw_propensity[0, 0]


@pytest.mark.ci
def test_irm_raw_propensity_lifecycle_respects_store_predictions(generate_data_irm):
    """Store and reset raw propensity only when nuisance predictions are stored."""
    x, y, d = generate_data_irm
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    dml_irm = DoubleMLIRM(
        obj_dml_data=dml_data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        ps_processor_config=PSProcessorConfig(clipping_threshold=0.05),
    )
    assert dml_irm.raw_propensity is None

    first_raw = np.full((dml_data.n_obs, 1), 0.2)
    dml_irm.fit(store_predictions=False, external_predictions={"d": {"ml_m": first_raw}})
    assert dml_irm.raw_propensity is None

    second_raw = np.full((dml_data.n_obs, 1), 0.8)
    dml_irm.fit(store_predictions=True, external_predictions={"d": {"ml_m": second_raw}})
    np.testing.assert_array_equal(dml_irm.raw_propensity[:, :, 0], second_raw)

    dml_irm.fit(store_predictions=False, external_predictions={"d": {"ml_m": first_raw}})
    assert dml_irm.raw_propensity is None


@pytest.mark.ci
def test_irm_raw_propensity_does_not_change_fit_results(generate_data_irm):
    """Keep fitted estimates identical whether processed predictions are stored or not."""
    x, y, d = generate_data_irm
    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)
    raw_propensity = np.full((dml_data.n_obs, 1), 0.5)
    external_predictions = {"d": {"ml_m": raw_propensity}}
    kwargs = {
        "obj_dml_data": dml_data,
        "ml_g": LinearRegression(),
        "ml_m": LogisticRegression(),
        "ps_processor_config": PSProcessorConfig(clipping_threshold=0.01),
    }
    stored = DoubleMLIRM(**kwargs)
    unstored = DoubleMLIRM(**kwargs)
    unstored.set_sample_splitting(stored.smpls)

    stored.fit(store_predictions=True, external_predictions=external_predictions)
    unstored.fit(store_predictions=False, external_predictions=external_predictions)

    assert unstored.predictions is None
    assert unstored.raw_propensity is None
    np.testing.assert_array_equal(stored.predictions["ml_m"], stored.raw_propensity)
    np.testing.assert_allclose(unstored.coef, stored.coef)
    np.testing.assert_allclose(unstored.se, stored.se)
