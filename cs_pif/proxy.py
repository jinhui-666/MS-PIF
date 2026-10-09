"""Two-stage B/N-backbone residual GP fitting and calibration utilities."""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
)

from .api import JsonValue
from .profiling import ProfileRecord, successful_records
from .backbone import (
    BackboneBasis,
    BackbonePair,
    default_backbone_pair,
    fitted_backbone_description,
)


_GP_INPUT_TRANSFORMS = {"B": "identity", "N": "log"}


@dataclass(frozen=True)
class GPConfig:
    """Frozen GP and observation-noise choices selected during development."""

    kernel: str = "matern"
    nu: float = 2.5
    noise_model: str = "repeat_variance"
    noise_floor: float = 1e-8
    constant_noise_variance: float = 1e-4
    predictive_noise_quantile: float = 0.9
    optimizer_restarts: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GPConfig":
        data = dict(value or {})
        config = cls(
            kernel=str(data.get("kernel", "matern")).lower(),
            nu=float(data.get("nu", 2.5)),
            noise_model=str(data.get("noise_model", "repeat_variance")).lower(),
            noise_floor=float(data.get("noise_floor", 1e-8)),
            constant_noise_variance=float(
                data.get("constant_noise_variance", 1e-4)
            ),
            predictive_noise_quantile=float(
                data.get("predictive_noise_quantile", 0.9)
            ),
            optimizer_restarts=int(data.get("optimizer_restarts", 0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.kernel not in {"matern", "rbf", "rational_quadratic"}:
            raise ValueError(f"Unsupported GP kernel {self.kernel!r}.")
        if self.kernel == "matern" and self.nu not in {0.5, 1.5, 2.5}:
            raise ValueError("Matérn nu must be one of 0.5, 1.5, or 2.5.")
        if self.noise_model not in {"repeat_variance", "constant"}:
            raise ValueError(f"Unsupported GP noise model {self.noise_model!r}.")
        if self.noise_floor <= 0 or self.constant_noise_variance <= 0:
            raise ValueError("GP noise variances must be positive.")
        if not 0 < self.predictive_noise_quantile <= 1:
            raise ValueError("predictive_noise_quantile must lie in (0, 1].")
        if self.optimizer_restarts < 0:
            raise ValueError("optimizer_restarts cannot be negative.")


@dataclass(frozen=True)
class ResourcePrediction:
    features: Mapping[str, float]
    batch_size: int
    max_size: int
    time_seconds: float
    memory_upper_bytes: float
    memory_mean_bytes: float
    memory_epistemic_std: float
    memory_predictive_std: float


class BackboneResidualGP:
    """Positive configured backbone followed by an exact log-residual GP."""

    def __init__(
        self,
        backbone_basis: BackboneBasis,
        gp_inputs: Sequence[str],
        config: GPConfig,
    ):
        self.backbone_basis = backbone_basis
        self.gp_inputs = tuple(gp_inputs)
        self.config = config
        self.backbone_coefficients: np.ndarray | None = None
        self.input_lower: np.ndarray | None = None
        self.input_upper: np.ndarray | None = None
        self.gp: GaussianProcessRegressor | None = None
        self.noise_std = 0.0

    def fit(
        self,
        features: Mapping[str, np.ndarray],
        labels: np.ndarray,
        repeat_noise_variances: np.ndarray,
    ) -> "BackboneResidualGP":
        labels = np.asarray(labels, dtype=float).reshape(-1)
        if np.any(labels <= 0) or not np.all(np.isfinite(labels)):
            raise ValueError("Resource labels must be finite and positive.")
        self.backbone_coefficients = _fit_positive_backbone(
            self.backbone_basis, features, labels
        )
        backbone = self.backbone_basis.evaluate(
            features, self.backbone_coefficients
        )
        residual = np.log(labels) - np.log(backbone)
        raw_inputs = self._raw_inputs(features)
        self.input_lower = raw_inputs.min(axis=0)
        self.input_upper = raw_inputs.max(axis=0)
        x = self._normalize(raw_inputs)
        repeat_noise = np.maximum(
            np.asarray(repeat_noise_variances, dtype=float),
            self.config.noise_floor,
        )
        if self.config.noise_model == "constant":
            alpha = np.full(len(labels), self.config.constant_noise_variance)
        else:
            alpha = repeat_noise
        self.gp = GaussianProcessRegressor(
            kernel=self._kernel(x.shape[1]),
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=self.config.optimizer_restarts,
            random_state=0,
        )
        self.gp.fit(x, residual)
        noise_source = (
            np.full(len(labels), self.config.constant_noise_variance)
            if self.config.noise_model == "constant"
            else repeat_noise
        )
        self.noise_std = float(
            np.quantile(
                np.sqrt(noise_source),
                self.config.predictive_noise_quantile,
            )
        )
        return self

    def predict(
        self,
        features: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.backbone_coefficients is None or self.gp is None:
            raise RuntimeError("Surrogate component has not been fitted.")
        backbone_log = np.log(
            self.backbone_basis.evaluate(features, self.backbone_coefficients)
        )
        residual_mean, epistemic = self.gp.predict(
            self._normalize(self._raw_inputs(features)),
            return_std=True,
        )
        predictive = np.sqrt(np.square(epistemic) + self.noise_std**2)
        return backbone_log + residual_mean, epistemic, predictive

    def _raw_inputs(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        missing = [name for name in self.gp_inputs if name not in features]
        if missing:
            raise ValueError(f"Missing GP resource features: {missing}")
        columns = []
        for name in self.gp_inputs:
            values = np.asarray(features[name], dtype=float).reshape(-1)
            if np.any(values <= 0):
                raise ValueError(f"GP input {name!r} must be positive.")
            transform = _GP_INPUT_TRANSFORMS.get(name)
            if transform is None:
                raise ValueError(f"Unsupported GP input {name!r}.")
            columns.append(np.log(values) if transform == "log" else values)
        return np.column_stack(columns)

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        if self.input_lower is None or self.input_upper is None:
            raise RuntimeError("GP normalization has not been fitted.")
        width = self.input_upper - self.input_lower
        return (raw - self.input_lower) / np.where(width > 0, width, 1.0)

    def _kernel(self, dimensions: int):
        amplitude = ConstantKernel(1.0, (1e-4, 1e4))
        length_scale = np.ones(dimensions)
        bounds = (1e-3, 1e3)
        if self.config.kernel == "matern":
            base = Matern(
                length_scale=length_scale,
                length_scale_bounds=bounds,
                nu=self.config.nu,
            )
        elif self.config.kernel == "rbf":
            base = RBF(length_scale=length_scale, length_scale_bounds=bounds)
        else:
            base = RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=bounds,
                alpha_bounds=(1e-3, 1e3),
            )
        return amplitude * base


def _fit_positive_backbone(
    basis: BackboneBasis,
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    """Minimize log-relative error subject to nonnegative coefficients."""
    components = basis.component_matrix(features)
    scales = np.maximum(np.median(components, axis=0), 1.0)
    scales[0] = 1.0
    normalized = components / scales[None, :]
    initial = np.full(
        normalized.shape[1],
        max(float(np.median(labels)) / normalized.shape[1], 1e-8),
    )

    def predict(coefficients: np.ndarray) -> np.ndarray:
        return normalized @ coefficients

    result = least_squares(
        lambda coefficients: np.log(np.maximum(predict(coefficients), 1e-12))
        - np.log(labels),
        initial,
        bounds=(np.full_like(initial, 1e-12), np.full_like(initial, np.inf)),
        max_nfev=20_000,
    )
    if not result.success:
        raise RuntimeError(f"Positive backbone fit failed: {result.message}")
    return np.asarray(result.x / scales, dtype=float)


class ResourceSurrogate:
    artifact_version = "7"

    def __init__(
        self,
        *,
        solver_signature: str,
        min_size: int,
        max_size: int,
        max_batch_size: int,
        beta_memory: float = 0.0,
        recommended_rho: float = 0.9,
        backbones: BackbonePair | None = None,
        time_gp_config: GPConfig | None = None,
        memory_gp_config: GPConfig | None = None,
    ):
        self.artifact_version = type(self).artifact_version
        self.solver_signature = solver_signature
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.max_batch_size = int(max_batch_size)
        self.beta_memory = float(beta_memory)
        self.recommended_rho = float(recommended_rho)
        if not 0 < self.recommended_rho < 1:
            raise ValueError("rho must lie strictly between zero and one.")
        self.backbones = backbones or default_backbone_pair()
        self.backbone_signature = self.backbones.signature()
        self.time_gp_config = time_gp_config or GPConfig()
        self.memory_gp_config = memory_gp_config or GPConfig()
        self.profile_config_ids: tuple[str, ...] = ()
        self.calibration_fold_ids: tuple[int, ...] = ()
        self.calibration_seed: int | None = None
        self.calibration_alpha: float | None = None
        self.time_model = BackboneResidualGP(
            self.backbones.time,
            self.backbones.gp_inputs,
            self.time_gp_config,
        )
        self.memory_model = BackboneResidualGP(
            self.backbones.memory,
            self.backbones.gp_inputs,
            self.memory_gp_config,
        )

    def fit(self, records: Sequence[ProfileRecord]) -> "ResourceSurrogate":
        records = successful_records(records)
        if len(records) < 4:
            raise ValueError(
                "At least four successful profiling configurations are required."
            )
        signatures = {record.solver_signature for record in records}
        if signatures != {self.solver_signature}:
            raise ValueError("Profiling records do not match the surrogate signature.")
        outside = [
            (record.batch_size, record.max_size)
            for record in records
            if not self.in_training_domain(record.batch_size, record.max_size)
        ]
        if outside:
            raise ValueError(
                f"Profiling records lie outside the declared domain: {outside}"
            )

        features = _record_feature_columns(records)
        time_labels = np.asarray(
            [record.time_label for record in records], dtype=float
        )
        memory_values = np.asarray(
            [record.memory_label for record in records], dtype=float
        )
        if np.any(memory_values <= 0):
            raise ValueError(
                "Memory labels must be positive; profile on an accelerator."
            )
        self.time_model.fit(
            features,
            time_labels,
            _repeat_log_variances(records, "seconds"),
        )
        self.memory_model.fit(
            features,
            memory_values,
            _repeat_log_variances(records, "peak_memory_bytes"),
        )
        return self

    def predict(self, features: Mapping[str, float]) -> ResourcePrediction:
        return self.predict_many([features])[0]

    def predict_many(
        self,
        feature_rows: Sequence[Mapping[str, float]],
    ) -> list[ResourcePrediction]:
        values = _feature_row_columns(feature_rows)
        batch_sizes = np.rint(values["B"]).astype(int)
        max_sizes = np.rint(values["N"]).astype(int)
        unsupported_batches = [
            int(batch_size)
            for batch_size in batch_sizes
            if not 1 <= int(batch_size) <= self.max_batch_size
        ]
        if unsupported_batches:
            raise ValueError(
                "Resource features contain unsupported batch sizes: "
                f"{unsupported_batches}"
            )
        time_mean, _, _ = self.time_model.predict(values)
        memory_mean, epistemic, predictive = self.memory_model.predict(values)
        return [
            ResourcePrediction(
                features={
                    name: float(column[index]) for name, column in values.items()
                },
                batch_size=int(batch_sizes[index]),
                max_size=int(max_sizes[index]),
                time_seconds=float(np.exp(time_mean[index])),
                memory_upper_bytes=float(
                    np.exp(
                        memory_mean[index]
                        + self.beta_memory * predictive[index]
                    )
                ),
                memory_mean_bytes=float(np.exp(memory_mean[index])),
                memory_epistemic_std=float(epistemic[index]),
                memory_predictive_std=float(predictive[index]),
            )
            for index in range(len(feature_rows))
        ]

    def backbone_description(self) -> dict[str, JsonValue]:
        return {
            "gp_inputs": list(self.backbones.gp_inputs),
            "gp_input_transforms": {
                name: _GP_INPUT_TRANSFORMS[name]
                for name in self.backbones.gp_inputs
            },
            "time": fitted_backbone_description(
                self.backbones.time,
                self.time_model.backbone_coefficients,
            ),
            "memory": fitted_backbone_description(
                self.backbones.memory,
                self.memory_model.backbone_coefficients,
            ),
            "time_gp": asdict(self.time_gp_config),
            "memory_gp": asdict(self.memory_gp_config),
        }

    def in_training_domain(self, batch_size: int, max_size: int) -> bool:
        return (
            1 <= batch_size <= self.max_batch_size
            and self.min_size <= max_size <= self.max_size
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_signature: str | None = None,
        expected_backbone_signature: str | None = None,
        expected_time_gp_config: GPConfig | None = None,
        expected_memory_gp_config: GPConfig | None = None,
        expected_rho: float | None = None,
    ) -> "ResourceSurrogate":
        try:
            with Path(path).open("rb") as handle:
                artifact = pickle.load(handle)
        except (AttributeError, ModuleNotFoundError) as exc:
            raise ValueError(
                "Unsupported surrogate artifact version; refit from profiles."
            ) from exc
        if not isinstance(artifact, cls):
            raise TypeError(f"{path} is not a ResourceSurrogate artifact.")
        if artifact.__dict__.get("artifact_version") != cls.artifact_version:
            raise ValueError("Unsupported surrogate artifact version.")
        if expected_signature and artifact.solver_signature != expected_signature:
            raise ValueError("Surrogate artifact solver signature mismatch.")
        if (
            expected_backbone_signature
            and artifact.backbone_signature != expected_backbone_signature
        ):
            raise ValueError("Surrogate artifact backbone mismatch.")
        if (
            expected_time_gp_config is not None
            and artifact.time_gp_config != expected_time_gp_config
        ):
            raise ValueError("Surrogate artifact time GP configuration mismatch.")
        if (
            expected_memory_gp_config is not None
            and artifact.memory_gp_config != expected_memory_gp_config
        ):
            raise ValueError("Surrogate artifact memory GP configuration mismatch.")
        if (
            expected_rho is not None
            and not math.isclose(artifact.recommended_rho, float(expected_rho))
        ):
            raise ValueError("Surrogate artifact rho mismatch.")
        return artifact


def build_unfitted_surrogate(
    *,
    solver_signature: str,
    min_size: int,
    max_size: int,
    max_batch_size: int = 100,
    recommended_rho: float = 0.9,
    backbones: BackbonePair | None = None,
    time_gp_config: GPConfig | None = None,
    memory_gp_config: GPConfig | None = None,
) -> ResourceSurrogate:
    return ResourceSurrogate(
        solver_signature=solver_signature,
        min_size=min_size,
        max_size=max_size,
        max_batch_size=max_batch_size,
        recommended_rho=recommended_rho,
        backbones=backbones,
        time_gp_config=time_gp_config,
        memory_gp_config=memory_gp_config,
    )


def _record_feature_columns(
    records: Sequence[ProfileRecord],
) -> dict[str, np.ndarray]:
    names = set(records[0].features)
    if any(set(record.features) != names for record in records):
        raise ValueError("Profiling records have inconsistent resource features.")
    return {
        name: np.asarray([record.features[name] for record in records], dtype=float)
        for name in sorted(names)
    }


def _feature_row_columns(
    feature_rows: Sequence[Mapping[str, float]],
) -> dict[str, np.ndarray]:
    if not feature_rows:
        raise ValueError("At least one resource feature row is required.")
    names = set(feature_rows[0])
    if any(set(row) != names for row in feature_rows):
        raise ValueError("Resource feature rows have inconsistent columns.")
    values = {
        str(name): np.asarray(
            [float(row[name]) for row in feature_rows],
            dtype=float,
        )
        for name in sorted(names)
    }
    if "B" not in values or "N" not in values:
        raise ValueError("Resource features must include B and N.")
    if any(
        not np.all(np.isfinite(value)) or np.any(value <= 0)
        for value in values.values()
    ):
        raise ValueError("Resource features must be finite and positive.")
    if set(values) != {"B", "N"}:
        raise ValueError("Resource features must contain exactly B and N.")
    return values


def _repeat_log_variances(
    records: Sequence[ProfileRecord],
    field: str,
) -> np.ndarray:
    values = []
    for record in records:
        repeats = np.asarray(
            [max(float(getattr(item, field)), 1e-12) for item in record.repeats]
        )
        variance = (
            np.var(np.log(repeats), ddof=1) if len(repeats) >= 2 else 1e-6
        )
        values.append(max(float(variance), 1e-8))
    return np.asarray(values)


@dataclass(frozen=True)
class OOFPrediction:
    config_id: str
    batch_size: int
    max_size: int
    actual_time_seconds: float
    predicted_time_seconds: float
    actual_memory_bytes: float
    predicted_memory_mean_bytes: float
    memory_predictive_std: float
    memory_epistemic_std: float


@dataclass(frozen=True)
class CalibrationSummary:
    beta_memory: float
    fold_ids: tuple[int, ...]
    standardized_residuals: tuple[float, ...]
    oof_predictions: tuple[OOFPrediction, ...]


def fit_surrogate(
    records: Sequence[ProfileRecord],
    *,
    solver_signature: str,
    min_size: int,
    max_size: int,
    max_batch_size: int = 100,
    recommended_rho: float = 0.9,
    backbones: BackbonePair | None = None,
    time_gp_config: GPConfig | None = None,
    memory_gp_config: GPConfig | None = None,
) -> ResourceSurrogate:
    """Fit once on all successful profile points without calibration."""
    signatures = {record.solver_signature for record in records}
    if signatures != {solver_signature}:
        raise ValueError(
            "Profiling records do not match the requested solver signature."
        )
    success = successful_records(records)
    model = build_unfitted_surrogate(
        solver_signature=solver_signature,
        min_size=min_size,
        max_size=max_size,
        max_batch_size=max_batch_size,
        recommended_rho=recommended_rho,
        backbones=backbones,
        time_gp_config=time_gp_config,
        memory_gp_config=memory_gp_config,
    ).fit(success)
    model.profile_config_ids = tuple(record.config_id for record in success)
    return model


def fit_calibrated_surrogate(
    records: Sequence[ProfileRecord],
    *,
    solver_signature: str,
    min_size: int,
    max_size: int,
    max_batch_size: int = 100,
    alpha: float = 0.05,
    recommended_rho: float = 0.9,
    folds: int = 5,
    seed: int = 0,
    backbones: BackbonePair | None = None,
    time_gp_config: GPConfig | None = None,
    memory_gp_config: GPConfig | None = None,
) -> tuple[ResourceSurrogate, CalibrationSummary]:
    signatures = {record.solver_signature for record in records}
    if signatures != {solver_signature}:
        raise ValueError(
            "Profiling records do not match the requested solver signature."
        )
    success = successful_records(records)
    if len(success) < max(5, folds):
        raise ValueError(f"At least {max(5, folds)} successful points are required.")
    fold_ids = spatial_fold_ids(success, folds=folds, seed=seed)
    standardized: list[float] = []
    oof_predictions: list[OOFPrediction] = []

    for fold in range(folds):
        train = [
            record for record, fold_id in zip(success, fold_ids) if fold_id != fold
        ]
        validation = [
            record for record, fold_id in zip(success, fold_ids) if fold_id == fold
        ]
        if not validation:
            continue
        model = build_unfitted_surrogate(
            solver_signature=solver_signature,
            min_size=min_size,
            max_size=max_size,
            max_batch_size=max_batch_size,
            recommended_rho=recommended_rho,
            backbones=backbones,
            time_gp_config=time_gp_config,
            memory_gp_config=memory_gp_config,
        ).fit(train)
        for record in validation:
            prediction = model.predict(record.features)
            actual_log_memory = math.log(record.memory_label)
            predicted_log_memory = math.log(prediction.memory_mean_bytes)
            standardized.append(
                (actual_log_memory - predicted_log_memory)
                / max(prediction.memory_predictive_std, 1e-8)
            )
            oof_predictions.append(
                OOFPrediction(
                    config_id=record.config_id,
                    batch_size=record.batch_size,
                    max_size=record.max_size,
                    actual_time_seconds=record.time_label,
                    predicted_time_seconds=prediction.time_seconds,
                    actual_memory_bytes=record.memory_label,
                    predicted_memory_mean_bytes=prediction.memory_mean_bytes,
                    memory_predictive_std=prediction.memory_predictive_std,
                    memory_epistemic_std=prediction.memory_epistemic_std,
                )
            )

    beta = max(0.0, finite_sample_upper_quantile(standardized, 1.0 - alpha))
    final = build_unfitted_surrogate(
        solver_signature=solver_signature,
        min_size=min_size,
        max_size=max_size,
        max_batch_size=max_batch_size,
        recommended_rho=recommended_rho,
        backbones=backbones,
        time_gp_config=time_gp_config,
        memory_gp_config=memory_gp_config,
    )
    final.beta_memory = beta
    final.profile_config_ids = tuple(record.config_id for record in success)
    final.calibration_fold_ids = tuple(fold_ids)
    final.calibration_seed = int(seed)
    final.calibration_alpha = float(alpha)
    final.fit(success)
    return final, CalibrationSummary(
        beta_memory=beta,
        fold_ids=tuple(fold_ids),
        standardized_residuals=tuple(float(value) for value in standardized),
        oof_predictions=tuple(oof_predictions),
    )


def finite_sample_upper_quantile(values: Sequence[float], coverage: float) -> float:
    if not values:
        raise ValueError("Cannot calibrate from an empty residual list.")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one.")
    ordered = sorted(float(value) for value in values)
    one_based = math.ceil((len(ordered) + 1) * coverage)
    one_based = min(max(one_based, 1), len(ordered))
    return ordered[one_based - 1]


def spatial_fold_ids(
    records: Sequence[ProfileRecord],
    *,
    folds: int,
    seed: int,
) -> list[int]:
    if folds < 2 or folds > len(records):
        raise ValueError("Invalid fold count.")
    batch = np.asarray([record.batch_size for record in records], dtype=float)
    size = np.log(np.asarray([record.max_size for record in records], dtype=float))
    batch_edges = np.quantile(batch, [0.33, 0.66])
    size_edges = np.quantile(size, [0.33, 0.66])
    cells: dict[tuple[int, int], list[int]] = {}
    for index, (batch_value, size_value) in enumerate(zip(batch, size)):
        cell = (
            int(np.searchsorted(batch_edges, batch_value, side="right")),
            int(np.searchsorted(size_edges, size_value, side="right")),
        )
        cells.setdefault(cell, []).append(index)

    rng = np.random.default_rng(seed)
    assignment = [-1] * len(records)
    offset = 0
    for cell in sorted(cells):
        indices = cells[cell]
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignment[index] = (offset + position) % folds
        offset = (offset + len(indices)) % folds
    return assignment
