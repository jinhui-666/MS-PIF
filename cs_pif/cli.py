from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import create_adapter, load_yaml, solver_signature
from .profiling import (
    FeasibilityAwareMaximinLHS,
    JsonlProfileStore,
    ProfilingRunner,
    StrictFeasibilityProbe,
    search_feasible_boundary,
)
from .runtime import BatchExecutor, check_adapter
from .scheduling import materialize_plan, schedule
from .proxy import GPConfig, ResourceSurrogate, fit_surrogate
from .backbone import BackbonePair, create_backbone_pair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cs-pif",
        description="Resource-aware batching for neural combinatorial solvers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    signature = subparsers.add_parser("signature")
    _add_config(signature)

    check = subparsers.add_parser("check-adapter")
    _add_config(check)
    check.add_argument("--small-size", type=int, default=20)
    check.add_argument("--large-size", type=int, default=50)

    profile = subparsers.add_parser("profile")
    _add_config(profile)
    profile.add_argument("--output", required=True)

    fit = subparsers.add_parser("fit-proxy")
    _add_config(fit)
    fit.add_argument("--profiles", required=True)
    fit.add_argument("--output", required=True)

    plan = subparsers.add_parser("plan")
    _add_config(plan)
    plan.add_argument("--surrogate", required=True)
    plan.add_argument("--data", required=True)
    plan.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    _add_config(run)
    run.add_argument("--surrogate", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--plan-output")
    run.add_argument("--output", required=True)
    return parser


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)


def _runtime(config_path: str) -> tuple[dict[str, Any], Any, str, BatchExecutor]:
    config = load_yaml(config_path)
    adapter_config = _mapping(config, "adapter")
    factory = str(adapter_config["factory"])
    options = adapter_config.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("adapter.options must be a mapping.")
    adapter = create_adapter(factory, options)
    device = str(config.get("device", "cuda:0"))
    signature = solver_signature(adapter, device)
    return config, adapter, signature, BatchExecutor(adapter, device)


def _profiling_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config, "profiling")


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_workload(adapter: Any, source: str) -> list[Any]:
    instances = list(adapter.problem.load(source))
    if not instances:
        raise ValueError(f"No instances were loaded from {source!r}.")
    return instances


def _load_surrogate(
    path: str,
    signature: str,
    backbones: BackbonePair,
    time_gp_config: GPConfig,
    memory_gp_config: GPConfig,
    rho: float,
) -> ResourceSurrogate:
    return ResourceSurrogate.load(
        path,
        expected_signature=signature,
        expected_backbone_signature=backbones.signature(),
        expected_time_gp_config=time_gp_config,
        expected_memory_gp_config=memory_gp_config,
        expected_rho=rho,
    )


def _backbone_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = config.get("backbone")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("surrogate.backbone must be a mapping.")
    return value


def _gp_config(config: Mapping[str, Any], key: str) -> GPConfig:
    value = config.get(key)
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"surrogate.{key} must be a mapping.")
    return GPConfig.from_mapping(value)


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    config, adapter, signature, executor = _runtime(args.config)
    seed = int(config.get("seed", 0))

    if args.command == "signature":
        surrogate_config = _mapping(config, "surrogate")
        backbones = create_backbone_pair(_backbone_config(surrogate_config))
        return {
            "solver_signature": signature,
            "manifest": adapter.manifest(),
            "backbones": backbones.description(),
            "backbone_signature": backbones.signature(),
        }

    if args.command == "check-adapter":
        report = check_adapter(
            adapter,
            executor,
            small_size=args.small_size,
            large_size=args.large_size,
            seed=seed,
        )
        return {"solver_signature": signature, "report": report}

    profiling = _profiling_config(config)
    if args.command == "profile":
        executor.load()
        safe = StrictFeasibilityProbe(
            adapter,
            executor,
            seed=seed,
            confirmations=int(profiling.get("safe_confirmations", 1)),
        )
        boundary = search_feasible_boundary(
            safe,
            min_size=int(profiling["min_size"]),
            max_size=int(profiling["max_size"]),
            max_batch_size=int(profiling.get("max_batch_size", 100)),
            anchor_count=int(profiling.get("safe_anchor_count", 16)),
        )
        design = FeasibilityAwareMaximinLHS(
            boundary,
            seed=seed,
            candidates=int(profiling.get("lhs_candidates", 32)),
            points=int(profiling.get("points", 100)),
        ).design()
        runner = ProfilingRunner(
            adapter,
            executor,
            signature,
            JsonlProfileStore(args.output),
            repeats=int(profiling.get("repeats", 5)),
            warmups=int(profiling.get("warmups", 1)),
            seed=seed,
        )
        records = runner.run(design)
        return {
            "solver_signature": signature,
            "profile_store": str(Path(args.output).resolve()),
            "records": len(records),
            "strict_feasible_boundary": boundary.to_dict(),
            "facm_lhs_points": len(design.points),
        }

    surrogate_config = _mapping(config, "surrogate")
    backbones = create_backbone_pair(_backbone_config(surrogate_config))
    time_gp_config = _gp_config(surrogate_config, "time_gp")
    memory_gp_config = _gp_config(surrogate_config, "memory_gp")
    rho = float(surrogate_config.get("rho", 0.9))
    if args.command == "fit-proxy":
        records = JsonlProfileStore(args.profiles).read()
        surrogate = fit_surrogate(
            records,
            solver_signature=signature,
            min_size=int(profiling["min_size"]),
            max_size=int(profiling["max_size"]),
            max_batch_size=int(profiling.get("max_batch_size", 100)),
            recommended_rho=rho,
            backbones=backbones,
            time_gp_config=time_gp_config,
            memory_gp_config=memory_gp_config,
        )
        surrogate.save(args.output)
        return {
            "solver_signature": signature,
            "surrogate": str(Path(args.output).resolve()),
            "backbones": surrogate.backbone_description(),
            "backbone_signature": backbones.signature(),
            "training": {
                "method": "direct_all_profile_points",
                "profile_points": len(surrogate.profile_config_ids),
                "calibration": False,
                "beta_memory": surrogate.beta_memory,
            },
        }

    executor.load()
    instances = _load_workload(adapter, args.data)
    surrogate = _load_surrogate(
        args.surrogate,
        signature,
        backbones,
        time_gp_config,
        memory_gp_config,
        rho,
    )
    plan = schedule(
        instances,
        adapter=adapter,
        surrogate=surrogate,
        free_memory_bytes=executor.free_memory_bytes(),
        rho=rho,
        max_batch_size=int(profiling.get("max_batch_size", 100)),
    )
    if args.command == "plan":
        plan.save(args.output)
        return {
            "solver_signature": signature,
            "plan": str(Path(args.output).resolve()),
            "batch_count": len(plan.batches),
            "predicted_total_seconds": plan.predicted_total_seconds,
        }

    if args.plan_output:
        plan.save(args.plan_output)
    results = []
    actual_seconds = 0.0
    peak_memory = 0
    oom_events = []
    for batch in materialize_plan(plan, instances, adapter.problem):
        batch_run = executor.run(batch, global_seed=seed, recover_oom=True)
        results.extend(batch_run.results)
        actual_seconds += batch_run.total_seconds
        peak_memory = max(peak_memory, batch_run.peak_incremental_memory_bytes)
        oom_events.extend(batch_run.oom_events)
    results_by_id = {result.instance_id: result for result in results}
    results = [
        results_by_id[adapter.problem.instance_id(instance)] for instance in instances
    ]
    output = {
        "solver_signature": signature,
        "config": config,
        "manifest": adapter.manifest(),
        "plan": plan,
        "actual_total_seconds": actual_seconds,
        "peak_incremental_memory_bytes": peak_memory,
        "oom_events": oom_events,
        "results": results,
    }
    _write_json(args.output, output)
    return {
        "solver_signature": signature,
        "results": str(Path(args.output).resolve()),
        "instance_count": len(results),
        "feasible_count": sum(result.feasible for result in results),
        "actual_total_seconds": actual_seconds,
        "oom_recoveries": len(oom_events),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _run_command(args)
    print(json.dumps(_json_value(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
