"""Opt-in CLIs. Both refuse to run without their explicit enable flag."""

import argparse
import json
import os
import sys

from .live_eval import DEFAULT_DEADLINE_SECONDS, DEFAULT_MAX_STEPS, DEFAULT_REPETITIONS
from .live_eval import run_live_eval
from .vendor_smoke import DEFAULT_PLAN, run_vendor_smoke, write_smoke_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opt-in live harnesses for the distribute_agent prototype."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser(
        "live-eval", help="5 briefs x 3 repetitions against a real model"
    )
    live.add_argument("--enable-live-model", action="store_true",
                      help="required: allow calls to a billed model provider")
    live.add_argument("--model-backend", choices=("litellm", "inference-client"),
                      default="litellm")
    live.add_argument("--model-id", help="provider model id, for example openai/gpt-4o-mini")
    live.add_argument("--api-key-env", default="HF_TOKEN",
                      help="environment variable holding the provider key")
    live.add_argument("--package", required=True, help="path to the shipped package")
    live.add_argument("--workspace", required=True, help="directory for per-run sandboxes")
    live.add_argument("--output", help="directory for live-eval.json")
    live.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    live.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    live.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    live.add_argument("--max-requests", type=int,
                      help="hard cap on provider requests for this invocation; "
                           "enforced at the model call, including final answers")

    vendor = subparsers.add_parser(
        "vendor-smoke", help="one real RunPod job, artifact checked by header"
    )
    vendor.add_argument("--enable-live-vendor", action="store_true",
                        help="required: allow one billed RunPod job")
    vendor.add_argument("--package", required=True, help="path to the shipped package")
    vendor.add_argument("--output", required=True, help="directory for job artifacts")
    vendor.add_argument("--budget-seconds", type=float, default=900.0)
    vendor.add_argument("--seed", type=int, default=DEFAULT_PLAN["seed"])
    vendor.add_argument("--width", type=int, default=DEFAULT_PLAN["width"])
    vendor.add_argument("--height", type=int, default=DEFAULT_PLAN["height"])
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "live-eval":
        report = run_live_eval(
            enable_live_model=arguments.enable_live_model,
            model_factory=(
                _model_factory(arguments) if arguments.enable_live_model else None
            ),
            package_root=arguments.package,
            workspace_root=arguments.workspace,
            output_dir=arguments.output,
            repetitions=arguments.repetitions,
            max_steps=arguments.max_steps,
            deadline_seconds=arguments.deadline_seconds,
            max_requests=arguments.max_requests,
            model_metadata={
                "backend": arguments.model_backend,
                "model_id": arguments.model_id or "(not set)",
                "api_key_env": arguments.api_key_env,
            },
        )
        print(json.dumps(report, indent=2))
        if report["status"] == "not-run":
            return 2
        return 0 if report["status"] == "completed" else 1

    report = run_vendor_smoke(
        enable_live_vendor=arguments.enable_live_vendor,
        package_root=arguments.package,
        output_dir=arguments.output,
        plan_data={
            **DEFAULT_PLAN,
            "seed": arguments.seed,
            "width": arguments.width,
            "height": arguments.height,
        },
        budget_seconds=arguments.budget_seconds,
    )
    if report["status"] != "not-run":
        write_smoke_report(report, arguments.output)
    print(json.dumps(report, indent=2))
    if report["status"] == "not-run":
        return 2
    return 0 if report["status"] == "passed" else 1


def _model_factory(arguments):
    model_id = arguments.model_id
    if not model_id:
        raise SystemExit("--model-id is required with --enable-live-model")
    key = os.environ.get(arguments.api_key_env)
    if not key:
        raise SystemExit(
            f"{arguments.api_key_env} is not set — export the provider key in the "
            "environment; it is never written to artifacts"
        )
    if arguments.model_backend == "litellm":
        from smolagents import LiteLLMModel

        model_class, model_kwargs = LiteLLMModel, {"api_key": key}
    else:
        # huggingface-hub, the only dependency of this backend, ships with base smolagents.
        from smolagents import InferenceClientModel

        model_class, model_kwargs = InferenceClientModel, {"token": key}

    def factory():
        return model_class(model_id=model_id, **model_kwargs)

    # Preflight once, before the matrix starts: constructing the model triggers the
    # backend's own dependency import (litellm for LiteLLMModel) but no network call.
    # A missing backend must not be discovered after the first bills are spent.
    try:
        factory()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"the {arguments.model_backend} backend cannot be constructed: {exc} — "
            "install it (this project's pyproject extra is 'litellm') and rerun. "
            "No provider call was made"
        ) from exc
    return factory


if __name__ == "__main__":
    sys.exit(main())
