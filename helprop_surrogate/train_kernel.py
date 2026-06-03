"""Train and save a conditional kernel surrogate from HelProp samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_bson_transitions, load_npz_transitions
from .kernel import ConditionalKernelSurrogate
from .model import HelPropKernelModel, parse_key_value_options, parse_range_options
from .neural import NeuralConditionalKernelSurrogate
from .torch_neural import TorchNeuralConditionalKernelSurrogate


def _load_dataset(path: Path, input_format: str, param_names: tuple[str, ...]):
    if input_format == "auto":
        input_format = "bson" if path.suffix.lower() in {".bson", ".bin"} else "npz"
    if input_format == "bson":
        return load_bson_transitions(path, param_names=param_names)
    if input_format == "npz":
        return load_npz_transitions(path)
    raise ValueError(f"unsupported input format: {input_format}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a conditional probabilistic kernel surrogate for HelProp."
    )
    parser.add_argument("input", type=Path, help="Transition sample file (.npz or HelProp BSON)")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output pickle file; kept for compatibility, prefer --model-out",
    )
    parser.add_argument("--model-out", type=Path, help="Output model pickle")
    parser.add_argument(
        "--backend",
        choices=["kde", "neural", "numpy-neural"],
        default="kde",
        help="Conditional density backend; neural uses PyTorch/GPU when available",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "npz", "bson"],
        default="auto",
        help="Input file format",
    )
    parser.add_argument(
        "--learn",
        nargs="+",
        default=["D0", "m"],
        help="Learned HelProp parameter names, in order",
    )
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        help="Fixed HelProp parameter as name=value; hidden at runtime",
    )
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        help="Training range as name:min:max; repeat for learned parameters",
    )
    parser.add_argument(
        "--etoa-grid",
        default=None,
        help="Store default TOA grid as min,max,n or file first column",
    )
    parser.add_argument(
        "--elis-grid",
        default=None,
        help="Store default LIS grid as min,max,n or file first column",
    )
    parser.add_argument(
        "--condition-bandwidth",
        type=float,
        default=0.6,
        help="Gaussian bandwidth in standardized condition space",
    )
    parser.add_argument(
        "--target-bandwidth",
        type=float,
        default=None,
        help="Bandwidth in log(ELIS / ETOA); default uses Scott/Silverman scaling",
    )
    parser.add_argument("--hidden-size", type=int, default=24, help="Neural backend hidden units")
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=None,
        help="PyTorch neural hidden layer sizes",
    )
    parser.add_argument("--components", type=int, default=4, help="Neural Gaussian mixture components")
    parser.add_argument("--epochs", type=int, default=300, help="Neural backend training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Neural backend batch size")
    parser.add_argument("--learning-rate", type=float, default=1.0e-2, help="Neural backend learning rate")
    parser.add_argument("--weight-decay", type=float, default=1.0e-5, help="PyTorch neural weight decay")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=123, help="Neural backend random seed")
    parser.add_argument("--verbose-train", action="store_true", help="Print neural training loss")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.model_out or args.output
    if output is None:
        raise ValueError("output model path is required; use --model-out or positional output")

    param_names = tuple(args.learn)
    dataset = _load_dataset(args.input, args.format, param_names)

    if args.backend == "kde":
        kernel = ConditionalKernelSurrogate(
            param_names=param_names,
            condition_bandwidth=args.condition_bandwidth,
            target_bandwidth=args.target_bandwidth,
        ).fit(dataset.etoa, dataset.elis, dataset.select_params(param_names))
    elif args.backend == "numpy-neural":
        kernel = NeuralConditionalKernelSurrogate(
            param_names=param_names,
            hidden_size=args.hidden_size,
            n_components=args.components,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            random_state=args.seed,
            verbose=args.verbose_train,
        ).fit(dataset.etoa, dataset.elis, dataset.select_params(param_names))
    else:
        kernel = TorchNeuralConditionalKernelSurrogate(
            param_names=param_names,
            hidden_sizes=tuple(args.hidden_sizes or [args.hidden_size, args.hidden_size]),
            n_components=args.components,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            random_state=args.seed,
            device=args.device,
            verbose=args.verbose_train,
        ).fit(dataset.etoa, dataset.elis, dataset.select_params(param_names))

    ranges = parse_range_options(args.ranges)
    unknown_ranges = set(ranges).difference(param_names)
    if unknown_ranges:
        names = ", ".join(sorted(unknown_ranges))
        raise ValueError(f"ranges supplied for non-learned parameters: {names}")

    model = HelPropKernelModel(
        kernel=kernel,
        learned=param_names,
        fixed=parse_key_value_options(args.fixed),
        ranges=ranges,
        etoa_grid=_read_grid_option(args.etoa_grid),
        elis_grid=_read_grid_option(args.elis_grid),
    )
    model.save(output)

    print(f"trained samples: {dataset.etoa.size}")
    print(f"learned parameters: {', '.join(param_names)}")
    print(f"fixed parameters: {', '.join(model.fixed) if model.fixed else '(none)'}")
    if hasattr(kernel, "target_bandwidth_"):
        print(f"target bandwidth: {kernel.target_bandwidth_:.6g}")
    else:
        print(f"backend: {args.backend}")
    print(f"saved: {output}")
    return 0


def _read_grid_option(spec: str | None) -> tuple[float, ...] | None:
    if spec is None:
        return None
    import numpy as np

    path = Path(spec)
    if path.exists():
        data = np.loadtxt(path)
        values = data if data.ndim == 1 else data[:, 0]
        return tuple(float(value) for value in values)

    parts = [float(item) for item in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("grid option must be a file path or min,max,n")
    emin, emax, nbin = parts
    return tuple(float(value) for value in np.geomspace(emin, emax, int(nbin)))


if __name__ == "__main__":
    raise SystemExit(main())
