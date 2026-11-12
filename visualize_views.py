"""Inspect generated SSL views in a local browser, without model dependencies.

Run ``python visualize_views.py --help`` for the single-keyframe CLI, or call
``serve_views(views, original=item["point"])`` with an existing augmentation result.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
from urllib.parse import urlsplit
import webbrowser

import numpy as np


ASSET_DIR = Path(__file__).resolve().parent / "visualization"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    "/vendor/three.module.min.js": (
        "vendor/three.module.min.js",
        "text/javascript; charset=utf-8",
    ),
    "/vendor/three.core.min.js": (
        "vendor/three.core.min.js",
        "text/javascript; charset=utf-8",
    ),
    "/vendor/OrbitControls.js": (
        "vendor/OrbitControls.js",
        "text/javascript; charset=utf-8",
    ),
}


def make_payload(views: dict, *, original: dict | None = None, seed=None) -> dict:
    """Serialize actual view points, preserving both coordinate representations.

    ``original`` is the raw reader's point dictionary. No display subsampling,
    regeneration, or modification of the supplied views takes place.
    """
    entries = []

    def append_view(name, kind, point):
        entry = {
            "name": name,
            "kind": kind,
            "count": len(point["coord"]),
            "coord": point["coord"].tolist(),
            "origin_coord": point["origin_coord"].tolist(),
            "strength": point["strength"].reshape(-1).tolist(),
        }
        if kind == "masked_global":
            entry["mask"] = point["mask"].tolist()
            entry["mask_size"] = float(point["mask_size"])
            entry["mask_ratio"] = float(point["mask_ratio"])
            for key in ("mask_step", "mask_total_steps"):
                if key in point:
                    entry[key] = int(point[key])
        entries.append(entry)

    if original is not None:
        append_view("Original keyframe", "original", original)
    for index, point in enumerate(views["global_views"]):
        suffix = " (principal)" if index == 0 else ""
        append_view(f"Global {index}{suffix}", "global", point)
    for index, point in enumerate(views.get("masked_global_views", [])):
        append_view(f"Masked global {index}", "masked_global", point)
    for index, point in enumerate(views["local_views"]):
        append_view(f"Local {index}", "local", point)
    if not entries:
        raise ValueError("There are no views to display")

    meta = dict(views.get("meta", {}))
    if views["global_views"]:
        meta["grid_size"] = float(views["global_views"][0]["grid_size"])
    if seed is not None:
        meta["seed"] = seed
    return {"meta": meta, "views": entries}


def create_server(
    views: dict, *, original: dict | None = None, port: int = 8765, seed=None
) -> ThreadingHTTPServer:
    """Create a loopback-only server; the caller starts and closes it."""
    payload = json.dumps(
        make_payload(views, original=original, seed=seed),
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assets = {
        route: ((ASSET_DIR / filename).read_bytes(), content_type)
        for route, (filename, content_type) in STATIC_FILES.items()
    }
    assets["/api/views"] = (payload, "application/json; charset=utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            resource = assets.get(urlsplit(self.path).path)
            if resource is None:
                self.send_error(404)
                return
            body, content_type = resource
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_views(
    views: dict,
    *,
    original: dict | None = None,
    port: int = 8765,
    open_browser: bool = True,
    seed=None,
) -> None:
    """Display an existing augmentation result until interrupted with Ctrl+C."""
    server = create_server(views, original=original, port=port, seed=seed)
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        print(f"View SSL crops at {url}\nPress Ctrl+C to stop.", flush=True)
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("/home/cristian/Datasets/nuScenes"),
        help="nuScenes root (default: %(default)s)",
    )
    parser.add_argument("--version", default="v1.0-mini")
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--sample-index", type=int, default=0)
    sample.add_argument("--sample-token", help="select a sample by its nuScenes token")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grid-size", type=float, default=0.10, help="voxel width in meters"
    )
    parser.add_argument(
        "--mask-size",
        type=float,
        help="fixed masking patch width in meters (default: 0.4)",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        help="fixed fraction of occupied patches to mask (default: 0.3)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        help="enable Sonata's progressive masking schedule for this training length",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="training step to inspect with --total-steps (default: 0)",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="local port; 0 chooses a free port"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print the URL without opening it"
    )
    args = parser.parse_args(argv)

    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    if args.total_steps is None and args.step is not None:
        parser.error("--step requires --total-steps")
    if args.total_steps is not None and (
        args.mask_size is not None or args.mask_ratio is not None
    ):
        parser.error(
            "--total-steps cannot be combined with --mask-size or --mask-ratio"
        )

    # Reading and augmentation remain the project's existing NumPy pipeline.
    from augmentation import NuScenesSSLViews, MaskSchedule
    from datasets import NuScenesKeyframeReader

    try:
        if args.total_steps is not None:
            augment = NuScenesSSLViews(
                grid_size=args.grid_size,
                mask_schedule=MaskSchedule(args.total_steps),
            )
        else:
            augment = NuScenesSSLViews(
                grid_size=args.grid_size,
                mask_size=0.4 if args.mask_size is None else args.mask_size,
                mask_ratio=0.3 if args.mask_ratio is None else args.mask_ratio,
            )
        print(f"Indexing {args.version} keyframes...", flush=True)
        reader = NuScenesKeyframeReader(args.dataroot, version=args.version)
        if args.sample_token is None:
            if not 0 <= args.sample_index < len(reader):
                parser.error(f"--sample-index must be in [0, {len(reader) - 1}]")
            token = reader.sample_tokens[args.sample_index]
        else:
            token = args.sample_token
        print(f"Generating views for {token} (seed={args.seed})...", flush=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        item = reader.read(token)
        if args.total_steps is not None:
            views = augment(item, step=0 if args.step is None else args.step)
        else:
            views = augment(item)
        for kind in ("global_views", "masked_global_views", "local_views"):
            print(f"{kind}: {[len(view['coord']) for view in views[kind]]}", flush=True)
        serve_views(
            views,
            original=item["point"],
            port=args.port,
            open_browser=not args.no_browser,
            seed=args.seed,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
