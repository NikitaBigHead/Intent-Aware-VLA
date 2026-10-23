"""Receives per-chunk telemetry from cosmos_right13_bridge.py (robot side,
see chunk_logger.py) and appends it to disk on GPU-02 -- durable, off-robot
storage for target-vs-achieved tracking error and 'c'-key camera captures.

Deploy (GPU-02 HOST, NOT inside tactile-train -- this needs only stdlib +
numpy + msgpack + pyzmq [+ opencv-python only if captures are ever
received], no GPU/CUDA/Isaac-GR00T, so it runs directly on the host's
system/conda python, same as the socat forwarder):

    scp scripts/robot_deploy/cosmos_right13/chunk_log_server.py \
        root@gpu-server-a:/root/tactile_ws/chunk_log_server.py -P 2222
    ssh -p 2222 root@gpu-server-a
    tmux new -s chunklog
    python3 /root/tactile_ws/chunk_log_server.py --port 5560 \
        --out-dir /root/tactile_ws/cosmos_right13_logs
    # Ctrl+B, then D to detach

Binds a plain ZMQ PULL on 0.0.0.0:<port> directly on the host -- no socat
hop needed (unlike the policy server, this receiver doesn't live inside a
container). The robot's --log-host/--log-port (cosmos_right13_bridge.py)
must be able to reach this port directly; open it the same way port 5555
already had to be.

Per run (run_id = timestamp_side[_mirror], set once at bridge startup),
writes under `<out-dir>/<run_id>/`:
    chunk_00000.npz, chunk_00001.npz, ...   -- one per received chunk record
    summary.jsonl                            -- one line per chunk, just the
                                                 scalars (tracking-error norm,
                                                 inference time) for `tail -f`
    captures/capture_<ts>_chunk<N>.png       -- 'c'-key frame captures
    session.log                              -- mirror of the bridge's own
                                                 console output (stdout+
                                                 stderr), streamed live by
                                                 its _Tee -- so a run that
                                                 dies before its first chunk
                                                 (e.g. policy-server ping
                                                 failure) still leaves a
                                                 transcript here even if the
                                                 robot itself goes unreachable
                                                 afterward
    ticks.csv                                -- one row per control-loop
                                                 tick (~50Hz): commanded vs.
                                                 freshly-measured 13-channel
                                                 state, plus which chunk/row
                                                 was playing and exactly which
                                                 tick triggered/consumed an
                                                 inference request -- see
                                                 chunk_logger.py's TickLogger

Wire format matches gr00t_zmq_client.py's ndarray envelope
(`{"__ndarray_class__": True, "as_npy": ...}`) -- decode helper is a local
copy here for the same reason chunk_logger.py keeps its own encode helper
rather than importing gr00t_zmq_client.py: a receiver for the log stream
has no business depending on the policy client.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zmq


def _decode_custom_classes(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode_custom_classes, raw=False)


def _handle_capture(record: dict, run_dir: Path) -> None:
    captures_dir = run_dir / "captures"
    captures_dir.mkdir(exist_ok=True)
    frame = record.get("frame")
    if frame is None:
        print(f"[chunk_log_server] capture record for {record.get('run_id')} had no frame, skipped")
        return
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(record.get("timestamp", time.time())))
    chunk_index = record.get("chunk_index", -1)
    path = captures_dir / f"capture_{ts}_chunk{chunk_index:05d}.png"
    try:
        import cv2
        # frame arrives in the model's RGB convention (see chunk_logger.py /
        # cosmos_right13_bridge.py's build_observation) -- cv2.imwrite wants BGR.
        cv2.imwrite(str(path), np.asarray(frame)[:, :, ::-1])
    except Exception as exc:
        print(f"[chunk_log_server] capture write failed: {exc}")
        return
    print(f"[chunk_log_server] {record.get('run_id')}: saved capture -> {path}")


def _handle_session_log(record: dict, run_dir: Path) -> None:
    text = record.get("text", "")
    if not text:
        return
    with open(run_dir / "session.log", "a", encoding="utf-8") as fh:
        fh.write(text)


def _handle_tick_batch(record: dict, run_dir: Path) -> None:
    columns = record.get("columns")
    rows = record.get("rows")
    if columns is None or rows is None:
        return
    path = run_dir / "ticks.csv"
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8") as fh:
        if is_new:
            fh.write(",".join(columns) + "\n")
        for row in np.asarray(rows):
            fh.write(",".join(repr(float(v)) for v in row) + "\n")


def _handle_chunk(record: dict, run_dir: Path) -> None:
    chunk_index = record.get("chunk_index", 0)
    arrays = {k: v for k, v in record.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in record.items() if not isinstance(v, np.ndarray) and v is not None}
    try:
        np.savez_compressed(run_dir / f"chunk_{chunk_index:05d}.npz", **arrays, **scalars)
    except Exception as exc:
        print(f"[chunk_log_server] npz write failed for chunk {chunk_index}: {exc}")

    tracking_error = record.get("tracking_error")
    err_norm = float(np.linalg.norm(tracking_error)) if isinstance(tracking_error, np.ndarray) else None
    with open(run_dir / "summary.jsonl", "a") as fh:
        fh.write(json.dumps({
            "chunk_index": chunk_index,
            "timestamp": record.get("timestamp"),
            "inference_seconds": record.get("inference_seconds"),
            "tracking_error_norm": err_norm,
        }) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--out-dir", default="/root/tactile_ws/cosmos_right13_logs")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[chunk_log_server] listening on 0.0.0.0:{args.port}, writing under {out_root}")

    n_received = 0
    while True:
        try:
            message = sock.recv()
        except KeyboardInterrupt:
            break
        try:
            record = _unpack(message)
        except Exception as exc:
            print(f"[chunk_log_server] bad message dropped: {exc}")
            continue

        run_id = record.get("run_id", "unknown_run")
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        kind = record.get("kind")
        if kind == "capture":
            _handle_capture(record, run_dir)
        elif kind == "session_log":
            _handle_session_log(record, run_dir)
        elif kind == "tick_batch":
            _handle_tick_batch(record, run_dir)
        else:
            _handle_chunk(record, run_dir)

        n_received += 1
        if n_received % 50 == 0:
            print(f"[chunk_log_server] {n_received} records received so far")


if __name__ == "__main__":
    main()
