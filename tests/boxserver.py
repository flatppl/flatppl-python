"""An in-process flatppl-bridge box server with closed-form members, for tests.

It answers the query forms the client sends: data parts, ``rnginit``, single
and broadcast queries, gradients and vector-Jacobian products, and the
content store. ``requests`` records every query text it answered.
"""

from __future__ import annotations

import email.parser
import email.policy
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pyarrow as pa
from scipy import stats

DESCRIBE: dict[str, dict[str, Any]] = {
    "posterior": {
        "kind": "measure",
        "queries": ["builtin_logdensityof"],
        "gradient": True,
        "vjp": False,
        "variate": "cartpow(reals, 2)",
        "normalized": False,
    },
    "K": {
        "kind": "kernel",
        "queries": ["builtin_logdensityof", "builtin_sample"],
        "gradient": True,
        "vjp": False,
        "input": "cartprod(mu = reals, sigma = posreals)",
        "variate": "reals",
        "normalized": True,
    },
    "std": {
        "kind": "measure",
        "queries": ["builtin_logdensityof", "builtin_sample"],
        "gradient": True,
        "vjp": False,
        "variate": "reals",
        "normalized": True,
    },
    "f": {
        "kind": "function",
        "queries": ["apply"],
        "gradient": True,
        "vjp": True,
        "input": "cartpow(reals, 3)",
        "output": "reals",
    },
    "g": {
        "kind": "function",
        "queries": ["apply"],
        "gradient": False,
        "vjp": True,
        "input": "cartpow(reals, 2)",
        "output": "cartpow(reals, 2)",
    },
}

INTERFACE = "".join(f"{name} = external(anything)\n" for name in DESCRIBE)


def posterior_logdensity(x: np.ndarray) -> np.ndarray:
    return -0.5 * np.sum((x - 1.0) ** 2, axis=-1)


def g(x: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(x[..., 0]), x[..., 0] * x[..., 1]], axis=-1)


def _read(data: bytes) -> pa.Table:
    return pa.ipc.open_stream(data).read_all()


def _numpy(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    dims = []
    values = column
    while pa.types.is_fixed_size_list(values.type):
        dims.append(values.type.list_size)
        values = values.flatten()
    return values.to_numpy(zero_copy_only=False).reshape([len(column), *dims])


def _part(value: np.ndarray, valueset: str) -> bytes:
    value = np.asarray(value)
    if value.ndim == 0:
        column = pa.array([value.item()])
    else:
        column = pa.array(value.reshape(-1))
        for size in reversed(value.shape[1:]):
            column = pa.FixedSizeListArray.from_arrays(column, size)
    table = pa.table({"value": column})
    table = table.replace_schema_metadata({"flatppl.valueset": valueset})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _record_part(fields: dict[str, np.ndarray]) -> bytes:
    table = pa.table(
        {k: pa.array(np.asarray(v).reshape(-1)) for k, v in fields.items()}
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


class Problem(Exception):
    def __init__(self, status: int, title: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.status, self.title, self.detail, self.extra = status, title, detail, extra


class FakeBoxServer:
    """Serve one box named ``demo`` on a free localhost port."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.store: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.derivatives: list[dict[str, Any]] = []
        self.uploads: list[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _problem(self, problem: Problem) -> None:
                body = {
                    "title": problem.title,
                    "detail": problem.detail,
                    **problem.extra,
                }
                self._send(
                    problem.status,
                    json.dumps(body).encode(),
                    "application/problem+json",
                )

            def _authorized(self) -> bool:
                if server.token is None:
                    return True
                if self.headers.get("Authorization") == f"Bearer {server.token}":
                    return True
                self._problem(Problem(401, "Unauthorized", "bad token"))
                return False

            def do_GET(self) -> None:
                if not self._authorized():
                    return
                if self.path == "/v1/boxes/demo/describe":
                    self._send(200, json.dumps(DESCRIBE).encode(), "application/json")
                else:
                    self._problem(Problem(404, "Not Found", self.path))

            def do_PUT(self) -> None:
                if not self._authorized():
                    return
                body = self.rfile.read(int(self.headers["Content-Length"]))
                digest = self.path.rsplit("/", 1)[-1]
                if hashlib.sha256(body).hexdigest() != digest:
                    self._problem(Problem(400, "Bad Request", "hash mismatch"))
                    return
                server.uploads.append(digest)
                server.store[digest] = body
                self._send(201, b"", "text/plain")

            def do_POST(self) -> None:
                if not self._authorized():
                    return
                body = self.rfile.read(int(self.headers["Content-Length"]))
                message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                    b"Content-Type: "
                    + self.headers["Content-Type"].encode()
                    + b"\r\n\r\n"
                    + body
                )
                parts: dict[str, bytes] = {}
                for p in message.iter_parts():
                    name = p.get_param("name", header="content-disposition")
                    payload = p.get_payload(decode=True)
                    if isinstance(name, str) and isinstance(payload, bytes):
                        parts[name] = payload
                try:
                    reply = server.answer(parts)
                except Problem as problem:
                    self._problem(problem)
                    return
                boundary = "fakeboundary"
                out = b""
                for name, data in reply.items():
                    out += (
                        (
                            f'--{boundary}\r\nContent-Disposition: attachment; name="{name}"\r\n'
                            "Content-Type: application/vnd.apache.arrow.stream\r\n\r\n"
                        ).encode()
                        + data
                        + b"\r\n"
                    )
                out += f"--{boundary}--\r\n".encode()
                self._send(200, out, f"multipart/mixed; boundary={boundary}")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeBoxServer:
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._server.shutdown()
        self._server.server_close()

    # ---- queries ---------------------------------------------------------

    def answer(self, parts: dict[str, bytes]) -> dict[str, bytes]:
        text = parts.pop("query").decode()
        self.requests.append(text)
        options = json.loads(parts.pop("options")) if "options" in parts else {}
        if "derivative" in options:
            self.derivatives.append(options["derivative"])
        data: dict[str, Any] = {}
        missing = []
        seeds: list[bytes] = []
        result = None
        for line in text.strip().splitlines():
            name, expr = (s.strip() for s in line.split("=", 1))
            if expr.startswith("load_module"):
                assert expr == 'load_module("box:demo")', expr
            elif expr.startswith("load_data"):
                match = re.fullmatch(
                    r'load_data\("(part|sha256):([^"]+)", (.*)\)', expr
                )
                assert match, expr
                kind, ref, valueset = match.groups()
                if kind == "sha256" and ref not in self.store:
                    missing.append(ref)
                    continue
                raw = parts[ref] if kind == "part" else self.store[ref]
                data[name] = (_read(raw), valueset)
            elif expr.startswith("rnginit(["):
                seeds = [bytes(int(b) for b in expr[9:-2].split(","))]
            else:
                assert name == "result", line
                result = expr
        if missing:
            raise Problem(404, "Not Found", "unknown hash", missing=missing)
        assert result is not None
        broadcast = result.startswith("broadcast(")
        args = re.fullmatch(r"(?:broadcast\()?([\w.]+)\((.*)\)\)?", result)
        if broadcast:
            inner = result[len("broadcast(") : -1].split(", ")
            callee, rest = inner[0], inner[1:]
        else:
            callee = result[: result.index("(")]
            rest = result[result.index("(") + 1 : -1].split(", ")
        del args
        if callee == "builtin_sample" and broadcast:
            rngs = rest.pop(0)
            assert rngs == "rnginit.(seeds)"
            table, _ = data.pop("seeds")
            seeds = [bytes(row.astype(np.uint8)) for row in _numpy(table.column(0))]
        elif callee == "builtin_sample":
            assert rest.pop(0) == "rng"
        if callee.startswith("B."):
            member, names = callee[2:], rest
        else:
            member, names = rest[0][2:], rest[1:]
        sizes = tuple(int(n) for n in names if n.isdigit())
        names = [n for n in names if not n.isdigit()]
        values = {n: self._value(data[n][0], data[n][1], broadcast) for n in names}
        rows = None
        if broadcast:
            rows = max(
                (len(v["mu"]) if isinstance(v, dict) else len(v))
                for v in values.values()
            )
            values = {
                n: (
                    {k: np.broadcast_to(c, (rows, *c.shape[1:])) for k, c in v.items()}
                    if isinstance(v, dict)
                    else np.broadcast_to(v, (rows, *v.shape[1:]))
                )
                for n, v in values.items()
            }
        return self._evaluate(callee, member, values, sizes, seeds, rows, options, data)

    @staticmethod
    def _value(table: pa.Table, valueset: str, broadcast: bool) -> Any:
        if table.column_names != ["value"]:
            fields = {c: _numpy(table.column(c)) for c in table.column_names}
            return fields if broadcast else {c: v[0] for c, v in fields.items()}
        column = _numpy(table.column(0))
        if broadcast or valueset.startswith("cartpow"):
            return column
        return column[0]

    def _evaluate(
        self,
        callee: str,
        member: str,
        values: dict[str, Any],
        sizes: tuple[int, ...],
        seeds: list[bytes],
        rows: int | None,
        options: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, bytes]:
        derivative = options.get("derivative")
        out: dict[str, bytes] = {}
        grads: dict[str, Any] = {}
        if callee == "builtin_sample":
            draws = []
            for i, seed in enumerate(seeds):
                rng = np.random.Generator(np.random.PCG64(list(seed)))
                if member == "K":
                    p = values["input"]
                    mu = p["mu"][i] if rows else p["mu"]
                    sigma = p["sigma"][i] if rows else p["sigma"]
                    draws.append(rng.normal(mu, sigma, sizes))
                else:
                    draws.append(rng.normal(size=sizes))
            value = np.stack(draws) if rows else draws[0]
            out["value"] = _part(value, "reals")
            out["rngstate"] = _part(np.zeros((len(seeds), 1), dtype=np.int64), "x")
            return out
        if member == "posterior":
            x = values["x"]
            value = posterior_logdensity(x)
            grads = {"x": -(x - 1.0)}
        elif member == "std":
            x = values["x"]
            value = stats.norm.logpdf(x)
            grads = {"x": -x}
        elif member == "K":
            p, x = values["input"], values["x"]
            value = stats.norm.logpdf(x, p["mu"], p["sigma"])
            z = (x - p["mu"]) / p["sigma"]
            grads = {
                "x": -z / p["sigma"],
                "input": {
                    "mu": z / p["sigma"],
                    "sigma": -1 / p["sigma"] + z**2 / p["sigma"],
                },
            }
        elif member == "f":
            x = values["input"]
            value = np.sum(x**2, axis=-1)
            grads = {"input": 2 * x}
        else:
            x = values["input"]
            value = g(x)
            if derivative is not None:
                ct = _numpy(data[derivative["cotangent"]][0].column(0))
                grads = {
                    "input": np.stack(
                        [
                            np.cos(x[..., 0]) * ct[..., 0] + x[..., 1] * ct[..., 1],
                            x[..., 0] * ct[..., 1],
                        ],
                        axis=-1,
                    )
                }
        out["value"] = _part(np.asarray(value), "reals")
        if derivative is not None:
            if derivative["kind"] == "gradient" and member == "g":
                raise Problem(422, "Unprocessable", "no gradient")
            for name in derivative["wrt"]:
                grad = grads[name]
                if isinstance(grad, dict):
                    out[f"d_{name}"] = _record_part(grad)
                else:
                    out[f"d_{name}"] = _part(np.asarray(grad), "reals")
        return out
