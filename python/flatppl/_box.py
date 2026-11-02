"""The flatppl-bridge client: describe and query black boxes over HTTP."""

from __future__ import annotations

import email.parser
import email.policy
import hashlib
import io
import json
import re
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from typing import Any


class BoxError(RuntimeError):
    """A box server refused or failed a request.

    Attributes:
        status: The HTTP status code: 400 for a malformed request, 404 for
            missing content, 422 for an unsupported query and 500 for a box
            failure.
        title: The short problem title the server sent.
        detail: The server's explanation.
    """

    def __init__(self, status: int, title: str, detail: str) -> None:
        super().__init__(f"box server answered {status} {title}: {detail}")
        self.status = status
        self.title = title
        self.detail = detail


class Box:
    """A black box served over the flatppl-bridge protocol.

    Pass boxes to :meth:`flatppl.Module.compile` to bind a model's interface
    modules. The model never names a server: the binding maps a
    ``load_module`` alias of the model to a box.

    Args:
        url: The server root, such as ``"http://127.0.0.1:8737"``.
        name: The box name on that server.
        token: A bearer token for the ``Authorization`` header, if the server
            requires one.
        timeout: Seconds to wait for one HTTP request.
        upload_threshold: Data parts of at least this many bytes travel once
            to the server's content store and are then referenced by hash.

    Example:
        >>> from flatppl import Box
        >>> Box("http://127.0.0.1:8737", "stan")
        Box('http://127.0.0.1:8737', 'stan')
    """

    def __init__(
        self,
        url: str,
        name: str,
        *,
        token: str | None = None,
        timeout: float = 600.0,
        upload_threshold: int = 1 << 16,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"box name {name!r} is not an identifier")
        self.url = url.rstrip("/")
        self.name = name
        self.token = token
        self.timeout = timeout
        self.upload_threshold = upload_threshold
        self._describe: dict[str, Any] | None = None
        self._stored: set[str] = set()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Box({self.url!r}, {self.name!r})"

    def describe(self) -> Mapping[str, Any]:
        """Return the box's describe reply, fetched once and cached.

        Returns:
            One entry per member with its ``kind``, ``queries``, derivative
            support, ``normalized`` flag and value sets.

        Raises:
            BoxError: The server refused the request.
        """
        if self._describe is None:
            body = self._request("GET", f"/v1/boxes/{self.name}/describe")[1]
            self._describe = json.loads(body)
        return self._describe

    def _describe_json(self) -> str:
        return json.dumps(self.describe())

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> tuple[dict[str, str], bytes]:
        request = urllib.request.Request(self.url + path, data=body, method=method)
        if content_type is not None:
            request.add_header("Content-Type", content_type)
        if self.token is not None:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return dict(reply.headers.items()), reply.read()
        except urllib.error.HTTPError as error:
            raise _problem(error) from None

    def _query(
        self, text: str, parts: Mapping[str, bytes], options: Mapping[str, Any] | None
    ) -> dict[str, bytes]:
        """Send one query and return the reply parts by name.

        Large parts go to the content store first and are referenced by
        their SHA-256 hash. A 404 for an evicted object uploads it again and
        retries once.
        """
        inline: dict[str, bytes] = {}
        stored: dict[str, bytes] = {}
        for name, data in parts.items():
            if len(data) >= self.upload_threshold:
                digest = hashlib.sha256(data).hexdigest()
                text = text.replace(f'"part:{name}"', f'"sha256:{digest}"')
                stored[digest] = data
            else:
                inline[name] = data
        with self._lock:
            pending = [d for d in stored if d not in self._stored]
        for digest in pending:
            self._upload(digest, stored[digest])
        for attempt in range(2):
            try:
                return self._post(text, inline, options)
            except BoxError as error:
                missing = getattr(error, "missing", None)
                if error.status != 404 or not missing or attempt:
                    raise
                for digest in missing:
                    if digest not in stored:
                        raise
                    self._upload(digest, stored[digest])
        raise AssertionError("unreachable")

    def _upload(self, digest: str, data: bytes) -> None:
        self._request(
            "PUT",
            f"/v1/objects/sha256/{digest}",
            data,
            "application/vnd.apache.arrow.stream",
        )
        with self._lock:
            self._stored.add(digest)

    def _post(
        self, text: str, parts: Mapping[str, bytes], options: Mapping[str, Any] | None
    ) -> dict[str, bytes]:
        boundary = uuid.uuid4().hex
        body = io.BytesIO()

        def add(name: str, content_type: str, data: bytes) -> None:
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"\r\n'.encode())
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.write(data)
            body.write(b"\r\n")

        add("query", "text/x-flatppl; charset=utf-8", text.encode())
        if options is not None:
            add("options", "application/json", json.dumps(options).encode())
        for name, data in parts.items():
            add(name, "application/vnd.apache.arrow.stream", data)
        body.write(f"--{boundary}--\r\n".encode())
        headers, reply = self._request(
            "POST",
            "/v1/query",
            body.getvalue(),
            f"multipart/form-data; boundary={boundary}",
        )
        content_type = next(
            (v for k, v in headers.items() if k.lower() == "content-type"), ""
        )
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + reply
        )
        out: dict[str, bytes] = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            payload = part.get_payload(decode=True)
            if isinstance(name, str) and isinstance(payload, bytes):
                out[name] = payload
        return out


def _problem(error: urllib.error.HTTPError) -> BoxError:
    try:
        problem = json.loads(error.read())
    except ValueError:
        problem = {}
    result = BoxError(
        error.code,
        str(problem.get("title", error.reason)),
        str(problem.get("detail", "")),
    )
    if "missing" in problem:
        result.missing = list(problem["missing"])  # pyright: ignore[reportAttributeAccessIssue]
    return result
