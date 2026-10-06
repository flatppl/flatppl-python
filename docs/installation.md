# Installation

## Supported environment

| Component | Qualified version or platform |
| --- | --- |
| Python | 3.12 |
| JAX and jaxlib | 0.11.2 |
| Enzyme (`enzyme-ad`) | 0.0.15 |
| Linux | x86-64 and ARM64, wheels tagged `manylinux_2_28` |
| macOS | Apple silicon, macOS 26.5 or newer |

The package pins JAX and Enzyme together. The macOS requirement comes from
Enzyme's native binary, despite its upstream wheel tag claiming macOS 11 support.
This Enzyme release has no Intel macOS wheel. Windows is not qualified.

The package's CI tests CPU execution on all three wheel targets.
GPU execution remains unqualified. See [JAX and differentiation](jax.md).

## Install a wheel

Wheels are currently distributed through GitHub Actions artifacts.
There is no published PyPI release yet.

1. Open the [Python wheels workflow](https://github.com/flatppl/flatppl-python/actions/workflows/test.yml).
2. Select a successful run on `main`.
3. Download the artifact for your platform while signed into GitHub.
4. Extract the artifact, which contains `dist/` and `constraints.txt`.
5. Run these commands from the extracted directory.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -c constraints.txt dist/*.whl
```

| Platform | Artifact |
| --- | --- |
| Linux x86-64 | `flatppl-python-ubuntu-24.04` |
| Linux ARM64 | `flatppl-python-ubuntu-24.04-arm` |
| macOS ARM64 | `flatppl-python-macos-26` |

Keep the constraints file beside the wheel. Enzyme 0.0.15's package index
incorrectly labels its Python 3.12 wheels as requiring Python 3.11.
The constraints select the official wheels by platform, URL, and hash.

The FlatPPL wheel contains the native compiler. Using it requires neither the
Rust toolchain nor the FlatPPL CLI.

## Check the installation

```{testcode}
from flatppl import flatppl

evaluate = flatppl("outputs = 2.0 + 3.0").compile()
print(float(evaluate()))
```

```{testoutput}
5.0
```

Continue with the [quickstart](quickstart.md).

## Install BlackJAX for the example

After installing FlatPPL, install the version used by the package's tests:

```sh
python -m pip install -c constraints.txt "blackjax==1.7.1"
```

BlackJAX is optional. The [sampling guide](blackjax.md) shows the complete example.

## Build from source

Install a Rust toolchain and Python 3.12, then run:

```sh
git clone https://github.com/flatppl/flatppl-python.git
cd flatppl-python
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install "maturin==1.15.0"
maturin build --release --locked --interpreter python --out dist
python -m pip install -c constraints.txt dist/*.whl
```

Cargo fetches the exact Rust compiler revision pinned in `Cargo.toml` and
`Cargo.lock`. See [development](development.md) for checks and documentation builds.
