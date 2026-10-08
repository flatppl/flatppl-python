# Installation

## Supported environment

| Component | Qualified version or platform |
| --- | --- |
| Python | 3.12, 3.13, 3.14 |
| JAX and jaxlib | 0.11.2 |
| Enzyme (`enzyme-ad`) | 0.0.15+flatppl.3, FlatPPL fork |
| Linux x86-64 | glibc 2.35 or newer, such as Ubuntu 22.04 |
| Linux ARM64 | glibc 2.28 or newer |
| macOS | Apple silicon, macOS 14 or newer |

The package pins JAX and Enzyme together. Enzyme wheels come from the
[FlatPPL fork](https://github.com/BJMCox/Enzyme-JAX/releases/tag/flatppl-alpha.3)
and include shared forward work for values and gradients, stable real
reciprocal-square-root derivatives, batching rules, and sharding support.
These are alpha builds. Intel macOS and Windows are not packaged.
Package metadata selects the Enzyme wheel by Python version and platform, with
a pinned URL and SHA-256 hash. No separate constraints file is needed.

CI tests each supported Python version on Linux x86-64 and macOS ARM64.
Linux ARM64 wheels are built and checked on an ARM server for each release.
GPU checks use two NVIDIA A100s with CUDA 12 and a GH200 with CUDA 13, both
using Python 3.14.
See [JAX and differentiation](jax.md).

## Install from GitHub with Pixi

Install [Pixi](https://pixi.prefix.dev/latest/installation/) and Rust 1.96 or newer.
Install the platform build tools: Xcode Command Line Tools on macOS, or a C
compiler and linker on Linux.

Create a project and install FlatPPL from its Git repository:

```sh
pixi init flatppl-demo
cd flatppl-demo
pixi add "python=3.14"
pixi add --pypi "flatppl @ git+https://github.com/flatppl/flatppl-python.git"
```

Pixi builds the native FlatPPL extension and installs JAX, NumPy, and the matching
Enzyme wheel automatically. Enzyme itself needs no local build.
Pixi records the Git commit and dependency versions in `pixi.lock`.
Use `pixi run python` to run Python in the project environment.

For the sampling example, add BlackJAX separately:

```sh
pixi add --pypi "blackjax==1.7.1"
```

## Install a wheel

Wheels are currently distributed through GitHub Actions artifacts.
There is no published PyPI release yet.

1. Open the [Python wheels workflow](https://github.com/flatppl/flatppl-python/actions/workflows/test.yml).
2. Select a successful run on `main`.
3. Download the artifact for your platform and Python version while signed into GitHub.
4. Extract the artifact, which contains `dist/` and `requirements.txt`.
5. Run these commands from the extracted directory.

```sh
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

| Platform | Artifact |
| --- | --- |
| Linux x86-64 | `flatppl-python-ubuntu-24.04-py3.14` |
| macOS ARM64 | `flatppl-python-macos-26-py3.14` |

The table shows Python 3.14 artifacts. Choose the corresponding `py3.12` or
`py3.13` artifact when using either earlier supported version.

Keep the extracted files together. The requirements file selects the local
FlatPPL wheel, whose metadata supplies all runtime dependencies.

The FlatPPL wheel contains the native compiler. Using it requires neither the
Rust toolchain nor the FlatPPL CLI.

Linux ARM64 users can install from GitHub with Pixi or build from source.
The adapter installs a prebuilt Enzyme wheel on that platform too.

The A100 checks use JAX's CUDA 12 runtime:

```sh
python -m pip install "jax[cuda12]==0.11.2"
```

The GH200 ARM64 checks use CUDA 13:

```sh
python -m pip install "jax[cuda13]==0.11.2"
```

Follow the [JAX driver requirements](https://docs.jax.dev/en/latest/installation.html)
for your GPU. Qualification used driver 570.124.06 on the A100s and
595.71.05 on the GH200.

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
python -m pip install "blackjax==1.7.1"
```

BlackJAX is optional. The [sampling guide](blackjax.md) shows the complete example.

## Build from source

Install Rust 1.96 or newer, the platform build tools, and Python 3.14, then run:

```sh
git clone https://github.com/flatppl/flatppl-python.git
cd flatppl-python
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install "maturin==1.15.0"
maturin build --release --locked --interpreter python --out dist
python -m pip install -r requirements.txt
```

Cargo fetches the exact Rust compiler revision pinned in `Cargo.toml` and
`Cargo.lock`. See [development](development.md) for checks and documentation builds.
