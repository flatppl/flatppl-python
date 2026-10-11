# Development and documentation

The Python repository owns the public Python API, JAX adapter, and wheel packaging.
Reusable source contexts and query exports live in
[`flatppl-host`](https://github.com/flatppl/flatppl-rust/tree/main/crates/host).
The PyO3 adapter lives in
[`flatppl-python-api`](https://github.com/flatppl/flatppl-rust/tree/main/crates/python-api).
Only the native module entry point remains in this repository.

## Run package tests

Build and install the wheel using the [source instructions](installation.md#build-from-source).
Then install the test tools and run the public tests:

```sh
python -m pip install "pytest>=8" "blackjax==1.7.1" "pandas==3.0.6" "polars>=2" "pyhf==0.7.6" "networkx>=3"
python -m pytest tests -q
```

The tests cover values and first derivatives, input/output structure, context
lifecycle, module imports, random-state chaining, device transfers, and BlackJAX sampling.

## Check style and types

Ruff formats and lints the package, tests and examples. Pyright checks types in
strict mode. Install both from the `lint` dependency group into the environment
that holds the installed wheel and its test dependencies:

```sh
python -m pip install --group lint
ruff format --check .
ruff check .
pyright
```

`python/flatppl/_native.pyi` describes the compiled module. A test compares it
with the parameter names and defaults of the installed extension, so update the
stub with every change to the Rust bindings. Stubs in `typings/` declare the
untyped imports as `Any`.

## Build the documentation

The API reference comes from the installed package's docstrings, and the
example gallery runs each script in `examples/`. Install the wheel first, as in
[build from source](installation.md#build-from-source). The corpus example
reads models from a [flatppl-examples](https://github.com/flatppl/flatppl-examples)
checkout. Then run:

```sh
python -m pip install -r docs/requirements.txt
export FLATPPL_EXAMPLES_DIR=/path/to/flatppl-examples/examples
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. Without `FLATPPL_EXAMPLES_DIR`
the corpus page does not run and the build warns.

Each example's docstring states its runtime budget, such as
`Runtime budget: 60 s.`, for the configuration the gallery runs. The build fails
when a page runs longer or states no budget. Write public docstrings in Google
style. A test fails when a public name lacks a docstring or an entry in
`docs/api.md`.

The site uses [Sphinx](https://www.sphinx-doc.org/),
[MyST Markdown](https://myst-parser.readthedocs.io/), and the
[Furo theme](https://pradyunsg.me/furo/).
Navigation lives in `docs/index.md`. Theme settings live in `docs/conf.py`.

## Check executable examples

In an environment with the installed FlatPPL wheel and documentation dependencies, run:

```sh
sphinx-build -b doctest -W --keep-going -D plot_gallery=0 docs docs/_build/doctest
```

The `testcode` blocks run in page-local namespaces. Their `testoutput` blocks
check the displayed results. The doctest builder also runs the examples in the
API docstrings. The BlackJAX guide includes `examples/nuts.py` directly, and
the package tests execute that file too.

CI builds the HTML with the executed gallery from the Linux x86-64 Python 3.14
wheel that the same run built. It executes the documentation examples against
the Linux x86-64 wheels for Python 3.12–3.14. The package's full runtime tests
run on Python 3.12–3.14 for both wheel targets.
After all checks pass on `main`, CI publishes the rendered site to
[GitHub Pages](https://flatppl.org/flatppl-python/).
Pull requests build the site and upload a preview artifact without deploying.
To redeploy, run the **Python wheels** workflow manually on `main`.
