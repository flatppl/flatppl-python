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
python -m pip install "pytest>=8" "blackjax==1.7.1" "pandas==3.0.6"
python -m pytest tests -q
```

The tests cover values and first derivatives, input/output structure, context
lifecycle, module imports, random-state chaining, device transfers, and BlackJAX sampling.

## Build the documentation

From the repository root, use a Python 3.14 environment and run:

```sh
python -m pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.
The HTML build needs only the documentation dependencies. It does not import
the native compiler, JAX, or Enzyme.

The site uses [Sphinx](https://www.sphinx-doc.org/),
[MyST Markdown](https://myst-parser.readthedocs.io/), and the
[Furo theme](https://pradyunsg.me/furo/).
Navigation lives in `docs/index.md`. Theme settings live in `docs/conf.py`.

## Check executable examples

In an environment with the installed FlatPPL wheel and documentation dependencies, run:

```sh
sphinx-build -b doctest -W --keep-going docs docs/_build/doctest
```

The `testcode` blocks run in page-local namespaces. Their `testoutput` blocks
check the displayed results. The BlackJAX guide includes `examples/nuts.py`
directly, and the existing package test executes that file.

CI builds the HTML separately and executes documentation examples against the
Linux x86-64 wheels. The package's full runtime tests run on Python 3.12–3.14
for both wheel targets.
After all checks pass on `main`, CI publishes the rendered site to
[GitHub Pages](https://flatppl.org/flatppl-python/).
Pull requests build the site and upload a preview artifact without deploying.
To redeploy, run the **Python wheels** workflow manually on `main`.
