Stubs that declare untyped imports as `Any` for pyright. Each module here ships
no `py.typed` and no stubs upstream. Pyright reads this directory through
`stubPath` in `pyproject.toml`. Calls into these modules are not type checked.
