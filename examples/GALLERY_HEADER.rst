Worked tutorials
================

Start with :doc:`/quickstart`, then follow a complete workflow below.
Each page combines the explanation, runnable code and output. Download its
Python script or Jupyter notebook from the bottom of the page.

* **New to inference?** Start with the Normal model and its known posterior.
* **Have a dataframe?** Fit a mixed model with automatic coordinate transforms.
* **Have arrays?** Pass runtime data to the Gaussian regression.
* **Want a portable analysis?** Save a regression to ZIP, reload it, and fit MAP.
* **Using an inference package?** Sample and approximate with BlackJAX, sample
  with NumPyro, fit with Optax, and diagnose with ArviZ, all from
  ``module.logdensity(...)``.
* **Fitting or sampling outside JAX?** Read Hesse and Minos errors with
  iminuit, minimize with optimistix, or sample with nutpie.
* **Have a FlatPPL model?** Query its bindings, inspect its graph, or run a file
  from the example corpus.

For a pyhf workspace, use :doc:`/pyhf`. The :doc:`/transforms` reference explains
how to pass constrained parameters to an unconstrained sampler.

The documentation build executes these scripts against the installed package.
Each script states its runtime budget; the build fails if it exceeds that limit.
