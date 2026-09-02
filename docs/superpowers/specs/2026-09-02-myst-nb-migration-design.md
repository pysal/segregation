# myst-nb migration for the segregation docs (PR #279)

**Date:** 2026-09-02
**Branch:** `joss_paper` (feeds PR pysal/segregation#279)
**Status:** design approved, not yet implemented

## Context

PR #279 addresses a JOSS review (openjournals/joss-reviews#11126). An earlier
iteration added `nbmake` as a notebook-testing dependency plus a dedicated
`.github/workflows/notebooks.yml`. In the PR discussion
(comments 5501393929 / 5501470385 / 5501525799 / 5502274798) reviewer
@knaaptime asked us instead to migrate the docs to **`myst-nb`** (the pattern
used by `pysal/tobler`). With `myst-nb`, notebooks execute as part of the
Sphinx build, so:

- a reviewer only needs to clone and run `cd docs && make html`;
- notebook breakage fails the docs build (same guarantee `nbmake` gave);
- no new dependency is adopted purely for reviewer convenience.

The **theme migration** away from `sphinx_bootstrap_theme` is explicitly a
separate follow-up and is **out of scope** here.

## Goals

1. Notebooks execute during the Sphinx build and a failure fails the build.
2. One canonical copy of each example notebook.
3. `nbmake` and its dedicated workflow are removed.
4. Notebook execution is verified on every pull request through existing
   infrastructure (the docs build), not a bespoke workflow.
5. Docs instructions let a fresh clone build (and thereby run) the notebooks.

## Non-goals

- Theme migration (`sphinx_bootstrap_theme` → `sphinx_immaterial` or other).
- Rewriting notebook content (the self-contained rewrites already in this PR
  stand as-is).
- Read the Docs configuration (the project publishes via the `gh-pages`
  workflow; no `.readthedocs.yaml` exists).

## Design

### 1. Notebook location — consolidate into `docs/notebooks/`

Current state: the maintained set lives in root `notebooks/` (8 notebooks,
rewritten earlier in this PR); `docs/notebooks/` is a stale 6-notebook
duplicate with old numbering (`05_inference_example`,
`06_decomposition_example`) and megabytes of committed outputs.

Action:

- `git rm` the stale `docs/notebooks/*.ipynb`.
- `git mv` the maintained notebooks from `notebooks/` to `docs/notebooks/`:
  - `01_singlegroup_indices.ipynb`
  - `02_multigroup_indices.ipynb`
  - `03_local_indices.ipynb`
  - `04_multiscalar_example.ipynb`
  - `05_simulating_random_population.ipynb`
  - `06_inference.ipynb`
  - `07_decomposition_example.ipynb`
  - `kl_divergence_profile_walkthrough.ipynb`
- Remove the now-empty root `notebooks/` directory (and its
  `anaconda_projects/` cruft if still present).
- The repo has a single notebooks directory afterwards: `docs/notebooks/`.

### 2. `docs/conf.py` — swap `nbsphinx` for `myst_nb`

- In `extensions`: remove `"nbsphinx"`, add `"myst_nb"`.
- Extend `source_suffix` so notebooks are recognised:

  ```python
  source_suffix = {
      ".rst": "restructuredtext",
      ".md": "myst-nb",
      ".ipynb": "myst-nb",
  }
  ```

- Add the execution / MyST configuration (mirrors `pysal/tobler`):

  ```python
  nb_execution_mode = "cache"
  nb_execution_raise_on_error = True
  nb_execution_timeout = -1
  nb_execution_show_tb = True
  nb_merge_streams = True
  nb_kernel_rgx_aliases = {".*": "python3"}
  myst_enable_extensions = [
      "amsmath",
      "colon_fence",
      "deflist",
      "dollarmath",
      "html_image",
  ]
  ```

- `nb_execution_mode = "cache"`: notebooks execute on first build and are
  re-executed only when their content changes; the cache lives under
  `_build/` (already git-ignored via the `build` entry, plus an explicit
  `docs/_build/` ignore for clarity).
- `nb_execution_raise_on_error = True`: any notebook error fails `make html`.
- `nb_kernel_rgx_aliases`: maps any kernelspec name to `python3`, so no
  per-notebook metadata editing is needed.
- Cells tagged `skip-execution` (network-distance cells in notebooks 01, 02,
  04 that need optional `pandarm` / `hvplot`) are skipped by `myst-nb`, so the
  docs build does not require those optional dependencies.
- Theme settings, bibtex, intersphinx, autosummary config: unchanged.

### 3. `docs/tutorial.rst` — fix the toctree

Replace the stale 6-entry list with the current sequence:

```rst
.. toctree::
    :maxdepth: 1
    :caption: Contents:

    notebooks/01_singlegroup_indices
    notebooks/02_multigroup_indices
    notebooks/03_local_indices
    notebooks/04_multiscalar_example
    notebooks/05_simulating_random_population
    notebooks/06_inference
    notebooks/07_decomposition_example
    notebooks/kl_divergence_profile_walkthrough
```

(`kl_divergence_profile_walkthrough` included so every notebook under
`docs/notebooks/` is in a toctree and Sphinx raises no "not included in any
toctree" warning; drop it from the list only if we also move it out of
`docs/notebooks/`.)

### 4. CI

**Delete:**
- `.github/workflows/notebooks.yml`
- `ci/notebooks.yaml`

**`.github/workflows/build_docs.yml`:** run the build on PRs too, but only
publish on push to `main`.

```yaml
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
      - joss_paper
```

Gate the publish steps:

```yaml
      - name: Commit Docs
        if: github.event_name == 'push'
        run: |
          ...
      - name: Push to gh-pages
        if: github.event_name == 'push'
        uses: ad-m/github-push-action@master
        ...
```

Net effect: on a PR, the job runs `cd docs && make html`, which executes the
notebooks and fails on error; nothing is pushed to `gh-pages`.

### 5. Dependency lists

**`ci/314-latest.yaml`** (used by `build_docs.yml`):
- remove `nbsphinx`
- add `myst-nb` and `jupyter-cache`

**`environment.yml`** (the documented dev environment, `installation.rst`
item iv): add the docs toolchain so `conda env create -f environment.yml`
→ `pip install -e .` → `cd docs && make html` works from a clean clone:
- `sphinx`
- `myst-nb`
- `jupyter-cache`
- `numpydoc`
- `sphinxcontrib-bibtex`
- `sphinx_bootstrap_theme`
- `ipywidgets`
- `ipykernel`

(`matplotlib`, `seaborn`, `mapclassify`, `libpysal` etc. are already present
and are what the notebooks import.)

### 6. Docs: how to run the notebooks

**`docs/installation.rst`** — add a short "Building the documentation"
section after the install options:

```rst
Building the documentation
==========================

The example notebooks are executed when the documentation is built, so
building the docs also runs every notebook::

    conda env create -f environment.yml
    conda activate segregation
    pip install -e .
    cd docs
    make html

To execute the notebooks on their own::

    jupyter nbconvert --execute --to notebook --inplace docs/notebooks/*.ipynb
```

**`README.md`** — repoint notebook links to `docs/notebooks/` and fix the
broken targets:
- `notebooks/10_singlegroup_indices.ipynb` → `docs/notebooks/01_singlegroup_indices.ipynb` (2 occurrences)
- `notebooks/02_multigroup_indices.ipynb` → `docs/notebooks/02_multigroup_indices.ipynb`
- `notebooks/03_local_indices.ipynb` → `docs/notebooks/03_local_indices.ipynb`
- `notebooks/04_multiscalar_example.ipynb` → `docs/notebooks/04_multiscalar_example.ipynb`
- `notebooks/kl_divergence_profile_walkthrough.ipynb` → `docs/notebooks/kl_divergence_profile_walkthrough.ipynb`
- `notebooks/06_inference.ipynb` → `docs/notebooks/06_inference.ipynb`
- `notebooks/07_decomposition_example.ipynb` → `docs/notebooks/07_decomposition_example.ipynb`
- `notebooks/inference_wrappers_example.ipynb` → `docs/notebooks/06_inference.ipynb` (2 occurrences)
- `notebooks/decomposition_wrapper_example.ipynb` → `docs/notebooks/07_decomposition_example.ipynb`
- `tree/master/notebooks` → `tree/master/docs/notebooks`

**`docs/index.rst`** — the raw-HTML thumbnail grid links `.../blob/master/notebooks/*`
become `.../blob/master/docs/notebooks/*`; fix `06_inference` / `07_decomposition_example`
names there too.

**`paper/paper.md`** — repoint any `notebooks/` links to `docs/notebooks/`.

### 7. `.gitignore`

Add `docs/_build/` (currently only the generic `build` entry exists).
`myst-nb` stores the jupyter-cache under `docs/_build/.jupyter_cache/`, so the
single `docs/_build/` ignore covers it.

## Verification

Local build is not possible on this machine (no scientific-Python env). The
CI docs job (`build_docs.yml` on the PR) is the source of truth:

1. `build_docs.yml` PR run completes green — proves all 8 notebooks execute
   under `myst-nb` with `nb_execution_raise_on_error = True`.
2. No stray Sphinx "document isn't included in any toctree" warnings for the
   notebooks.
3. `notebooks.yml` no longer appears in the Actions list.
4. Rendered tutorial pages on the PR's gh-pages preview (if available) or on
   the next `main` publish show executed cell outputs.

## Risks / open questions

- `sphinx_bootstrap_theme` is old and may render `myst-nb` output less
  cleanly than a modern theme; acceptable because the theme swap is the
  agreed next step.
- `nb_execution_timeout = -1` (no timeout) means a hung notebook hangs the
  job until the workflow's `timeout-minutes: 90`. Acceptable; can add a
  finite `nb_execution_timeout` (e.g. 900) if builds get slow.
- Notebook 04 is large (~3.7 MB with embedded outputs). Keeping committed
  outputs for `skip-execution` cells is required so those cells still render.
