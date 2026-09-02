# myst-nb Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the segregation docs from `nbsphinx` + a bespoke `nbmake` workflow to `myst-nb`, so example notebooks execute during the Sphinx build and any failure fails CI.

**Architecture:** Consolidate the maintained notebooks into a single `docs/notebooks/` directory, switch `docs/conf.py` to the `myst_nb` extension with `pysal/tobler`'s execution settings, delete the `nbmake` workflow, and make `build_docs.yml` run (without publishing) on pull requests so notebook execution is verified per-PR.

**Tech Stack:** Sphinx, `myst-nb`, `jupyter-cache`, `sphinx_bootstrap_theme` (unchanged), GitHub Actions, micromamba.

**Spec:** `docs/superpowers/specs/2026-09-02-myst-nb-migration-design.md`

## Global Constraints

- **Do not run `git commit`, `git add`, `git push`, or any history-changing git command.** The user reviews the full working-tree diff and commits it themselves. Where a task says "checkpoint", stop and report the list of changed/created/deleted files so the user can review `git diff`.
- `git mv` / `git rm` **are allowed** (they only stage moves/deletions and are the correct way to preserve rename history); they do not create commits.
- Work happens on the `joss_paper` branch (already checked out, tracks `origin/joss_paper`).
- Theme migration away from `sphinx_bootstrap_theme` is **out of scope**.
- No local scientific-Python / Sphinx environment exists on this machine. Local verification is limited to static checks (file existence, `python -m json.tool` on notebooks, `grep` for stale references). **Authoritative verification is the `build_docs.yml` run on the PR** (Task 6).
- Notebook execution config must match the spec verbatim:
  - `nb_execution_mode = "cache"`
  - `nb_execution_raise_on_error = True`
  - `nb_execution_timeout = -1`
  - `nb_execution_show_tb = True`
  - `nb_merge_streams = True`
  - `nb_kernel_rgx_aliases = {".*": "python3"}`
- Canonical notebook set (8 files), final location `docs/notebooks/`:
  `01_singlegroup_indices`, `02_multigroup_indices`, `03_local_indices`,
  `04_multiscalar_example`, `05_simulating_random_population`,
  `06_inference`, `07_decomposition_example`,
  `kl_divergence_profile_walkthrough`.

---

## File Structure

| Path | Change | Responsibility |
| --- | --- | --- |
| `docs/notebooks/*.ipynb` | replace 6 stale files with the 8 canonical ones | the single source of truth for example notebooks; executed by the docs build |
| `notebooks/` (repo root) | delete | removed; no longer a notebooks location |
| `docs/conf.py` | modify | Sphinx config: `myst_nb` extension + execution settings |
| `docs/tutorial.rst` | modify | toctree listing every notebook |
| `ci/314-latest.yaml` | modify | docs-build conda env (used by `build_docs.yml`) |
| `environment.yml` | modify | documented dev env; must be able to build docs |
| `.github/workflows/notebooks.yml` | delete | the `nbmake` workflow, superseded |
| `ci/notebooks.yaml` | delete | conda env for the `nbmake` workflow |
| `.github/workflows/build_docs.yml` | modify | also build (not publish) on PRs |
| `docs/installation.rst` | modify | add "Building the documentation" section |
| `README.md` | modify | repoint / fix notebook links |
| `docs/index.rst` | modify | repoint thumbnail-grid notebook links |
| `.gitignore` | modify | ignore `docs/_build/` |

---

## Task 1: Consolidate notebooks into `docs/notebooks/`

**Files:**
- Delete: `docs/notebooks/05_inference_example.ipynb`, `docs/notebooks/06_decomposition_example.ipynb`
- Overwrite (via move): `docs/notebooks/01_singlegroup_indices.ipynb`, `02_multigroup_indices.ipynb`, `03_local_indices.ipynb`, `04_multiscalar_example.ipynb`
- Create (via move): `docs/notebooks/05_simulating_random_population.ipynb`, `06_inference.ipynb`, `07_decomposition_example.ipynb`, `kl_divergence_profile_walkthrough.ipynb`
- Delete: `notebooks/` directory (repo root), all 8 tracked `.ipynb`

**Interfaces:**
- Produces: the 8 canonical notebook files at `docs/notebooks/<name>.ipynb`. Tasks 2 and 3 reference these paths.

- [ ] **Step 1: Remove the two stale docs notebooks that have no canonical counterpart**

```bash
git rm docs/notebooks/05_inference_example.ipynb docs/notebooks/06_decomposition_example.ipynb
```

- [ ] **Step 2: Move the four notebooks that keep their name (overwrites the stale docs copies)**

```bash
git mv -f notebooks/01_singlegroup_indices.ipynb docs/notebooks/01_singlegroup_indices.ipynb
git mv -f notebooks/02_multigroup_indices.ipynb docs/notebooks/02_multigroup_indices.ipynb
git mv -f notebooks/03_local_indices.ipynb docs/notebooks/03_local_indices.ipynb
git mv -f notebooks/04_multiscalar_example.ipynb docs/notebooks/04_multiscalar_example.ipynb
```

- [ ] **Step 3: Move the four remaining notebooks (new paths in `docs/notebooks/`)**

```bash
git mv notebooks/05_simulating_random_population.ipynb docs/notebooks/05_simulating_random_population.ipynb
git mv notebooks/06_inference.ipynb docs/notebooks/06_inference.ipynb
git mv notebooks/07_decomposition_example.ipynb docs/notebooks/07_decomposition_example.ipynb
git mv notebooks/kl_divergence_profile_walkthrough.ipynb docs/notebooks/kl_divergence_profile_walkthrough.ipynb
```

- [ ] **Step 4: Remove the now-empty root `notebooks/` directory**

```bash
rmdir notebooks 2>/dev/null || true
ls notebooks 2>/dev/null && echo "STILL EXISTS - investigate" || echo "notebooks/ removed"
```

If `notebooks/` still exists, list its contents (`git status --porcelain notebooks/` and `ls -la notebooks/`) and remove any leftover untracked cruft (e.g. `anaconda_projects/`) with `rm -rf notebooks/`.

- [ ] **Step 5: Verify the canonical set is present and valid JSON**

```bash
ls docs/notebooks/
for f in docs/notebooks/*.ipynb; do python -m json.tool "$f" > /dev/null && echo "OK  $f" || echo "BAD $f"; done
```

Expected: exactly these 8 files, each `OK`:
`01_singlegroup_indices.ipynb 02_multigroup_indices.ipynb 03_local_indices.ipynb 04_multiscalar_example.ipynb 05_simulating_random_population.ipynb 06_inference.ipynb 07_decomposition_example.ipynb kl_divergence_profile_walkthrough.ipynb`

- [ ] **Step 6: Verify no other tracked file still points at the old root path**

```bash
git grep -n "blob/master/notebooks/\|tree/master/notebooks\|notebooks/0\|(\.\./notebooks\|\"notebooks/" -- . ':!docs/superpowers' || echo "no stale refs"
```

Expected: hits only in `README.md` and `docs/index.rst` (fixed in Task 5). Note them; do not fix here.

- [ ] **Step 7: Checkpoint**

Report: files deleted (2 stale + root `notebooks/` tree), files moved (8), and the `git status --porcelain` output. Wait for user review.

---

## Task 2: Switch Sphinx to `myst-nb` (`docs/conf.py`)

**Files:**
- Modify: `docs/conf.py` (extensions list ~line 31-44; `source_suffix` ~line 63)

**Interfaces:**
- Consumes: notebook files at `docs/notebooks/*.ipynb` (Task 1).
- Produces: a Sphinx config where `.ipynb` files are parsed and executed by `myst_nb`. Task 3's toctree relies on `.ipynb` being a recognized source suffix.

- [ ] **Step 1: Replace `"nbsphinx"` with `"myst_nb"` in the `extensions` list**

In `docs/conf.py`, the `extensions` list currently ends with:

```python
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
]
```

Change to:

```python
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
]
```

- [ ] **Step 2: Update `source_suffix` to register notebooks**

Replace:

```python
source_suffix = [".rst", ".md"]
```

with:

```python
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
```

- [ ] **Step 3: Add the MyST / execution configuration**

Immediately after the `source_suffix` block, add:

```python
# -- MyST / myst-nb configuration --------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Execute notebooks on build; re-run only when their content changes.
nb_execution_mode = "cache"
# Fail the build if any notebook raises.
nb_execution_raise_on_error = True
# No per-cell timeout (the CI job has its own overall timeout).
nb_execution_timeout = -1
nb_execution_show_tb = True
nb_merge_streams = True
# Map any kernelspec name to the environment's python3 kernel.
nb_kernel_rgx_aliases = {".*": "python3"}
```

- [ ] **Step 4: Verify `conf.py` is valid Python and no `nbsphinx` reference remains**

```bash
python -m py_compile docs/conf.py && echo "conf.py compiles"
grep -n "nbsphinx" docs/conf.py && echo "STALE nbsphinx ref" || echo "no nbsphinx ref"
grep -n "myst_nb\|nb_execution_mode\|nb_kernel_rgx_aliases" docs/conf.py
```

Expected: compiles; no `nbsphinx` ref; the three greps in the last line all match.

- [ ] **Step 5: Checkpoint**

Report the `git diff docs/conf.py`. Wait for user review.

---

## Task 3: Fix the tutorial toctree (`docs/tutorial.rst`)

**Files:**
- Modify: `docs/tutorial.rst` (entire toctree body)

**Interfaces:**
- Consumes: notebook files at `docs/notebooks/*.ipynb` (Task 1); `.ipynb` source suffix (Task 2).
- Produces: every notebook is in a toctree (no "not included in any toctree" Sphinx warning).

- [ ] **Step 1: Replace the toctree body**

Set `docs/tutorial.rst` to:

```rst
Segregation Tutorial
======================

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

- [ ] **Step 2: Verify every toctree entry resolves to a file**

```bash
for n in 01_singlegroup_indices 02_multigroup_indices 03_local_indices 04_multiscalar_example 05_simulating_random_population 06_inference 07_decomposition_example kl_divergence_profile_walkthrough; do
  test -f "docs/notebooks/$n.ipynb" && echo "OK  $n" || echo "MISSING $n"
done
```

Expected: all 8 `OK`.

- [ ] **Step 3: Checkpoint**

Report `git diff docs/tutorial.rst`. Wait for user review.

---

## Task 4: Update dependency lists

**Files:**
- Modify: `ci/314-latest.yaml` (docs deps block, ~lines 32-41)
- Modify: `environment.yml` (dependencies list)

**Interfaces:**
- Produces: conda environments that contain `myst-nb` + `jupyter-cache` and not `nbsphinx`. Task 6's CI run consumes `ci/314-latest.yaml`.

- [ ] **Step 1: Edit `ci/314-latest.yaml`**

In the `# docs` block, remove the `- nbsphinx` line and add `- myst-nb` and `- jupyter-cache`. Resulting block (keep alphabetical-ish order, matching the file's style):

```yaml
  # docs
  - ipywidgets
  - jupyter-cache
  - myst-nb
  - numpydoc
  - quilt3
  - sphinx
  - sphinxcontrib-napoleon
  - sphinx-gallery
  - sphinxcontrib-bibtex
  - sphinx_bootstrap_theme
  - watermark
```

- [ ] **Step 2: Edit `environment.yml`**

The current `dependencies:` list ends at `- tqdm` with no trailing newline. Append the docs toolchain so a clean `conda env create -f environment.yml` can build the docs. Final `dependencies:` list:

```yaml
dependencies:
  - python >=3.12
  - geopandas
  - joblib
  - libpysal
  - mapclassify
  - matplotlib
  - numba
  - numpy
  - pandas
  - pip
  - pyproj >=3
  - scikit-learn
  - scipy
  - seaborn
  - tqdm
  # docs (building the docs also executes the example notebooks)
  - sphinx
  - myst-nb
  - jupyter-cache
  - ipykernel
  - ipywidgets
  - numpydoc
  - sphinxcontrib-bibtex
  - sphinx_bootstrap_theme
```

Ensure the file ends with a newline.

- [ ] **Step 3: Verify YAML parses and content is right**

```bash
python -c "import json,subprocess" # noop guard
python - <<'PY'
import sys
try:
    import yaml
except ModuleNotFoundError:
    print("pyyaml not installed - skipping structural parse, doing text checks only")
    sys.exit(0)
for path in ("ci/314-latest.yaml", "environment.yml"):
    d = yaml.safe_load(open(path))
    deps = [x for x in d["dependencies"] if isinstance(x, str)]
    assert not any("nbsphinx" in x for x in deps), (path, "still has nbsphinx")
    assert any(x.startswith("myst-nb") for x in deps), (path, "missing myst-nb")
    print("OK", path)
PY
grep -n "nbsphinx" ci/314-latest.yaml environment.yml && echo "STALE nbsphinx" || echo "no nbsphinx"
grep -n "myst-nb\|jupyter-cache" ci/314-latest.yaml environment.yml
```

Expected: no `nbsphinx`; `myst-nb` present in both files; `jupyter-cache` present in both.

- [ ] **Step 4: Checkpoint**

Report `git diff ci/314-latest.yaml environment.yml`. Wait for user review.

---

## Task 5: CI workflows

**Files:**
- Delete: `.github/workflows/notebooks.yml`
- Delete: `ci/notebooks.yaml`
- Modify: `.github/workflows/build_docs.yml`

**Interfaces:**
- Consumes: `ci/314-latest.yaml` with `myst-nb` (Task 4); notebooks at `docs/notebooks/` (Task 1).
- Produces: `build_docs.yml` runs `cd docs; make html` on PRs targeting `main` or `joss_paper`, without touching `gh-pages`.

- [ ] **Step 1: Delete the nbmake workflow and its env**

```bash
git rm .github/workflows/notebooks.yml ci/notebooks.yaml
```

- [ ] **Step 2: Add a `pull_request` trigger to `build_docs.yml`**

Replace the `on:` block (currently):

```yaml
 on:
   push:
     branches:
     - main
```

with:

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

(Preserve the existing leading-space indentation style of this file — every line in it is indented one extra space.)

- [ ] **Step 3: Gate the publish steps to push events**

Add `if: github.event_name == 'push'` to the `Commit Docs` and `Push to gh-pages` steps:

```yaml
       - name: Commit Docs
         if: github.event_name == 'push'
         run: |
           git clone https://github.com/ammaraskar/sphinx-action-test.git --branch gh-pages --single-branch gh-pages
           cp -r docs/_build/html/* gh-pages/
           cd gh-pages
           git config --local user.email "action@github.com"
           git config --local user.name "GitHub Action"
           git add .
           git commit -m "Update documentation" -a || true
           # The above command will fail if no changes were present,
           # so we ignore the return code.

       - name: Push to gh-pages
         if: github.event_name == 'push'
         uses: ad-m/github-push-action@master
         with:
            branch: gh-pages
            directory: gh-pages
            github_token: ${{ secrets.GITHUB_TOKEN }}
            force: true
```

- [ ] **Step 4: Verify the workflow file**

```bash
python - <<'PY'
import sys
try:
    import yaml
except ModuleNotFoundError:
    print("pyyaml missing - text checks only"); sys.exit(0)
d = yaml.safe_load(open(".github/workflows/build_docs.yml"))
on = d[True] if True in d else d.get("on")
assert "pull_request" in on, on
steps = d["jobs"]["docs"]["steps"]
gated = [s for s in steps if s.get("name") in ("Commit Docs", "Push to gh-pages")]
assert len(gated) == 2 and all(s.get("if") == "github.event_name == 'push'" for s in gated), gated
print("OK build_docs.yml")
PY
test ! -e .github/workflows/notebooks.yml && test ! -e ci/notebooks.yaml && echo "nbmake infra removed"
grep -rn "nbmake" .github ci || echo "no nbmake references anywhere"
```

Expected: `OK build_docs.yml`; `nbmake infra removed`; `no nbmake references anywhere`.

- [ ] **Step 5: Checkpoint**

Report the two deletions and `git diff .github/workflows/build_docs.yml`. Wait for user review.

---

## Task 6: Docs prose, link fixes, `.gitignore`

**Files:**
- Modify: `docs/installation.rst`
- Modify: `README.md`
- Modify: `docs/index.rst`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: notebook final paths `docs/notebooks/*` (Task 1); the `make html` build workflow (Task 5).
- Produces: user-facing instructions and correct links. No downstream task depends on this.

- [ ] **Step 1: Add "Building the documentation" to `docs/installation.rst`**

Append to the end of the file:

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

(Use tab indentation for the literal blocks, matching the rest of the file.)

- [ ] **Step 2: Fix notebook links in `README.md`**

Apply these exact replacements (each is a substring of a URL):

| Find | Replace with |
| --- | --- |
| `blob/master/notebooks/10_singlegroup_indices.ipynb` | `blob/master/docs/notebooks/01_singlegroup_indices.ipynb` |
| `blob/master/notebooks/02_multigroup_indices.ipynb` | `blob/master/docs/notebooks/02_multigroup_indices.ipynb` |
| `blob/master/notebooks/03_local_indices.ipynb` | `blob/master/docs/notebooks/03_local_indices.ipynb` |
| `blob/master/notebooks/04_multiscalar_example.ipynb` | `blob/master/docs/notebooks/04_multiscalar_example.ipynb` |
| `blob/master/notebooks/kl_divergence_profile_walkthrough.ipynb` | `blob/master/docs/notebooks/kl_divergence_profile_walkthrough.ipynb` |
| `blob/master/notebooks/06_inference.ipynb` | `blob/master/docs/notebooks/06_inference.ipynb` |
| `blob/master/notebooks/07_decomposition_example.ipynb` | `blob/master/docs/notebooks/07_decomposition_example.ipynb` |
| `blob/master/notebooks/inference_wrappers_example.ipynb` | `blob/master/docs/notebooks/06_inference.ipynb` |
| `blob/master/notebooks/decomposition_wrapper_example.ipynb` | `blob/master/docs/notebooks/07_decomposition_example.ipynb` |
| `tree/master/notebooks` | `tree/master/docs/notebooks` |

The `10_singlegroup_indices` and `inference_wrappers_example` strings each appear **twice** — replace all occurrences (`sed -i` global, or editor "replace all").

- [ ] **Step 3: Fix notebook links in `docs/index.rst`**

In the raw-HTML thumbnail grid, replace every `blob/master/notebooks/` with `blob/master/docs/notebooks/` (6 occurrences, lines ~14-57). The notebook basenames there are already correct.

- [ ] **Step 4: Add `docs/_build/` to `.gitignore`**

Append `docs/_build/` under the `# Packages` section (or anywhere sensible). `myst-nb` stores its jupyter-cache under `docs/_build/.jupyter_cache/`, so this one entry covers it.

- [ ] **Step 5: Verify no stale notebook references remain anywhere in tracked files**

```bash
git grep -n "master/notebooks/\|tree/master/notebooks\|10_singlegroup\|inference_wrappers_example\|decomposition_wrapper_example" -- . ':!docs/superpowers' || echo "ALL CLEAN"
grep -n "_build" .gitignore
grep -n "Building the documentation" docs/installation.rst
```

Expected: `ALL CLEAN`; `.gitignore` shows a `docs/_build/` entry; the installation heading is present.

- [ ] **Step 6: Checkpoint**

Report `git diff` for the four files. Wait for user review.

---

## Task 7: Verify via CI

**Files:** none (verification only).

**Interfaces:**
- Consumes: everything from Tasks 1-6.

- [ ] **Step 1: Static pre-flight summary**

Run and report:

```bash
git status
git diff --stat HEAD
git grep -n "nbsphinx\|nbmake" -- . ':!docs/superpowers' || echo "no nbsphinx/nbmake anywhere"
ls docs/notebooks/
```

Expected: 8 notebooks in `docs/notebooks/`; no `nbsphinx`/`nbmake` references; root `notebooks/` gone.

- [ ] **Step 2: Hand off to the user for commit + push**

The user reviews the full diff and commits/pushes to `origin/joss_paper` themselves (this plan does not commit). Tell the user explicitly: "Ready for your review. After you push, the `Build Docs` workflow will run on the PR and execute all 8 notebooks."

- [ ] **Step 3: Watch the PR `Build Docs` run**

Once the user confirms they have pushed, monitor the run (the user can paste the Actions URL, or check `https://github.com/pysal/segregation/actions/workflows/build_docs.yml`). Confirm:
  - `Build Docs` job is green (all notebooks executed under `nb_execution_raise_on_error = True`).
  - The `Commit Docs` / `Push to gh-pages` steps are **skipped** (PR event, not push).
  - `Notebooks` workflow no longer appears in the Actions list for the branch.

- [ ] **Step 4: If the build fails**

Read the failing step log. Typical causes and fixes:
  - A notebook raises because a `skip-execution` tag is missing on an optional-dependency cell → add the tag to that cell's metadata (`"tags": ["skip-execution"]`).
  - Missing import in `ci/314-latest.yaml` → add the package.
  - Kernel not found → confirm `nb_kernel_rgx_aliases = {".*": "python3"}` is in `conf.py` and the env has `ipykernel`.
  - Sphinx "not included in any toctree" warning treated as error → confirm all 8 notebooks are in `docs/tutorial.rst` (Task 3).
  Apply the fix, report it, and ask the user to push again.

- [ ] **Step 5: Done**

Report final CI status and summarize what changed for the PR description / review reply to @knaaptime.

---

## Self-Review

**Spec coverage:**
- Spec §1 (notebook location, Option A) → Task 1. ✓
- Spec §2 (`conf.py` myst-nb) → Task 2. ✓
- Spec §3 (`tutorial.rst`) → Task 3. ✓
- Spec §4 (CI: delete nbmake infra, PR trigger, gated publish, env deps) → Task 5 (+ env deps in Task 4). ✓
- Spec §5 (`environment.yml`) → Task 4. ✓
- Spec §6 (docs instructions + README/index link fixes) → Task 6. Note: spec mentioned `paper/paper.md`, but `grep` shows it has no `notebooks/` links, so no change needed there — dropped. ✓
- Spec §7 (`.gitignore`) → Task 6 Step 4. ✓
- Spec "Verification" → Task 7. ✓

**Placeholder scan:** No TBD/TODO. Task 7 Step 4 lists concrete failure causes rather than "handle errors". Config values given verbatim. ✓

**Type consistency:** Notebook basenames identical across Tasks 1, 3, 6, 7. `nb_execution_*` setting names identical between Global Constraints and Task 2. `ci/314-latest.yaml` / `environment.yml` package names (`myst-nb`, `jupyter-cache`) consistent across Tasks 4 and 5. ✓

**Deviations from spec:** `paper/paper.md` edit removed (no matching links present). `docs/index.rst` note in spec said fix `06_inference`/`07_decomposition_example` names — verified those basenames are already correct on the branch, so Task 6 Step 3 only fixes the path prefix.
