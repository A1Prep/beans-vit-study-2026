"""Build the separate follow-up notebooks without executing training."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import argparse
import base64
import hashlib
import io
import json
import textwrap
import zipfile
import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]

payload_names = sorted(['requirements.txt', 'THIRD_PARTY_NOTICES.md', 'ASSISTANCE.md',
    'results/model_comparison/study_plan.json', 'results/model_comparison/timing_analysis_amendment.json',
    'results/data_audit/review_decisions.csv', 'results/data_audit/visual_review_record.json']
    + [p.relative_to(PROJECT).as_posix() for p in (PROJECT / 'src').glob('*.py')])
payload_hashes = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in payload_names}
payload_buffer = io.BytesIO()
with zipfile.ZipFile(payload_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in payload_names:
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 24, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, (PROJECT / name).read_bytes())
payload_bytes = payload_buffer.getvalue()
payload_b64 = base64.b64encode(payload_bytes).decode('ascii')
payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + '\n')


def code(source):
    source = textwrap.dedent(source).strip() + '\n'
    ast.parse(source)
    return nbf.v4.new_code_cell(source)


def notebook(cells):
    result = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    })
    nbf.validate(result)
    return result


reproduction = notebook([
    md('''
    # SigLIP2 versus DINOv2: a low-resource bean-leaf classification comparison

    Reproduce **12 new runs**: two official backbones, frozen and partial fine-tuning arms,
    and seeds 17, 42 and 2026. The newer checkpoint is
    `google/siglip2-base-patch16-224` (2025); the comparator is
    `facebook/dinov2-small` (2023). Both are publicly downloadable at pinned revisions.
    SigLIP2's vision tower and pretrained image pooler supply the representation; a new
    three-class head is trained. This is supervised image classification, not zero-shot
    text prompting. Partial adaptation updates the final two blocks and final normalization;
    SigLIP2's pretrained attention pooler remains frozen.

    **Interpretation:** the original DINOv2 test results were already known when this
    follow-up was planned. The comparison is explicitly exploratory. A newer competitive
    model family does not establish the best model available in October 2026 or a
    state-of-the-art Beans result. Model sizes, pretraining, pooling and native preprocessing
    differ. The fixed learning rates are not independently optimized for each model.

    **Notebook status:** supplied unexecuted. The project's saved results document local
    execution; they do not establish that this notebook was executed in Google Colab.
    `model_comparison_review.ipynb` reads those saved results without training.

    Read the [current technical report](report/technical_report.pdf) for the comparison
    findings. The [original DINOv2 report](report/archive/dinov2_study_2026-10-23/technical_report.pdf)
    is retained as historical evidence. Neither the newer release date nor this notebook
    implies that SigLIP2 achieves a higher score; use the completed results.
    ''') ,
    md('''
    ## 1. Locate the project and install the reference environment

    In Colab choose **Runtime → Change runtime type → T4 GPU**, and select runtime version
    **2026.07** (Python 3.12) if the current default uses Python 3.13. This pinned PyTorch
    stack supports Python 3.11/3.12. GPU allocation is not guaranteed. The notebook contains
    a small SHA-256-verified project bundle, so uploading **this notebook alone** is enough;
    no private GitHub clone, GitHub token or second project ZIP upload is needed.
    The dataset and official model downloads are public. Google sign-in is still needed
    to use a hosted Colab runtime.

    Colab dependencies install into an isolated project environment. All training and
    analysis run in fresh subprocesses there; the live notebook's preloaded PyTorch,
    NumPy and pandas are not replaced. No kernel restart is required by this setup.
    Python 3.11 is the local reference; an actual Colab run records its Python/GPU versions.
    Downloads, installation, loading and analysis add time beyond measured training loops.
    ''') ,
    code('''
    from pathlib import Path
    from datetime import datetime, timezone
    import base64, copy, hashlib, io, json, os, shutil, subprocess, sys, zipfile

    IN_COLAB = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
    EMBEDDED_PROJECT_B64 = __PAYLOAD_BASE64__
    EMBEDDED_PROJECT_SHA256 = __PAYLOAD_SHA256__
    EMBEDDED_FILE_SHA256 = __PAYLOAD_FILE_HASHES__

    def extract_embedded_project(target):
        """Verify every bundled byte and refuse to overwrite different existing files."""
        raw = base64.b64decode(EMBEDDED_PROJECT_B64, validate=True)
        if hashlib.sha256(raw).hexdigest() != EMBEDDED_PROJECT_SHA256:
            raise ValueError("Embedded project archive checksum mismatch")
        target = Path(target).resolve()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(EMBEDDED_FILE_SHA256):
                raise ValueError("Embedded archive manifest differs")
            checked = []
            for name in names:
                destination = (target / name).resolve()
                if not destination.is_relative_to(target):
                    raise ValueError("Unsafe embedded ZIP path")
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != EMBEDDED_FILE_SHA256[name]:
                    raise ValueError("Embedded source checksum mismatch: " + name)
                if destination.exists() and destination.read_bytes() != data:
                    raise FileExistsError("Refusing to overwrite a different project file: " + str(destination))
                checked.append((destination, data))
            for destination, data in checked:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with destination.open("xb") as stream:
                        stream.write(data)
        return target

    def complete_project(path):
        return all((Path(path) / name).is_file() for name in EMBEDDED_FILE_SHA256)

    candidates = [Path.cwd(), Path.cwd() / "beans_vit_project",
                  Path.cwd() / "outputs/beans_vit_project", Path("/content/beans_vit_project")]
    PROJECT_ROOT = next((p.resolve() for p in candidates if complete_project(p)), None)
    if PROJECT_ROOT is None:
        base = Path("/content") if IN_COLAB else Path.cwd()
        PROJECT_ROOT = extract_embedded_project(base / ("beans_vit_comparison_" + EMBEDDED_PROJECT_SHA256[:12]))
        print("Verified self-contained project:", len(EMBEDDED_FILE_SHA256), "files")
    assert complete_project(PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    os.environ["HF_HOME"] = str(PROJECT_ROOT / "work/huggingface")
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    print("Project:", PROJECT_ROOT, "| Python:", sys.version)
    '''.replace('__PAYLOAD_BASE64__', repr(payload_b64)).replace('__PAYLOAD_SHA256__', repr(payload_sha256)).replace('__PAYLOAD_FILE_HASHES__', repr(payload_hashes))) ,
    code('''
    ALLOW_CPU = globals().get("ALLOW_CPU", False)
    USE_ISOLATED_ENV = globals().get("USE_ISOLATED_ENV", IN_COLAB)
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise RuntimeError("This pinned stack needs Python 3.11 or 3.12. In Colab choose Runtime > Change runtime type > Runtime Version 2026.07, then reconnect and run all.")
    if IN_COLAB and not USE_ISOLATED_ENV:
        raise RuntimeError("Keep USE_ISOLATED_ENV=True in Colab to preserve its live kernel packages.")
    SETUP_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    SETUP_LOGS = PROJECT_ROOT / "work/setup_logs" / SETUP_ID
    SETUP_LOGS.mkdir(parents=True, exist_ok=True)

    def logged_command(command, name, env=None):
        """Record command output, never dump the process environment or credentials."""
        command = list(map(str, command))
        with (SETUP_LOGS / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

    if IN_COLAB and not ALLOW_CPU:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True) if shutil.which("nvidia-smi") else None
        if probe is None or probe.returncode != 0:
            raise RuntimeError("No allocated NVIDIA GPU. Select T4 GPU and reconnect before installing; CPU is disabled by default.")
        print("Allocated GPU:", probe.stdout.strip())
        (SETUP_LOGS / "nvidia-smi.txt").write_text(probe.stdout, encoding="utf-8")

    if USE_ISOLATED_ENV:
        requirement_hash = hashlib.sha256((PROJECT_ROOT / "requirements.txt").read_bytes()).hexdigest()
        ENV_ROOT = PROJECT_ROOT / "work/runtime" / (f"py{sys.version_info.major}{sys.version_info.minor}_cu124_" + requirement_hash[:12])
        PROJECT_PYTHON = ENV_ROOT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not PROJECT_PYTHON.is_file():
            # virtualenv has its own bundled seed wheels and does not need Ubuntu ensurepip/python-venv.
            bootstrap = PROJECT_ROOT / "work/bootstrap/virtualenv_20_34_0"
            logged_command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                            "--target", bootstrap, "virtualenv==20.34.0"], "virtualenv_bootstrap")
            bootstrap_env = os.environ.copy()
            bootstrap_env["PYTHONPATH"] = str(bootstrap)
            bootstrap_env["PYTHONNOUSERSITE"] = "1"
            logged_command([sys.executable, "-m", "virtualenv", "--no-periodic-update", "--no-download",
                            "--python", sys.executable, ENV_ROOT], "create_environment", bootstrap_env)
        if IN_COLAB:
            logged_command([PROJECT_PYTHON, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                            "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"], "install_cuda_torch")
    else:
        # Local fallback may explicitly reuse an already provisioned project environment.
        PROJECT_PYTHON = Path(sys.executable)
    logged_command([PROJECT_PYTHON, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                    "-r", PROJECT_ROOT / "requirements.txt"], "install_requirements")
    logged_command([PROJECT_PYTHON, "-m", "pip", "check"], "dependency_check")
    print("Project interpreter:", PROJECT_PYTHON)
    print("The notebook kernel's packages were not changed by the isolated Colab setup.")
    ''') ,
    md('''
    ## 2. Check resources and create a separate reproduction directory

    This workflow preserves the supplied study plan and original evidence. A fresh run ID
    gives the reproduction its own plan, audit, checkpoints and results. It uses the exact
    three seeds and settings in the supplied plan. Do not silently reduce runs or tune from
    test scores. An intentional protocol change requires a separately documented study.
    To resume after a kernel restart, replace the `RUN_ID` expression with the exact
    previously printed ID. Existing audit/plan files are reused and verified, not rewritten.

    CPU execution is disabled by default. Set `ALLOW_CPU=True` only as an explicit decision;
    no CPU completion-time promise is made. Allow several GB for model downloads and all
    selected checkpoints. Temporary Colab storage is not a backup.
    ''') ,
    code('''
    import pandas as pd
    from IPython.display import display, Image

    RUN_ID = globals().get("RUN_ID", "comparison_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    runtime_probe = r"""
    import importlib.metadata, json, platform, sys
    import numpy, pandas, PIL, sklearn, torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, SiglipVisionModel
    from pathlib import Path
    requirements = Path(sys.argv[1]).read_text().splitlines()
    versions = {}
    for line in requirements:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        package, expected = line.split('==')
        actual = importlib.metadata.version(package)
        if actual.split('+')[0] != expected:
            raise RuntimeError(f'Project environment version mismatch: {package} {actual} != {expected}')
        versions[package] = actual
    record = {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'base_prefix': sys.base_prefix, 'platform': platform.platform(),
              'packages': versions, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'cuda_build': torch.version.cuda}
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 10:
            raise RuntimeError('This pinned CUDA12.4/PyTorch2.6 stack does not target Blackwell GPUs; choose a T4 runtime.')
        gpu = torch.cuda.get_device_properties(0)
        value = torch.ones(1, device='cuda').add_(1).item()
        torch.cuda.synchronize()
        assert value == 2.0
        record['gpu'] = {'name': gpu.name, 'memory_bytes': gpu.total_memory,
                         'compute_capability': list(capability), 'cuda_tensor_smoke_test': 'passed'}
    print(json.dumps(record))
    """
    checked_runtime = subprocess.run([str(PROJECT_PYTHON), "-c", runtime_probe,
                                      str(PROJECT_ROOT / "requirements.txt")],
                                     capture_output=True, text=True)
    (SETUP_LOGS / "runtime_probe.stdout.json").write_text(checked_runtime.stdout, encoding="utf-8")
    (SETUP_LOGS / "runtime_probe.stderr.log").write_text(checked_runtime.stderr, encoding="utf-8")
    if checked_runtime.returncode:
        print(checked_runtime.stderr)
        checked_runtime.check_returncode()
    RUNTIME_RECORD = json.loads(checked_runtime.stdout)
    DEVICE = RUNTIME_RECORD["device"]
    if DEVICE == "cpu" and not ALLOW_CPU:
        raise RuntimeError("No CUDA GPU allocated. Select a GPU runtime or explicitly set ALLOW_CPU=True.")
    if USE_ISOLATED_ENV:
        assert RUNTIME_RECORD["prefix"] != RUNTIME_RECORD["base_prefix"]
    print(json.dumps(RUNTIME_RECORD, indent=2))

    ORIGINAL_STUDY = PROJECT_ROOT / "results/model_comparison"
    ORIGINAL_PLAN = ORIGINAL_STUDY / "study_plan.json"
    ORIGINAL_TIMING = ORIGINAL_STUDY / "timing_analysis_amendment.json"
    STUDY = PROJECT_ROOT / "results/reproduction" / RUN_ID
    RUN_ROOT = PROJECT_ROOT / "work/comparison_runs" / RUN_ID
    DATA_ROOT = PROJECT_ROOT / "work/data/beans"
    IMAGE_ROOT = DATA_ROOT / "images"
    AUDIT = STUDY / "data_audit"
    MANIFEST = AUDIT / "manifest.csv"
    PLAN_PATH = STUDY / "study_plan.json"
    TIMING_PATH = STUDY / "timing_analysis_amendment.json"
    assert STUDY.resolve() != ORIGINAL_STUDY.resolve()
    STUDY.mkdir(parents=True, exist_ok=True)
    RUNTIME_RECORD.update({"recorded_utc": datetime.now(timezone.utc).isoformat(),
                           "execution_surface": "Google Colab" if IN_COLAB else "Local notebook",
                           "isolated_project_environment": USE_ISOLATED_ENV,
                           "notebook_kernel_python": sys.version,
                           "display_only_pandas_version": pd.__version__,
                           "embedded_project_sha256": EMBEDDED_PROJECT_SHA256})
    setup_evidence = STUDY / "runtime_setup" / SETUP_ID
    shutil.copytree(SETUP_LOGS, setup_evidence, dirs_exist_ok=True)
    (setup_evidence / "runtime_record.json").write_text(json.dumps(RUNTIME_RECORD, indent=2) + "\\n", encoding="utf-8")

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def run(script, *args):
        command = [str(PROJECT_PYTHON), str(PROJECT_ROOT / "src" / script), *map(str, args)]
        print("Running", script, flush=True)
        log_folder = STUDY / "execution_logs"
        log_folder.mkdir(parents=True, exist_ok=True)
        log_path = log_folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + Path(script).stem + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)

    print("New evidence:", STUDY)
    print("Checkpoint directory:", RUN_ROOT)
    print("Disk free GiB:", round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 1))
    ''') ,
    md('''
    ## 3. Download and audit the pinned Beans dataset

    Preserve the published 1,034/133/128 train/validation/test split. Download verification,
    image decoding, exact file/pixel duplicate checks and pHash screening run again here.
    Reuse recorded AI-assisted visual decisions only for matching pairs from this exact
    release. This does not validate agricultural labels or establish farm independence.
    The original integrity audit inspected test images before modelling; the follow-up
    also starts with previous test-result exposure, which remains disclosed.
    ''') ,
    code('''
    if PLAN_PATH.exists() or (STUDY / "execution_identity.json").exists():
        existing_plan = read_json(PLAN_PATH)
        assert sha256(MANIFEST) == existing_plan["dataset"]["manifest_sha256"]
        print("Reusing the existing immutable manifest:", MANIFEST)
    else:
        AUDIT.mkdir(parents=True, exist_ok=True)
        for name in ["review_decisions.csv", "visual_review_record.json"]:
            source = PROJECT_ROOT / "results/data_audit" / name
            if source.is_file():
                shutil.copy2(source, AUDIT / name)
        run("prepare_data.py", "--data-root", DATA_ROOT, "--output-dir", AUDIT,
            "--review-decisions", AUDIT / "review_decisions.csv")
    audit = read_json(AUDIT / "audit_summary.json")
    assert audit["dataset_revision"] == "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
    assert audit["included_counts"] == {"train": 1034, "validation": 133, "test": 128}
    assert not audit["corrupt_images"]
    unresolved = {key: count for key, count in audit["visual_review_status"].items()
                  if key not in {"distinct", "confirmed_duplicate"} and count}
    if unresolved:
        raise RuntimeError(f"Inspect unresolved duplicate candidates before proceeding: {unresolved}")
    print(json.dumps({key: audit[key] for key in ["included_counts", "corrupt_images",
                      "near_duplicate_candidate_pairs", "visual_review_status"]}, indent=2))
    display(Image(filename=str(AUDIT / "dataset_examples.png"), width=850))
    ''') ,
    md('''
    ## 4. Record the reproduction plan before the pilot and full training

    Absolute paths in the regenerated manifest change its byte hash. Copy the original
    plan into this reproduction, replace that manifest path/hash, record the new creation
    time and the original plan's checksum, and keep every experimental setting unchanged.
    The original plan and original results remain intact. The historical test-exposure
    limitation is retained; a reproduction does not undo previous knowledge of the test set.
    Record the descriptive timing policy before execution as well, linking the original
    timing amendment. Its 809-second observation belongs to the original run; it is not
    asserted to have occurred in your reproduction.
    ''') ,
    code('''
    original_hash = sha256(ORIGINAL_PLAN)
    original = read_json(ORIGINAL_PLAN)
    original_timing_hash = sha256(ORIGINAL_TIMING)
    original_timing = read_json(ORIGINAL_TIMING)
    if TIMING_PATH.exists():
        timing_policy = read_json(TIMING_PATH)
        assert timing_policy["original_amendment_sha256"] == original_timing_hash
        assert timing_policy["before_new_test_evaluation"] is True
        assert timing_policy["raw_times_preserved"] is True
    else:
        if (STUDY / "execution_identity.json").exists():
            raise RuntimeError("Execution began without the timing policy; do not retrospectively claim a pre-execution record.")
        observed_test_files = [path for pattern in ["metrics_test*.json", "predictions_test*.csv"]
                               for path in RUN_ROOT.rglob(pattern)]
        assert not observed_test_files, "Cannot record a pre-evaluation policy after test outputs exist."
        timing_policy = {key: original_timing[key] for key in [
            "training_changes", "checkpoint_selection_changes", "test_evaluation_changes",
            "raw_times_preserved", "existing_metrics_retained", "additional_descriptive_statistics",
            "quantile_method", "cost_figure_metric", "exclusions_or_corrections", "interpretation"]}
        timing_policy.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "Timing summary policy carried forward before this reproduction's execution",
            "before_new_test_evaluation": True,
            "original_dinov2_test_results_previously_known": True,
            "observed_new_test_artifact_count": 0,
            "original_amendment": "results/model_comparison/timing_analysis_amendment.json",
            "original_amendment_sha256": original_timing_hash,
            "original_amendment_created_utc": original_timing["created_utc"],
            "motivation": "Apply the original descriptive timing policy to all reproduction epochs; the original unusually long epoch is historical and is not asserted for this runtime.",
        })
        with TIMING_PATH.open("x", encoding="utf-8") as stream:
            json.dump(timing_policy, stream, indent=2)
            stream.write("\\n")
    if PLAN_PATH.exists():
        plan = read_json(PLAN_PATH)
        assert plan["reproduction"]["original_plan_sha256"] == original_hash
        assert plan["dataset"]["manifest_sha256"] == sha256(MANIFEST)
        assert plan["reproduction"]["run_id"] == RUN_ID
        print("Reusing the existing plan without changing it.")
    else:
        if (STUDY / "execution_identity.json").exists():
            raise RuntimeError("Execution began but its plan is missing; do not reconstruct it retrospectively.")
        plan = copy.deepcopy(original)
        plan["created_utc"] = datetime.now(timezone.utc).isoformat()
        plan["study_status"] = "Reproduction of the fixed exploratory follow-up; original test exposure remains disclosed"
        plan["dataset"]["manifest"] = str(MANIFEST.resolve())
        plan["dataset"]["manifest_sha256"] = sha256(MANIFEST)
        plan["reproduction"] = {
            "run_id": RUN_ID,
            "original_plan": "results/model_comparison/study_plan.json",
            "original_plan_sha256": original_hash,
            "original_plan_created_utc": original["created_utc"],
            "experimental_settings_changed": False,
            "regenerated_manifest_has_runtime_specific_absolute_paths": True,
            "prior_test_exposure_retained": True,
            "timing_policy_sha256": sha256(TIMING_PATH),
        }
        with PLAN_PATH.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\\n")
    assert sha256(ORIGINAL_PLAN) == original_hash
    assert plan["settings"] == original["settings"] and plan["models"] == original["models"]
    assert plan["reproduction"]["timing_policy_sha256"] == sha256(TIMING_PATH)
    assert sha256(ORIGINAL_TIMING) == original_timing_hash
    print(json.dumps({"models": plan["models"], "settings": plan["settings"]}, indent=2))
    print("Recorded reproduction plan:", PLAN_PATH, "| SHA256:", sha256(PLAN_PATH))

    def phase(name):
        run("run_model_comparison.py", "--manifest", MANIFEST, "--data-root", IMAGE_ROOT,
            "--run-root", RUN_ROOT, "--out", STUDY, "--study-plan", PLAN_PATH,
            "--device", DEVICE, "--phase", name, "--resume")
    ''') ,
    md('''
    ## 5. Run full-epoch pilots for both models and both arms

    Each pilot checks gradients, frozen parameters and checkpoint round trips, and
    benchmarks one complete training epoch. Full experiments restart from the pinned
    pretrained weights. The scripts download official checkpoints automatically and use
    their pinned revisions; no remote executable model code is required. No test
    prediction is requested by this phase. Check actual memory and timing before proceeding.
    ''') ,
    code('''
    phase("pilot")
    pilots = []
    for model in plan["models"]:
        records = read_json(STUDY / model / "pilot_report.json")
        assert len(records) == 2 and all(item["status"] == "passed" for item in records)
        for item in records:
            pilots.append({"model": model, "arm": item["arm"],
                           "one_training_epoch_seconds": item["training"]["seconds"],
                           "peak_cuda_allocated_bytes": item["peak_cuda_allocated_bytes"]})
    display(pd.DataFrame(pilots))
    training_minutes = sum(row["one_training_epoch_seconds"] for row in pilots) * plan["settings"]["epochs"] * len(plan["settings"]["seeds"]) / 60
    print("Full-epoch-budget training-only estimate in minutes:", round(training_minutes, 1))
    print("This excludes validation, downloads, loading, checkpoint I/O and analysis.")
    ''') ,
    md('''
    ## 6. Train all 12 runs with validation-only checkpoint selection

    Images receive the model's recorded full-image resizing and normalization. Augmentation
    randomness and image order are matched across backbones by seed, epoch and image ID.
    Classifier initialization is paired within a backbone; embedding dimensions differ
    across backbones. AdamW settings, batch size, epoch limit and stopping rule are fixed.
    Select by highest validation macro-F1, then lower loss, then earlier epoch. Only a strict
    F1 improvement resets patience. The implementation permits platform-specific
    nondeterministic CUDA operations with a warning; bitwise identical reruns are not promised.

    Completed matching runs may be reused with `--resume`; incomplete runs are never
    overwritten automatically. Keep temporary runtime checkpoints if resumption is needed.
    ''') ,
    code('''
    phase("train")
    summaries = []
    for model in plan["models"]:
        for arm in plan["settings"]["arms"]:
            for seed in plan["settings"]["seeds"]:
                folder = RUN_ROOT / model / "runs" / arm / f"seed_{seed}"
                summary = read_json(folder / "summary.json")
                assert (folder / "best.pt").is_file()
                summaries.append({"model": model, **{key: summary[key] for key in
                    ["arm", "seed", "best_epoch", "epochs_completed", "best_validation_macro_f1", "training_seconds"]}})
    assert len(summaries) == 12
    display(pd.DataFrame(summaries))
    print("Training complete. The table shows validation selection; this phase did not request test inference.")
    ''') ,
    md('''
    ## 7. Lock the selected checkpoints and evaluate the existing test split

    The driver requires every planned run, verifies configurations and hashes, and creates
    both model locks before new test inference. The validator checks membership and hashes
    before each evaluation. The lock records that this is an exploratory reuse of a test
    set evaluated in the earlier study, not a fresh confirmatory holdout.

    Evaluate clean test images for all 12 selected checkpoints. Seed 17 additionally uses
    the already-fixed blur radius 1 and brightness ×0.7 conditions. These are paired
    synthetic sensitivity checks, not new independent field observations. Do not use their
    scores for more tuning or replace checkpoints after seeing test results.
    ''') ,
    code('''
    phase("evaluate")
    print(json.dumps(read_json(STUDY / "evaluation_completion.json"), indent=2))
    for model in plan["models"]:
        lock = read_json(STUDY / model / "protocol.lock.json")
        assert lock["original_dinov2_test_results_known_before_followup"] is True
        assert lock["new_followup_test_predictions_evaluated_before_lock"] is False
        print(model, "lock:", sha256(STUDY / model / "protocol.lock.json"))
    ''') ,
    md('''
    ## 8. Analyse metrics, uncertainty, cost and overfitting

    The analysis recomputes metrics from saved predictions and produces all-seed summaries,
    class metrics, confusion matrices, real sample errors and synthetic sensitivity figures.
    The primary model difference is partial SigLIP2 minus partial DINOv2. Seed 17 uses
    2,000 paired class-stratified bootstrap resamples with seed 20261023. This descriptive
    interval is conditional on the reused test sample and reference pair; it is not an
    interval for the three-seed mean or an independent superiority confirmation.

    **Timing analysis amendment:** an unusually long epoch was observed during the original
    follow-up's training, before its test evaluation. The amended analysis adds median,
    quartiles and range of observed epoch times while retaining every raw epoch duration
    and total. These describe the recorded hardware session; they are not a controlled
    throughput benchmark or proof of the slowdown's cause. Your reproduction may have
    different timing variation. Compare total observed training cost as well as medians.
    The completed original follow-up also contains a 184.36-second SigLIP2 partial epoch,
    as well as the 809.10-second frozen epoch that motivated the amendment. Both remain
    in all timing summaries; neither has an established cause.

    Clean training and validation inference scores each unchanged selected checkpoint with
    no augmentation or parameter updates. Saved test metrics provide context. All-seed
    learning curves reveal late deterioration; a small clean train-validation gap cannot
    prove absence of overfitting, leakage or distribution shift. Online augmented training
    loss and fixed-checkpoint clean loss are different evaluations.
    ''') ,
    code('''
    phase("analyse")
    completion = read_json(STUDY / "completion.json")
    assert completion["status"] == "completed" and completion["run_count"] == 12
    summary = pd.read_csv(STUDY / "combined_summary.csv")
    display(summary)
    timing_columns = ["model", "arm", "pooled_epochs", "pooled_epoch_median_seconds",
                      "pooled_epoch_q1_seconds", "pooled_epoch_q3_seconds", "pooled_epoch_iqr_seconds",
                      "pooled_epoch_min_seconds", "pooled_epoch_max_seconds", "pooled_training_seconds",
                      "pooled_training_seconds_per_epoch", "training_seconds_mean"]
    display(summary[timing_columns])
    display(pd.read_csv(STUDY / "paired_model_differences.csv"))
    analysis = read_json(STUDY / "comparison_analysis.json")
    print(json.dumps(analysis["primary_comparison"], indent=2))
    print("Timing interpretation:", analysis["timing_analysis"]["interpretation"])
    for stem in ["performance_cost", "partial_confusion", "reference_learning_curves",
                 "siglip_sample_predictions", "model_generalisation"]:
        display(Image(filename=str(STUDY / "figures" / (stem + ".png")), width=1000))
    for model in plan["models"]:
        print(model, "selected-checkpoint clean gaps and trajectory diagnostics")
        display(pd.read_csv(STUDY / model / "overfitting/summary.csv"))
        display(pd.read_csv(STUDY / model / "overfitting/history_diagnostics.csv"))
    print("Interpretation limits:", *analysis["interpretation_limits"], sep="\\n- ")
    ''') ,
    md('''
    ## 9. Export evidence before the runtime ends

    Export audit files, plans, locks, configurations, histories, predictions, metrics,
    figures and the environment listing. Weights, raw images and caches are excluded.
    Save `RUN_ROOT` separately if you need checkpoint inference without retraining.
    The bundle documents this reproduction; it must not replace the original reported
    evidence without updating the report and identifying the new run.
    ''') ,
    code('''
    environment = subprocess.run([str(PROJECT_PYTHON), "-m", "pip", "freeze"],
                                 check=True, capture_output=True, text=True).stdout
    (STUDY / "environment.txt").write_text(environment, encoding="utf-8")
    export = PROJECT_ROOT / "work" / (RUN_ID + "_evidence.zip")
    export.parent.mkdir(parents=True, exist_ok=True)
    allowed = {".json", ".csv", ".png", ".pdf", ".md", ".txt", ".log"}
    with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STUDY.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                archive.write(path, arcname=str(path.relative_to(STUDY)))
    print("Saved:", export, "| MiB:", round(export.stat().st_size / 2**20, 2))
    if IN_COLAB:
        from google.colab import files
        files.download(str(export))
    ''') ,
    md('''
    ## References, attribution and responsible interpretation

    - [Makerere Beans source](https://github.com/AI-Lab-Makerere/ibean) and
      [pinned distribution](https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/27aa014ce09b193e1a6f58112d4a66e0eddb69c5).
      MIT license; retain `THIRD_PARTY_NOTICES.md` when redistributing.
    - [DINOv2 paper](https://arxiv.org/abs/2304.07193) and
      [official pinned DINOv2-Small checkpoint](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056).
    - [SigLIP 2 paper](https://arxiv.org/abs/2502.14786) and
      [official pinned SigLIP2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2).
    - [Colab resource and runtime FAQ](https://research.google.com/colaboratory/faq.html).
    - [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html),
      [official PyTorch 2.6 CUDA wheels](https://docs.pytorch.org/get-started/previous-versions/),
      and [virtualenv 20.34.0](https://pypi.org/project/virtualenv/20.34.0/) used for isolated setup.

    The dataset has only three categories and limited geography. Unknown farm/plant grouping,
    unknown pretraining overlap and uncalibrated confidence constrain deployment claims.
    Errors can produce missed disease or unnecessary treatment; this benchmark does not
    validate field diagnosis. Disclose AI assistance according to course rules and verify
    the agricultural interpretation yourself. No automated review guarantees a grade.
    ''') ,
])

review = notebook([
    md('''
    # Read the executed SigLIP2–DINOv2 comparison

    This notebook reads the saved results of the 12-run exploratory follow-up. It performs
    **no training, optimization, model downloading or test inference**. The reproduction
    workflow is `model_comparison.ipynb`; its unexecuted status is separate from this reader.
    The original DINOv2 test scores were known before this follow-up, so the test set is
    reused. Treat comparisons as descriptive evidence under fixed settings.

    [Current technical report](report/technical_report.pdf) ·
    [Archived original DINOv2 report](report/archive/dinov2_study_2026-10-23/technical_report.pdf)

    The newer model is not assumed to win. All tables below read the actual completed runs.
    ''') ,
    code('''
    from pathlib import Path
    import json
    import pandas as pd
    from IPython.display import display, Image, Markdown

    candidates = [Path.cwd(), Path.cwd() / "beans_vit_project", Path.cwd() / "outputs/beans_vit_project"]
    ROOT = next((path.resolve() for path in candidates
                 if (path / "results/model_comparison/study_plan.json").is_file()), None)
    if ROOT is None:
        raise RuntimeError("Open this notebook from the complete extracted project or repository.")
    STUDY = ROOT / "results/model_comparison"
    def read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    completion = read_json(STUDY / "completion.json")
    assert completion["status"] == "completed" and completion["run_count"] == 12
    plan = read_json(STUDY / "study_plan.json")
    analysis = read_json(STUDY / "comparison_analysis.json")
    print("Completed:", completion["completed_utc"])
    print("Run count:", completion["run_count"], "| Previous test exposure:", completion["prior_test_exposure"])
    print(plan["study_status"])
    ''') ,
    md('''
    ## Every run and the aggregate comparison

    Accuracy and macro-F1 values are proportions; `*_pp` columns are percentage-point gaps.
    Sample standard deviations describe three training seeds, not independent datasets.
    Parameter counts, observed epoch times and peak allocated memory describe resource cost.
    A low-resource fixed comparison does not measure the best achievable score of either model.

    **Timing amendment:** during training, SigLIP2 frozen seed 17 epoch 9 took 809.10
    seconds while its other epochs were about 13 seconds. The completed follow-up also
    contains a 184.36-second training epoch (SigLIP2 partial seed 42, epoch 6), compared
    with about 16 seconds for its other epochs. Both causes are unresolved.
    Median, quartiles and range were added before the follow-up test evaluation to describe
    the observed variation. The full run table and aggregate table retain raw training
    totals and arithmetic means; neither long epoch was deleted or replaced. Median epoch
    cost is descriptive of this execution, not a controlled speed or throughput comparison.
    Pooled medians include every epoch, so seeds completing more epochs receive more weight.
    Read the [recorded timing amendment](results/model_comparison/timing_analysis_amendment.json)
    for the dated decision and the original frozen-epoch observation that motivated it.
    That historical amendment is unchanged; the later partial-epoch observation is
    disclosed here from the completed run's saved history.
    ''') ,
    code('''
    runs = pd.read_csv(STUDY / "combined_run_metrics.csv")
    assert len(runs) == 12
    assert set(runs["model"]) == {"dinov2", "siglip2"}
    assert set(runs["arm"]) == {"frozen", "partial"}
    assert set(runs["seed"]) == {17, 42, 2026}
    display(runs)
    summary = pd.read_csv(STUDY / "combined_summary.csv")
    display(summary)
    timing_columns = ["model", "arm", "pooled_epochs", "pooled_epoch_median_seconds",
                      "pooled_epoch_q1_seconds", "pooled_epoch_q3_seconds", "pooled_epoch_iqr_seconds",
                      "pooled_epoch_min_seconds", "pooled_epoch_max_seconds", "pooled_training_seconds",
                      "pooled_training_seconds_per_epoch", "training_seconds_mean"]
    display(Markdown("**Descriptive observed epoch cost, retaining total training time**"))
    display(summary[timing_columns])
    timing = analysis["timing_analysis"]
    assert timing["raw_times_preserved"] is True
    assert timing["amendment"]["before_new_test_evaluation"] is True
    print("Timing interpretation:", timing["interpretation"])
    print("Timing amendment recorded:", timing["amendment"]["created_utc"])
    display(pd.read_csv(STUDY / "paired_model_differences.csv"))
    print(json.dumps(analysis["primary_comparison"], indent=2))
    display(Image(filename=str(STUDY / "figures/performance_cost.png"), width=1000))
    ''') ,
    md('''
    ## Class errors and real sample outputs

    These are reference seed-17 outputs. Confidence scores are uncalibrated softmax values;
    a confident error can still occur. The bootstrap comparison is conditional on this
    reference pair and this small reused test sample. It does not establish independent
    farm-level or external-field superiority.
    ''') ,
    code('''
    display(Image(filename=str(STUDY / "figures/partial_confusion.png"), width=1000))
    display(Image(filename=str(STUDY / "figures/siglip_sample_predictions.png"), width=1000))
    for model in ["dinov2", "siglip2"]:
        display(Markdown("### " + model + " class metrics and synthetic sensitivity"))
        display(pd.read_csv(STUDY / model / "per_class_metrics.csv"))
        display(pd.read_csv(STUDY / model / "stress_metrics.csv"))
    ''') ,
    md('''
    ## Overfitting diagnostics for every selected checkpoint

    Clean training and validation metrics use identical deterministic preprocessing and
    evaluation mode with each fixed checkpoint. Test metrics are reused from the recorded
    evaluations. All histories are included because later validation deterioration can be
    hidden by inspecting only the selected epoch. Online augmented training loss is not
    directly equivalent to a fixed-checkpoint clean loss. Small average gaps do not prove
    absence of overfitting, data dependence or distribution shift.
    ''') ,
    code('''
    display(Image(filename=str(STUDY / "figures/model_generalisation.png"), width=1000))
    display(Image(filename=str(STUDY / "figures/reference_learning_curves.png"), width=1000))
    for model in ["dinov2", "siglip2"]:
        display(Markdown("### " + model + " clean checkpoint metrics and all-seed trajectories"))
        folder = STUDY / model / "overfitting"
        audit = read_json(folder / "audit.json")
        assert audit["optimization_performed"] is False
        assert audit["new_test_evaluation_performed"] is False
        assert audit["validation_predictions_match_original"] is True
        display(pd.read_csv(folder / "summary.csv"))
        display(pd.read_csv(folder / "checkpoint_gaps.csv"))
        display(pd.read_csv(folder / "history_diagnostics.csv"))
        display(Image(filename=str(folder / "all_seed_learning_curves.png"), width=1000))
    ''') ,
    md('''
    ## Interpretation and provenance

    The plan was fixed before this follow-up's training/evaluation, while the original
    study's test results were already known. Both backbones were retrained. Model families
    differ in size, pretraining, image readout and native preprocessing, so the experiment
    does not isolate the causal effect of model age. SigLIP2 is a newer competitive family;
    being newer does not itself guarantee better Beans performance. The formal technical
    report explains dataset limits, treatment consequences and responsible use.
    ''') ,
    code('''
    for item in analysis["interpretation_limits"]:
        print("-", item)
    print("Study plan SHA256:", completion["study_plan_sha256"])
    print("Training source SHA256:", completion["training_source_sha256"])
    print("This notebook displayed saved evidence; no model was trained or evaluated here.")
    ''') ,
])

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--all-unexecuted', action='store_true',
                    help='Also build an unexecuted review notebook only when no executed reader exists')
args = parser.parse_args()
if args.all_unexecuted and (PROJECT / 'model_comparison_review.ipynb').exists():
    previous = nbf.read(PROJECT / 'model_comparison_review.ipynb', as_version=4)
    if any(c.cell_type == 'code' and (c.outputs or c.execution_count is not None) for c in previous.cells):
        raise RuntimeError('Refusing to overwrite an executed saved-results reader.')
nbf.write(reproduction, PROJECT / 'model_comparison.ipynb')
if args.all_unexecuted:
    nbf.write(review, PROJECT / 'model_comparison_review.ipynb')

evidence = {
    'validated_utc': datetime.now(timezone.utc).isoformat(),
    'training_notebook': 'model_comparison.ipynb',
    'training_notebook_cells': len(reproduction.cells),
    'training_notebook_executed': False,
    'training_notebook_static_validation': 'nbformat schema and all Python cell syntax passed',
    'review_notebook': 'model_comparison_review.ipynb',
    'review_notebook_cells': len(review.cells),
    'review_notebook_executed': False,
    'review_notebook_static_validation': 'nbformat schema and all Python cell syntax passed',
    'colab_execution_verified': False,
    'execution_note': 'The saved-results reader will be executed only after actual completion artifacts exist.',
}
# Validation evidence belongs to the actual validation/execution step. Never overwrite
# the already-executed reader's record simply because the portable notebook was rebuilt.
print(json.dumps({'training_notebook_cells': len(reproduction.cells), 'schema_and_syntax': 'passed',
                  'embedded_project_files': len(payload_names), 'embedded_zip_bytes': len(payload_bytes),
                  'embedded_project_sha256': payload_sha256,
                  'review_notebook_changed': args.all_unexecuted,
                  'existing_notebook_validation_unchanged': True}, indent=2))
