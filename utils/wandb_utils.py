"""Scalar-only Weights & Biases logging, shared by every pipeline stage.

The pipeline runs each stage as its own subprocess (see `run_command` in
inference.py), so there is no single long-lived process to own a wandb run.
Instead `inference.py` mints a run id up front and exports it; every child
resumes that same id with `resume="allow"`. wandb resumes the internal step
counter from the server, so history stays monotonic across the whole pipeline,
and `define_metric` declarations persist server-side, so each stage only has to
declare its own axes.

That arrangement carries one invariant: **no two processes may hold the run at
once.** Stages are strictly sequential (`subprocess.run` blocks), and the parent
opens the run exactly twice -- once in `start_pipeline`, once in
`finish_pipeline` -- finishing immediately each time. Overlapping writers would
need wandb's `mode="shared"`, which this does not use.

Everything here is a hard no-op unless `RI3D_WANDB=1` is in the environment, and
`import wandb` happens inside `attach()` rather than at module scope, so this
module imports fine on a machine without wandb and degrades to a single warning.

Only scalars are ever logged: `_scalar` coerces tensors and numpy arrays to a
float and drops anything else, and `console="off"` keeps stdout out of the run
(stage 2a alone emits 60k tqdm carriage returns).
"""

import atexit
import os

# Env contract, all set by inference.py's start_pipeline().
ENV_ENABLED = "RI3D_WANDB"
ENV_STAGE = "RI3D_WANDB_STAGE"
ENV_LOG_EVERY = "RI3D_WANDB_LOG_EVERY"

_RUN = None
_FAILED = False
_WARNED = set()
_PIPELINE_DONE = False


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def enabled():
    """True when the pipeline asked for logging and nothing has failed yet."""
    return not _FAILED and os.environ.get(ENV_ENABLED) == "1"


def stage(default=None):
    """The stage label this process is running under, e.g. '1b' or '2a'."""
    return os.environ.get(ENV_STAGE) or default


def log_every(default=25):
    try:
        return max(1, int(os.environ.get(ENV_LOG_EVERY, default)))
    except (TypeError, ValueError):
        return default


def _warn(msg, once_key=None):
    if once_key is not None:
        if once_key in _WARNED:
            return
        _WARNED.add(once_key)
    print(f"[!] wandb: {msg}")


def _fail(msg):
    """Latch off permanently -- a logging problem must never kill a stage."""
    global _FAILED, _RUN
    _FAILED = True
    _RUN = None
    _warn(f"{msg}; scalar logging disabled for this process", once_key="_fail")


def _scalar(key, value):
    """Coerce to a float, or return None to drop the key.

    Duck-typed so this module never imports torch or numpy: `tools/` stages
    should not pay for that. Handles 0-dim cuda tensors, the shape-(1,) result
    of the pearson depth branch, and numpy arrays (stage 2b's parameter diffs).
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if hasattr(value, "detach"):  # torch.Tensor
            return float(value.detach().float().mean().item())
        if hasattr(value, "dtype") and hasattr(value, "mean"):  # np.ndarray
            return float(value.mean())
    except Exception as e:
        _warn(f"could not coerce '{key}' ({e})", once_key=f"coerce:{key}")
        return None
    _warn(f"dropping non-scalar '{key}' ({type(value).__name__})",
          once_key=f"drop:{key}")
    return None


# ---------------------------------------------------------------------------
# child-side API
# ---------------------------------------------------------------------------

def _make_settings(wandb):
    """Best-effort Settings; wandb's Settings is extra='forbid', so an older
    version rejecting one field must not take the whole run down."""
    for kwargs in (
        dict(init_timeout=30.0, finish_timeout=60.0, console="off",
             x_stats_sampling_interval=30.0),
        dict(init_timeout=30.0, console="off"),
        dict(console="off"),
    ):
        try:
            return wandb.Settings(**kwargs)
        except Exception:
            continue
    return None


def attach(job_type=None, config=None, name=None, group=None, tags=None):
    """Open or resume the pipeline run for this process. None when disabled."""
    global _RUN
    if _RUN is not None:
        return _RUN
    if not enabled():
        return None

    try:
        import wandb
    except Exception as e:
        _fail(f"unavailable ({e})")
        return None

    try:
        _RUN = wandb.init(
            id=os.environ.get("WANDB_RUN_ID") or None,
            project=os.environ.get("WANDB_PROJECT") or "ri3d",
            entity=os.environ.get("WANDB_ENTITY") or None,
            group=group or os.environ.get("WANDB_RUN_GROUP") or None,
            job_type=job_type or stage() or "stage",
            name=name,
            tags=tags,
            config=config,
            # "allow" not "must": a lost id should create a run, not raise
            # halfway through a 60-minute stage.
            resume="allow",
            settings=_make_settings(wandb),
        )
    except Exception as e:
        _fail(f"init failed ({e})")
        return None

    atexit.register(_atexit)
    return _RUN


def _atexit():
    try:
        finish()
    except Exception:
        pass


def define_axis(prefix, axis="step"):
    """Give `<prefix>/*` its own x-axis at `<prefix>/<axis>`.

    Declarations persist on the run, so each stage (and each leave-one-out view)
    declares only its own; re-declaring is idempotent.
    """
    if _RUN is None:
        return
    try:
        # Exact name first, then the glob -- exact declarations take precedence.
        _RUN.define_metric(f"{prefix}/{axis}")
        _RUN.define_metric(f"{prefix}/*", step_metric=f"{prefix}/{axis}",
                           step_sync=True)
    except Exception as e:
        _warn(f"define_metric('{prefix}') failed ({e})", once_key=f"axis:{prefix}")


def log(data):
    """One history row. Never passes step= -- the declared axis does the work."""
    if _RUN is None:
        return
    row = {}
    for k, v in data.items():
        s = _scalar(k, v)
        if s is not None:
            row[k] = s
    if not row:
        return
    try:
        _RUN.log(row)
    except Exception as e:
        _warn(f"log failed ({e})", once_key="log")


def log_summary(data):
    if _RUN is None:
        return
    for k, v in data.items():
        s = _scalar(k, v)
        if s is None:
            continue
        try:
            _RUN.summary[k] = s
        except Exception as e:
            _warn(f"summary['{k}'] failed ({e})", once_key="summary")


def due(i, last=None, every=None):
    """Cadence gate. Always fires on the first and last step of a loop, which
    matters for stage 1c's 20-iteration bursts."""
    if _RUN is None:
        return False
    every = every or log_every()
    return i == 1 or i % every == 0 or (last is not None and i == last)


def finish(exit_code=0):
    global _RUN
    if _RUN is None:
        return
    run, _RUN = _RUN, None
    try:
        run.finish(exit_code=exit_code)
    except Exception as e:
        _warn(f"finish failed ({e})", once_key="finish")


# ---------------------------------------------------------------------------
# parent-side API (inference.py only)
# ---------------------------------------------------------------------------

def start_pipeline(project="ri3d", entity=None, run_name=None, group=None,
                   config=None, run_id=None, tags=None, wandb_dir=None,
                   every=25, dry_run=False):
    """Mint the run id, export the env every child reads, and seed the run.

    Returns the run id, or None when wandb could not be set up.
    """
    try:
        import wandb
    except Exception as e:
        _warn(f"unavailable ({e}); --wandb ignored")
        return None

    rid = run_id or wandb.util.generate_id()
    env = {
        ENV_ENABLED: "1",
        ENV_LOG_EVERY: str(every),
        "WANDB_RUN_ID": rid,
        "WANDB_PROJECT": project,
        "WANDB_RESUME": "allow",
        # Children cd elsewhere (the GSFix3D root, the GGPT env), so without a
        # fixed dir their run files scatter across the filesystem.
        "WANDB_DIR": wandb_dir or os.getcwd(),
        # Scalars only: the default "auto" would capture every tqdm redraw.
        "WANDB_CONSOLE": "off",
    }
    if entity:
        env["WANDB_ENTITY"] = entity
    if group:
        env["WANDB_RUN_GROUP"] = group

    if dry_run:
        print("[i] wandb (dry run) would export:")
        for k, v in env.items():
            print(f"      {k}={v}")
        return rid

    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
    os.environ.update(env)

    run = attach(job_type="pipeline", config=config, name=run_name,
                 group=group, tags=tags)
    if run is None:
        return None
    url = getattr(run, "url", None)
    print(f"[i] wandb run '{run_name or rid}' ({rid}){f' -> {url}' if url else ''}")
    # Release it immediately: no child may contend with the parent.
    finish()
    return rid


def child_env(env, stage_name=None, external=False):
    """Prepare a subprocess environment. Mutates and returns `env`."""
    if not enabled():
        return env
    if external:
        # GSFix3D and the GGPT worker call wandb.init() themselves without an
        # id -- and env beats defaults, so an inherited WANDB_RUN_ID would make
        # them hijack the pipeline run. They keep project/group/dir so their own
        # runs land alongside it.
        for key in (ENV_ENABLED, ENV_STAGE, "WANDB_RUN_ID", "WANDB_RESUME"):
            env.pop(key, None)
    elif stage_name:
        env[ENV_STAGE] = stage_name
    return env


def finish_pipeline(stage_seconds=None, total_seconds=None, exit_code=0):
    """Write the pipeline-level summary. Safe to call more than once."""
    global _PIPELINE_DONE
    if _PIPELINE_DONE or not enabled():
        return
    _PIPELINE_DONE = True

    if attach(job_type="pipeline") is None:
        return
    summary = {f"pipeline/seconds/{k}": v for k, v in (stage_seconds or {}).items()}
    if total_seconds is not None:
        summary["pipeline/total_seconds"] = total_seconds
    summary["pipeline/n_stages"] = len(stage_seconds or {})
    summary["pipeline/exit_code"] = exit_code
    log_summary(summary)
    finish(exit_code=exit_code)
