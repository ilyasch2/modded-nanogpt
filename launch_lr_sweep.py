#!/usr/bin/env python3
import datetime
import os
import subprocess

# Edit this list to try different LR schedule hyperparameters.
DEFAULT_BATCH_SIZES = [131072, 262144, 393216, 393216]
CONSTANT_FINAL_BATCH_SIZES = [131072, 262144, 262144, 262144]
CONSTANT_DOUBLE_FINAL_BATCH_SIZES = [131072, 262144, 524288, 524288]
CONFIGS = [
    # {
    #     "tag": "baseline_std",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt.py",
    # },
    # {
    #     "tag": "baseline_muon_new",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_muon_new.py",
    # },
    # Previous runs (baseline comparison)
    # {
    #     "tag": "baseline_std",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt.py",
    #     "val_loss_every": 50,
    # },
    # {
    #     "tag": "spectral_lr_0.1",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_muon_new.py",
    #     "spectral_lr_mul": 0.1,
    #     "val_loss_every": 50,
    # },
    # {
    #     "tag": "spectral_lr_0.5",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_muon_new.py",
    #     "spectral_lr_mul": 0.5,
    #     "val_loss_every": 50,
    # },

    # Fine-grained sweep between 0 and 0.3
    # {
    #     "tag": "baseline_muon_init",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_init.py",
    #     "val_loss_every": 50,
    # },
    # {
    #     "tag": "spectral_lr_0.05",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_muon_new.py",
    #     "spectral_lr_mul": 0.05,
    #     "val_loss_every": 50,
    # },
    # {
    #     "tag": "spectral_lr_0.08",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_muon_new.py",
    #     "spectral_lr_mul": 0.08,
    #     "val_loss_every": 50,
    # },
    # {
    #     "tag": "spectral_lr_0.15",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_muon_new.py",
    #     "spectral_lr_mul": 0.15,
    #     "val_loss_every": 50,
    # },

    # New fine-grained sweep: 0.02, 0.03, 0.05, 0.07, 0.12, 0.15, 0.2, 0.25, 0.3
    {
        "tag": "spectral_lr_0.02",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.02,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.03",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.03,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.05",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.05,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.07",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.07,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.12",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.12,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.15",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.15,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.2",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.2,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.25",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.25,
        "val_loss_every": 50,
    },
    {
        "tag": "spectral_lr_0.3",
        "lr_muls": [1.0, 1.52, 1.73, 1.0],
        "cooldown_frac": 0.55,
        "batch_sizes": DEFAULT_BATCH_SIZES,
        "script": "train_gpt_muon_new.py",
        "spectral_lr_mul": 0.3,
        "val_loss_every": 50,
    },
    # {
    #     "tag": "baseline_muon_new_spec0",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    #     "script": "train_gpt_muon_new.py",
    #     "spectral_lr_mul": 0.0,
    # },
    # {
    #     "tag": "baseline",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    # },
    # {
    #     "tag": "baseline_001",
    #     "lr_muls": [1.0, 1.52, 1.73, 1.0],
    #     "cooldown_frac": 0.55,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    # },
    # {
    #     "tag": "hi_stage1",
    #     "lr_muls": [2.0, 1.45, 1.65, 1.0],
    #     "cooldown_frac": 0.55,
    # },
    # {
    #     "tag": "two_stage_300_1",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.8075,
    # },

    #### AAQAAA
    # {
    #     "tag": "two_stage_early_decay",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.9075,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    # },
    # {
    #     "tag": "two_stage_late_decay",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.6675,
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    # },
    
    # ##### exppppp
    # {
    #     "tag": "two_stage_300_exp_decay",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.8075,
    #     "lr_decay_type": "exp",
    #     "batch_sizes": DEFAULT_BATCH_SIZES,
    # },


    # {
    #     "tag": "two_stage_300_exp_decay",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.8075,
    #     "lr_decay_type": "exp",
    #     "batch_sizes": CONSTANT_FINAL_BATCH_SIZES,
    # },
    # {
    #     "tag": "two_stage_300_exp_decayx2",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.8075,
    #     "lr_decay_type": "exp",
    #     "batch_sizes": CONSTANT_DOUBLE_FINAL_BATCH_SIZES,
    # },
    # {
    #     "tag": "two_stage_300_exp_decayx2_const1000_1",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.8075,
    #     "lr_decay_type": "exp",
    #     "lr_decay_switch_step": 1000,
    #     "lr_decay_second_type": "constant",
    #     "batch_sizes": CONSTANT_DOUBLE_FINAL_BATCH_SIZES,
    # },
    # {
    #     "tag": "two_stage_300_exp_decayx2_linear1000_1",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.8075,
    #     "lr_decay_type": "exp",
    #     "lr_decay_switch_step": 1000,
    #     "lr_decay_second_type": "linear",
    #     "batch_sizes": CONSTANT_DOUBLE_FINAL_BATCH_SIZES,
    # },
    # {
    #     "tag": "two_stage_1000_constx2",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 1000 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.358974358974,
    #     "lr_decay_type": "linear",
    #     "lr_decay_final": 1.52,
    #     "batch_sizes": CONSTANT_DOUBLE_FINAL_BATCH_SIZES,
    # },
    # {
    #     "tag": "two_stage_1000_linearx2",
    #     "lr_muls": [1.52, 1.52, 1.52, 1.52],
    #     # cooldown starts at ~step 1000 for NUM_SCHEDULED_ITERATIONS=1560
    #     "cooldown_frac": 0.358974358974,
    #     "lr_decay_type": "linear",
    #     "batch_sizes": CONSTANT_DOUBLE_FINAL_BATCH_SIZES,
    # },

    ##### ZZZZZZ

    #####
    # {
    #     "tag": "low_stage2_s3",
    #     "lr_muls": [1.0, 1.25, 1.45, 1.0],
    #     "cooldown_frac": 0.55,
    # },
    # {
    #     "tag": "high_stage2_s3",
    #     "lr_muls": [1.0, 1.75, 2.00, 1.0],
    #     "cooldown_frac": 0.55,
    # },
    # {
    #     "tag": "early_cooldown",
    #     "lr_muls": [1.0, 1.45, 1.65, 1.0],
    #     "cooldown_frac": 0.70,
    # },
    # {
    #     "tag": "late_cooldown",
    #     "lr_muls": [1.0, 1.45, 1.65, 1.0],
    #     "cooldown_frac": 0.40,
    # },
]


def _fmt_lr_muls(values: list[float]) -> str:
    return "-".join(f"{v:g}" for v in values)


def _run_id(cfg: dict, timestamp: str) -> str:
    lr_tag = _fmt_lr_muls(cfg["lr_muls"])
    cd_tag = f"{cfg['cooldown_frac']:g}"
    return f"lrmul-{lr_tag}-cd{cd_tag}-{cfg['tag']}-{timestamp}"


def main() -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    nproc = int(os.environ.get("NPROC_PER_NODE", "8"))

    for idx, cfg in enumerate(CONFIGS, start=1):
        run_id = _run_id(cfg, timestamp)
        env = os.environ.copy()
        env["RUN_ID"] = run_id
        env["LR_MULS"] = ",".join(str(v) for v in cfg["lr_muls"])
        env["COOLDOWN_FRAC"] = str(cfg["cooldown_frac"])
        env["LR_DECAY_TYPE"] = cfg.get("lr_decay_type", "linear")
        if "lr_decay_final" in cfg:
            env["LR_DECAY_FINAL"] = str(cfg["lr_decay_final"])
        if "lr_decay_switch_step" in cfg:
            env["LR_DECAY_SWITCH_STEP"] = str(cfg["lr_decay_switch_step"])
        if "lr_decay_second_type" in cfg:
            env["LR_DECAY_SECOND_TYPE"] = str(cfg["lr_decay_second_type"])
        if "lr_decay_second_final" in cfg:
            env["LR_DECAY_SECOND_FINAL"] = str(cfg["lr_decay_second_final"])
        if "spectral_lr_mul" in cfg:
            env["SPECTRAL_LR_MUL"] = str(cfg["spectral_lr_mul"])
        batch_sizes = cfg.get("batch_sizes", DEFAULT_BATCH_SIZES)
        env["BATCH_SIZES"] = ",".join(str(v) for v in batch_sizes)
        env["NUM_SCHEDULED_ITERATIONS"] = "1560"
        env["NUM_EXTENSION_ITERATIONS"] = "40"
        env["VAL_LOSS_EVERY"] = str(cfg.get("val_loss_every", 10))
        env["VAL_LOSS_LAST_STEPS"] = "100"
        env["VAL_LOSS_EVERY_LAST"] = "10"
        env["LD_LIBRARY_PATH"] = (
            "/opt/conda/lib/python3.11/site-packages/nvidia/cusparselt/lib:"
            + env.get("LD_LIBRARY_PATH", "")
        )

        script = cfg.get("script", "train_gpt.py")
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            script,
        ]

        print(f"[{idx}/{len(CONFIGS)}] RUN_ID={run_id}")
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
