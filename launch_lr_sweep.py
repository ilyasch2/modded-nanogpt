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
    {
        "tag": "two_stage_300_exp_decayx2_const1000_1",
        "lr_muls": [1.52, 1.52, 1.52, 1.52],
        # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
        "cooldown_frac": 0.8075,
        "lr_decay_type": "exp",
        "lr_decay_switch_step": 1000,
        "lr_decay_second_type": "constant",
        "batch_sizes": CONSTANT_DOUBLE_FINAL_BATCH_SIZES,
    },
    {
        "tag": "two_stage_300_exp_decayx2_linear1000_1",
        "lr_muls": [1.52, 1.52, 1.52, 1.52],
        # cooldown starts at ~step 300 for NUM_SCHEDULED_ITERATIONS=1560
        "cooldown_frac": 0.8075,
        "lr_decay_type": "exp",
        "lr_decay_switch_step": 1000,
        "lr_decay_second_type": "linear",
        "batch_sizes": CONSTANT_DOUBLE_FINAL_BATCH_SIZES,
    },
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
        batch_sizes = cfg.get("batch_sizes", DEFAULT_BATCH_SIZES)
        env["BATCH_SIZES"] = ",".join(str(v) for v in batch_sizes)
        env["NUM_SCHEDULED_ITERATIONS"] = "1560"
        env["NUM_EXTENSION_ITERATIONS"] = "40"
        env["VAL_LOSS_EVERY"] = "10"
        env["VAL_LOSS_LAST_STEPS"] = "100"
        env["VAL_LOSS_EVERY_LAST"] = "10"
        env["LD_LIBRARY_PATH"] = (
            "/opt/conda/lib/python3.11/site-packages/nvidia/cusparselt/lib:"
            + env.get("LD_LIBRARY_PATH", "")
        )

        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
            "train_gpt.py",
        ]

        print(f"[{idx}/{len(CONFIGS)}] RUN_ID={run_id}")
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
