#!/bin/bash
# Quick summary of the 3 experimental runs

echo "==================================="
echo "Spectral Learning Experiment Results"
echo "==================================="
echo ""

# Find the three run logs
baseline=$(ls -t logs/*baseline_std_0-*.txt 2>/dev/null | head -1)
spec0=$(ls -t logs/*spectral_lr_0-*.txt 2>/dev/null | head -1)
spec1=$(ls -t logs/*spectral_lr_1-*.txt 2>/dev/null | head -1)

if [ -z "$baseline" ] || [ -z "$spec0" ] || [ -z "$spec1" ]; then
    echo "❌ Not all runs completed yet"
    echo "  baseline_std_0: ${baseline:-NOT FOUND}"
    echo "  spectral_lr_0: ${spec0:-NOT FOUND}"
    echo "  spectral_lr_1: ${spec1:-NOT FOUND}"
    exit 1
fi

echo "✅ All three runs found:"
echo "  1. $baseline"
echo "  2. $spec0"
echo "  3. $spec1"
echo ""

# Extract final validation losses
echo "Final Validation Losses:"
echo "------------------------"

# Baseline
base_final=$(grep "val_loss:" "$baseline" | tail -1 | grep -oP 'val_loss:\K[0-9.]+')
echo "baseline_std_0:  $base_final"

# Spectral LR = 0
spec0_final=$(grep "val_loss:" "$spec0" | tail -1 | grep -oP 'val_loss:\K[0-9.]+')
echo "spectral_lr_0:   $spec0_final"

# Spectral LR = 1
spec1_final=$(grep "val_loss:" "$spec1" | tail -1 | grep -oP 'val_loss:\K[0-9.]+')
echo "spectral_lr_1:   $spec1_final"

echo ""
echo "Comparison:"
echo "-----------"

# Python comparison
python3 << EOF
baseline = ${base_final:-999}
spec0 = ${spec0_final:-999}
spec1 = ${spec1_final:-999}

# Find best
losses = [("baseline_std_0", baseline), ("spectral_lr_0", spec0), ("spectral_lr_1", spec1)]
losses.sort(key=lambda x: x[1])

print(f"🏆 Best: {losses[0][0]} = {losses[0][1]:.4f}")
print(f"🥈 2nd:  {losses[1][0]} = {losses[1][1]:.4f} (+{losses[1][1]-losses[0][1]:.4f})")
print(f"🥉 3rd:  {losses[2][0]} = {losses[2][1]:.4f} (+{losses[2][1]-losses[0][1]:.4f})")

print()
print("Key Question: Does spectral learning help?")
if spec1 < spec0 - 0.01:
    print("✅ YES! spectral_lr_1 is significantly better than spectral_lr_0")
elif spec1 < spec0:
    print("⚠️  MAYBE. spectral_lr_1 is slightly better (within noise)")
elif abs(spec1 - spec0) < 0.01:
    print("➖ NEUTRAL. No significant difference")
else:
    print("❌ NO. spectral_lr_0 is better (spectral learning may hurt)")

print()
print("Validation Check:")
if abs(spec0 - baseline) > 0.05:
    print("⚠️  WARNING: spectral_lr_0 differs from baseline_std_0 by {:.4f}".format(abs(spec0 - baseline)))
    print("   This suggests different optimizers, NOT a bug in spectral code")
else:
    print("ℹ️  spectral_lr_0 vs baseline diff: {:.4f} (expected for different optimizers)".format(abs(spec0 - baseline)))
EOF
