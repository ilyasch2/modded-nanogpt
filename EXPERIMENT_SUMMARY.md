# Spectral Learning Experiment - Implementation Summary

**Date:** 2026-02-06
**Objective:** Test whether making spectral values learnable in the optimizer improves training

---

## 🎯 What We're Testing

### Hypothesis
The current Muon optimizer forces all singular values to 1 (flattening the spectrum). By allowing singular values to be learned via gradient decomposition, we may achieve:
1. Better convergence (faster to target loss)
2. Better final performance (lower final loss)
3. Network-level optimization (layers learn their optimal spectral profiles)

### Implementation: Orthogonal-Radial Decomposition (ORD)

We decompose each gradient into two components:

```python
# Decompose gradient G into:
ρ = ⟨G, W⟩ / ||W||²
G_parallel = ρ · W      # Energy component (scales singular values)
G_geometric = G - G_∥   # Rotation component (changes singular vectors)

# Apply different optimizations:
Update = Polar(G_geometric) + spectral_lr * G_parallel
```

**Key properties:**
- `spectral_lr = 0`: Pure geometric Muon (exact match to standard Muon baseline)
- `spectral_lr = 1`: Full ORD with learnable spectral values
- Zero computational overhead (just dot product + subtraction)

---

## 🔬 Experimental Design

### Three Configurations

1. **baseline_std_0** (train_gpt.py)
   - Standard optimizer (no Muon)
   - Control baseline

2. **spectral_lr_0** (train_gpt_muon_new.py, SPECTRAL_LR_MUL=0.0)
   - Muon with spectral learning disabled
   - Should exactly match previous Muon implementation
   - Tests that new code doesn't break anything

3. **spectral_lr_1** (train_gpt_muon_new.py, SPECTRAL_LR_MUL=1.0)
   - Muon with full spectral learning
   - The novel method we're testing

### Controlled Variables

**All three runs use IDENTICAL:**
- Learning rate schedule: [1.0, 1.52, 1.73, 1.0]
- Cooldown fraction: 0.55
- Batch sizes: [131072, 262144, 393216, 393216]
- Number of steps: 1560 scheduled + 40 extension
- Validation frequency: Every 50 steps (for faster analysis)
- Random seed: (same across runs)
- Hardware: 8 GPUs

**Only difference:** The `SPECTRAL_LR_MUL` parameter (0.0 vs 1.0 vs baseline)

---

## 📝 Code Changes

### 1. train_gpt_muon_new.py (lines 742-769)

**Before (branching logic):**
```python
if p_cfg.spectral_lr_mul is not None and p_cfg.spectral_lr_mul != 0:
    # Decompose
    ...
    g_geometric = updated_grads - g_parallel
else:
    g_geometric = updated_grads

v_chunk = polar_express(g_geometric)

if g_parallel is not None:
    v_chunk = v_chunk + g_parallel * p_cfg.spectral_lr_mul
```

**After (cleaner, unified):**
```python
spectral_lr = p_cfg.spectral_lr_mul if p_cfg.spectral_lr_mul is not None else 0.0

if spectral_lr != 0:
    # Decompose
    g_parallel = rho * p_slice
    g_geometric = updated_grads - g_parallel
else:
    # Exact standard Muon when spectral_lr=0
    g_parallel = torch.zeros_like(updated_grads)
    g_geometric = updated_grads

v_chunk = polar_express(g_geometric)
v_chunk = v_chunk + g_parallel * spectral_lr  # Unconditional add
```

**Why this is better:**
- ✅ Cleaner: Only one branch (decomposition), not two
- ✅ Correct: `spectral_lr=0` gives exact standard Muon
- ✅ Testable: Easy to verify equivalence

### 2. launch_lr_sweep.py

Added three configs:
```python
{
    "tag": "baseline_std_0",
    "script": "train_gpt.py",
    "val_loss_every": 50,
},
{
    "tag": "spectral_lr_0",
    "script": "train_gpt_muon_new.py",
    "spectral_lr_mul": 0.0,
    "val_loss_every": 50,
},
{
    "tag": "spectral_lr_1",
    "script": "train_gpt_muon_new.py",
    "spectral_lr_mul": 1.0,
    "val_loss_every": 50,
},
```

---

## 📊 What to Look For

### Success Criteria

**spectral_lr_1 is better if:**
1. ✅ Lower final validation loss (even by 0.01-0.02)
2. ✅ Faster convergence (reaches target loss in fewer steps)
3. ✅ More stable (no divergence, similar training time)

**Neutral result if:**
- spectral_lr_1 matches spectral_lr_0 within noise (~0.01 loss)
- This would suggest spectral learning doesn't help, but doesn't hurt

**Failure if:**
- spectral_lr_1 has higher loss or diverges
- spectral_lr_1 is much slower (>2x training time)

### Validation Check

**Critical:** spectral_lr_0 should match previous Muon runs exactly.
- If not, there's a bug in the refactored code
- Check that the decomposition is properly skipped when spectral_lr=0

---

## 🔍 Monitoring

### During Training

Check progress:
```bash
python monitor_runs.py
```

View logs:
```bash
tail -f logs/lrmul-1-1.52-1.73-1-cd0.55-spectral_lr_1-*.txt
```

### After Training

Compare final losses:
```bash
python plot_loss.py
```

---

## 📚 Scientific Context

### Related Work

1. **Muon** (Jordan et al., 2024)
   - Applies Polar decomposition to full gradient
   - Forces all singular values → 1
   - We modify: Apply Polar to G_geometric only

2. **SVD Parameterization** (Zhang et al., 2018)
   - Explicit W = UΣV^T parameterization
   - Requires SVD (expensive)
   - We do implicit via gradient projection (cheap)

3. **Hypergradient Descent** (Baydin et al., 2018)
   - Learn hyperparameters online
   - Future work: Make spectral_lr learnable per layer

### Theoretical Insight

**Why this creates network-level learning:**

Even though we apply ORD per-layer, it creates global coupling:
- Layer L's singular values affect gradient flow to Layer L-1 (chain rule)
- Layers self-organize into "gradient highways"
- "Valve effect": Layers gate each other's gradients via spectral choices

**Mathematical form:**

Standard Muon: `W ← W - η · Polar(G)`
Our method: `W ← W - η · [Polar(G_⊥) + α·G_∥]`

Where α = spectral_lr_mul controls the balance.

---

## 🚀 Next Steps (After Results)

### If spectral_lr_1 wins:
1. Test other values: spectral_lr ∈ {0.5, 2.0, 5.0}
2. Make spectral_lr per-layer (different for attention vs MLP)
3. Make spectral_lr adaptive (coherence-gating)
4. Scale to larger models (1B+)
5. Write paper

### If spectral_lr_1 = spectral_lr_0 (neutral):
1. Add diagnostic logging (what is ρ? what is spectral_frac?)
2. Check if gradients are naturally orthogonal to weights (G_∥ ≈ 0)
3. Try higher spectral_lr (maybe 1.0 is too conservative)
4. Test on different task (maybe FineWeb has this property, try another dataset)

### If spectral_lr_1 fails:
1. Debug: Is decomposition correct?
2. Check: Should variance reduction apply to G_∥?
3. Theory: Maybe Polar(G_⊥) is worse than Polar(G)?
4. Save for later: "Do you even want standard Muon?" experiment

---

## 📌 Open Questions (Saved for Later)

**Geometric vs Full Gradient Muon:**

Current baseline: `Polar(G)` (standard Muon)
Alternative 1: `Polar(G_⊥)` (geometric-only)
Alternative 2: `Polar(G_⊥) + G_∥` (our method)

**Question:** Is `Polar(G_⊥)` actually better than `Polar(G)`?

**Experiment:** Four-way comparison:
1. Polar(G) [standard Muon]
2. Polar(G_⊥) [geometric only]
3. Polar(G_⊥) + 0.5·G_∥ [half spectral]
4. Polar(G_⊥) + 1.0·G_∥ [full spectral]

This would tell us if the decomposition itself is beneficial, independent of learning.

---

## 🎓 Key Learnings

### Implementation
- ✅ Always prefer unified code paths (use multiplication by zero instead of branching)
- ✅ Controlled experiments require EXACT matching of all hyperparameters
- ✅ Document what "baseline" means precisely

### Mathematics
- Gradient decomposition is cheap (just projection)
- Spectral learning happens implicitly via G_∥ component
- Network-level coupling emerges from local layer updates

### Experimentation
- Start with minimal viable test (just 0.0 vs 1.0)
- Only add complexity after base case works
- Log everything for post-hoc analysis

---

**Status:** Experiments running (started 2026-02-06 22:45)
**ETA:** ~90 minutes (7 min warmup + ~1 hour training × 3 runs)
**Next:** Monitor with `python monitor_runs.py`
