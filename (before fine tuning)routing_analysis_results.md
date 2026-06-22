# Routing Analysis Results — Pre-Finetuning Baseline

> [!NOTE]
> These results are from a **randomly-initialized MoE router** on a pretrained DiT-XL/2 backbone. The routing patterns show the router's initial behavior before any MoE-specific training. This serves as the **baseline** to compare against after finetuning.

---

## 1. Expert Utilization Heatmap

![Expert utilization heatmap](/home/min/a/awasthi9/.gemini/antigravity/brain/40869116-784a-40bf-a780-f2c500f55cde/expert_utilization_heatmap.png)

### Key Observations

**Severe expert collapse in early MoE blocks (20-23):**
- Blocks 20-23 show 1-2 dominant experts capturing ~49% of all tokens each (e.g., Block 20: Expert 1 = 49%, Expert 5 = 31%)
- Several experts receive **0% of tokens** (e.g., Block 20: Experts 3,4 ≈ 0%)
- This is classic **expert collapse** — the randomly-initialized router concentrates on a few experts

**Better distribution in later blocks (24-27):**
- Blocks 24-27 show more spread across experts (e.g., Block 24: 5 experts each get 22-24%)
- Block 27 has the most balanced routing: E0=16%, E1=19%, E4=32%, E7=17%
- This suggests later blocks' feature representations are more diverse, making the router naturally spread tokens more evenly

**With 8 experts and top-2 routing, perfect balance = 0.25 per expert.** Most blocks are far from this, confirming the load-balancing auxiliary loss needs training to be effective.

---

## 2. Temporal Routing (Expert Usage Over Denoising Steps)

![Temporal routing](/home/min/a/awasthi9/.gemini/antigravity/brain/40869116-784a-40bf-a780-f2c500f55cde/temporal_routing.png)

### Key Observations

**Routing changes across denoising timesteps — even without training:**
- In **Block 20**, Expert 1 dominates at the start (high noise, t≈T) but Expert 5 gradually takes over around step 20-30
- In **Block 24**, there's a dramatic shift: Experts 0,6,7 dominate early (coarse structure) but by step 25-30, Experts 4,5 take over
- **Block 27** shows the most dynamic routing — multiple crossover points where different experts gain/lose dominance

**This is significant:** Even with random router weights, the router's behavior changes across timesteps because the **input features themselves change** as denoising progresses (high-noise → low-noise). This validates the Diff-MoE paper's core premise — timestep-aware routing is naturally useful because different denoising phases produce fundamentally different feature representations.

**After finetuning**, we expect these temporal patterns to become even sharper, with specific experts specializing for specific denoising phases (coarse structure experts vs. fine detail experts).

---

## 3. Router Entropy

![Router entropy](/home/min/a/awasthi9/.gemini/antigravity/brain/40869116-784a-40bf-a780-f2c500f55cde/router_entropy.png)

### Key Observations

**Most blocks have near-maximum entropy (≈2.08, max is 2.08 for 8 experts):**
- This means the **probability distribution** across experts is nearly uniform — the router assigns similar softmax probabilities to all experts
- The discrepancy between high entropy (nearly uniform probabilities) and the unbalanced utilization heatmap is because **small differences in probability get amplified by top-k selection** — even a 51% vs 49% split means one expert always wins

**Block 26 (green line) shows entropy dropping sharply at the end:**
- Entropy drops from ≈2.075 to ≈2.043 in the last 10 steps
- This means the router becomes more confident/specialized for final refinement steps
- This is the most interesting temporal signal — it suggests the last few denoising steps produce features that the router can most clearly differentiate

**Block 27 (yellow line) is consistently the lowest entropy:**
- The deepest MoE block has the most specialized routing throughout
- This makes intuitive sense — deeper layers have more abstract/processed features that are easier to specialize on

---

## 4. Spatial Expert Assignment

![Spatial routing](/home/min/a/awasthi9/.gemini/antigravity/brain/40869116-784a-40bf-a780-f2c500f55cde/spatial_routing.png)

### Key Observations

**Block 20**: Dominated by Expert 5 (orange) with scattered patches of other experts — shows initial bias but some spatial variation

**Block 21**: Two-expert split — Expert 3 (purple, upper-left) and Expert 7 (orange, rest) — interesting spatial partition even without training

**Block 22**: Mostly Expert 1 (orange) with scattered Expert 3 — less spatial structure

**Block 23**: Almost entirely Expert 1 (orange) — maximum expert collapse

**The spatial patterns are noisy but not random** — there are clusters and regions, suggesting the underlying features have some spatial structure that the router picks up even with random weights. After finetuning, we'd expect to see clearer semantic regions (e.g., object vs. background, edges vs. textures).

---

## 5. Generated Samples (Pre-Finetuning)

![Generated samples](/home/min/a/awasthi9/.gemini/antigravity/brain/40869116-784a-40bf-a780-f2c500f55cde/sample_diffmoe_analyzed.png)

The images show recognizable structure (golden retriever top-left, mountain landscape middle-right, white flowers bottom-right) but with heavy artifacts, especially color distortion. This is expected — the MoE blocks' experts are random, so they corrupt the features that the pretrained attention layers produce. After finetuning, quality should recover to near-vanilla-DiT levels while benefiting from expert specialization.

---

## Summary Table

| Metric | Finding | Implication |
|--------|---------|-------------|
| Expert utilization | Severe collapse in blocks 20-23 (1-2 experts get ~50% each) | Load-balancing loss needs training |
| Temporal routing | Expert dominance shifts across denoising steps | Validates timestep-aware routing premise |
| Router entropy | Near-uniform probabilities but top-k amplifies small differences | Router weights need training for clear specialization |
| Spatial routing | Noisy but non-random spatial patterns | Features have spatial structure the router can learn from |
| Entropy drop at end | Block 26-27 entropy drops in final steps | Fine-detail phase naturally needs more specialized routing |

---

## What Finetuning Should Change

1. **Load balance** → aux loss will push utilization toward 12.5% per expert (1/8)
2. **Temporal specialization** → experts will learn to specialize for specific denoising phases
3. **Spatial specialization** → experts will learn semantic regions (object/background/texture)
4. **Image quality** → should recover to near-vanilla or exceed (the point of MoE)
5. **Entropy** → expect lower entropy (more confident routing) especially in later blocks
