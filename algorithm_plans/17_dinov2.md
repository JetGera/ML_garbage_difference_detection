## Plan: DINOv2 Cleanup Degree (0-100)

Goal update: estimate cleanup quality between before/after street photos as a continuous 0-100 cleanup percentage with highest possible precision on ~40 real pairs, using TACO to learn trash semantics. Initial release avoids a hard pass/fail threshold because all 40 pairs are assumed sufficiently cleaned and no negative calibration set is available.

**Steps**
1. Phase A - Problem framing and target definition.
2. Define target output from the DINOv2 method as `cleanup_percent` in [0,100], where higher means cleaner after-state relative to before.
3. Keep `enough/not enough` decision disabled in stage 1; expose only confidence bands (high/medium/low confidence) to avoid false certainty.
4. Phase B - Method integration scaffold.
5. Add a new method entry `dinov2_cd` to launcher registries and conda env mapping, following existing change-detection integration pattern.
6. Create `conda_envs/dinov2_cd.yml` with torch/torchvision/timm/opencv/albumentations and any DINOv2 dependency required by selected loading path.
7. Add `DinoV2CdRunner` dispatch in `launcher/runners.py:create_runner` with same guarded import style used by `changeformer` and `siamese_unet_cd`.
8. Phase C - High-precision zero-shot DINOv2 change signal.
9. Implement `launcher/method_scripts/dinov2_cd.py` with the same `AnalysisResult` contract and artifact-saving behavior as existing runners.
10. Reuse robust pair alignment from `ChangeformerRunner` (ECC/homography fallbacks + overlap mask) to minimize geometric false positives.
11. Extract multi-scale DINOv2 features for before/after and compute semantic change map (cosine/L2 dissimilarity with robust percentile normalization).
12. Produce primary output artifacts: semantic heatmap, overlay on after image, preview panel.
13. Derive robust proxy metrics from heatmap: `semantic_change_ratio`, `localized_change_mass`, `scene_consistency`, `alignment_quality`, `inference_ms`.
14. Phase D - Cleanup percent estimator (no hard threshold yet).
15. Convert raw signals into `cleanup_percent` using calibrated fusion:
16. Feature set: DINO semantic change map stats + existing `cleanup_delta` signal from EfficientNet pair classifier + overlap/alignment reliability.
17. Train a lightweight calibration model (isotonic regression or monotonic linear model) on pseudo-targets created from your known-good 40 pairs and TACO-derived synthetic perturbation pairs.
18. Output `cleanup_percent` plus `confidence` (based on alignment quality, domain distance, and model uncertainty).
19. Keep pass/fail unset; if needed, show recommended tentative bands only (e.g., <60 review, 60-80 uncertain, >80 likely good) marked as non-final.
20. Phase E - Precision-oriented validation with small data.
21. Build leave-one-pair-out validation across the 40 real pairs to estimate variance and prevent overfitting claims.
22. Run paired comparison vs `siamese_unet_cd` and `changeformer` on identical inputs; compare false-change behavior under illumination and viewpoint shifts.
23. Add stress tests with synthetic nuisance transforms (brightness, shadows, slight blur, minor affine drift) to quantify robustness.
24. Track calibration diagnostics: rank consistency, score spread, and instability across reruns.
25. Phase F - Decision stage readiness (future).
26. When negative/insufficient-cleaning examples become available, fit decision threshold(s) for `enough/not enough` with target precision-first constraint and explicit abstain zone.

**Relevant files**
- `c:\Coding\AI\projekt\launcher\runners.py` - register `dinov2_cd`, create runner dispatch.
- `c:\Coding\AI\projekt\launcher\methods.py` - method label/env metadata for GUI.
- `c:\Coding\AI\projekt\launcher\method_scripts\changeformer.py` - alignment, overlap, artifact and metric patterns to reuse.
- `c:\Coding\AI\projekt\launcher\method_scripts\efficientnet_cls.py` - reuse `cleanup_delta` concept as complementary global cleanliness signal.
- `c:\Coding\AI\projekt\conda_envs\dinov2_cd.yml` - new environment.
- `c:\Coding\AI\projekt\algorithm_plans\17_dinov2_change_detection.md` - new algorithm plan with cleanup-percent objective.
- `c:\Coding\AI\projekt\README.md` - add method and output semantics (0-100 cleanup percent).
- `c:\Coding\AI\projekt\install-conda-envs.ps1` - add usage example for new env.
- `c:\Coding\AI\projekt\test_dinov2_inference.py` - single-pair smoke + metric printout.
- `c:\Coding\AI\projekt\diagnose_raw_predictions.py` (or new diagnostics script) - aggregate cross-method comparison and calibration diagnostics.

**Verification**
1. Env smoke: `C:\ProgramData\anaconda3\Scripts\conda.exe run -n projekt-dinov2-cd python -c "import torch,timm,cv2;print(torch.__version__)"`.
2. Runner smoke on one pair: verify artifacts + metrics + non-empty `cleanup_percent`.
3. 40-pair batch run: produce CSV with per-pair `cleanup_percent`, confidence, alignment quality, and runtime.
4. LOPO validation: confirm score stability and identify outlier scenes; require variance within agreed tolerance before trusting scores.
5. Cross-method check: compare DINOv2 score behavior to Siamese/ChangeFormer and ensure fewer semantic false positives on lighting-only changes.

**Decisions**
- Included: cleanup estimate as continuous 0-100 score, DINOv2 + change detection + auxiliary cleanliness fusion.
- Excluded (for now): hard enough/not-enough threshold because no negative real pairs.
- Data assumption: ~40 real pairs are positive (sufficiently cleaned); TACO used to transfer trash semantics and generate robustness perturbations.

**Further Considerations**
1. Accuracy recommendation: prioritize `dinov2_vitb14` if GPU permits; fall back to `vits14` for speed.
2. Reliability recommendation: always weight down score when alignment quality or scene consistency is poor.
3. Product recommendation: add explicit `manual_review_recommended` flag when confidence is low, even if `cleanup_percent` is high.