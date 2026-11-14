# C1: public-data OCT classifier versus multimodal LLMs for intraretinal fluid in DMO

Analysis code for the preregistered study "A compact public-data OCT classifier versus general-purpose
multimodal large language models for detecting intraretinal fluid in diabetic macular oedema: a preregistered
external comparison" (Yang X, Zhang L, Li X, Zhou G; manuscript submitted).

Preregistration: OSF osf.io/3276t (registered 10 November 2026, before the final model was trained and
before any test prediction).

## What this repository contains, and what it does not
- **Code only.** No images, reference standards, reader grades, multimodal LLM outputs, model weights or
  prediction probabilities are included.
- Public image data must be obtained from the original sources under their licences (all CC BY 4.0):
  OLIVES (Zenodo, doi:10.5281/zenodo.7105232), OCTDL (Mendeley/Zenodo, Kulyabin et al., Sci Data 2024),
  TowardPi DMO (figshare, doi:10.6084/m9.figshare.30582035; Huang et al., Sci Data 2026).
- The frozen reference standards come from an earlier, separately preregistered study (osf.io/pbrq6) and
  will be released, together with the model weights (with SHA-256) and predictions, on publication.
- Code comments are partly in Chinese; function names and outputs are in English or self-explanatory.

## Pipeline (run order)
| Step | File | Where it ran | Purpose |
|---|---|---|---|
| 0 | code/c00_prep_external.py | local computer | Package the 305 external test images and a manifest (image_id, set, path); no reference labels |
| 1 | code/c01_colab_olives.py, code/c01a_fetch_inner.py | Google Colab | Extract OLIVES non-TREX training B-scans (13-37 of each volume) by HTTP range requests; exclude any TREX-DME patient |
| 2 | code/c02_train.py (uses code/common.py) | Google Colab, T4 GPU | `cv`: patient-grouped 5-fold cross-validation for model selection; `final`: train the final model and hash its weights |
| 3 | code/c03_predict.py | Google Colab, T4 GPU | Predict the external images once with the frozen model and write a seal file with SHA-256 hashes; no reference labels in the cloud |
| 4 | code/c04_evaluate.py | local computer | Fix thresholds on the development set; compute AC1, sensitivity, specificity, AUC and paired bootstrap differences versus the MLLMs and readers |
| 5 | code/c05_fig2_roc_v3.py | local computer | Figure 2 (ROC curves with MLLM operating points); plotting only |
| 6 | code/c06_post_hoc_calibration.py | local computer | Post hoc, not preregistered: C-grade distribution and calibration (intercept, slope, Brier, reliability plot) |

Final model setting: ConvNeXt-Tiny (ImageNet weights), learning rate 1e-4, binary cross-entropy, 2 epochs,
384 x 384 input, seed 20261107, pos_weight disabled. Bootstrap: 2,000 cluster resamples, seed 20261023.

Typical commands are given in the docstring at the top of each script. Paths inside c00 and c04-c06 follow the
authors' local folder layout and must be adapted.

## Environment
See requirements.txt. Analysis (c04-c06): Python 3.10.21, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2,
SciPy 1.15.2. Final training and prediction (c02 final, c03): Python 3.13.16, PyTorch 2.14.1+cu130,
torchvision 0.29.1+cu130 on an NVIDIA T4 (Google Colab). Library versions were not recorded for the
cross-validation runs.

## Integrity
SHA256SUMS lists the SHA-256 of every file in this release. The versions of common.py, c02_train.py and
c03_predict.py are those named in the preregistration (SHA-256 prefixes 019076296387e131, d40a4acf60857797
and dd247a3ccf488139).

## Licence
MIT (see LICENSE).
