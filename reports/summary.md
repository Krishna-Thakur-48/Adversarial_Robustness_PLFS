# 🧠 Adversarial Robustness in Employment Classification using PLFS Data

### 📍 IIT Kharagpur | Electrical Engineering (Software Systems Project)  
**Author:** Krishna Thakur (2025)  
**Tools:** PyTorch, Scikit-learn, NumPy, Pandas, Matplotlib  

---

## 1️⃣ Project Overview

This project investigates **adversarial robustness** in predicting employment activity categories using the **Periodic Labour Force Survey (PLFS)** dataset (Government of India).  
The goal is to study how small, crafted perturbations in input data (adversarial attacks) can degrade model performance and how **adversarial training** can make such models more stable and reliable.

Three major notebooks guide this analysis:
1. `1_pipeline.ipynb` – Data preprocessing, baseline model, FGSM attack pipeline  
2. `2_robustness.ipynb` – Per-class robustness analysis and visual evaluation  
3. `3_smoothedReLU.ipynb` – Exploring smooth activation functions for improved robustness  

---

## 2️⃣ Dataset and Features

We use the processed PLFS microdata (2019–2023), which contains individual-level socio-economic and employment information.

### 🧩 Input Features
| Type | Features |
|------|-----------|
| **Demographic (Robust)** | `AGE`, `SEX`, `GEDU_LVL`, `ST`, `SEC`, `MARST`, `REL` |
| **Economic (Non-Robust)** | `ERN_REG`, `ERN_SELF`, `total_weekly_earnings`, `VOC`, `total_work_hours`, `total_available_hours`, `work_intensity` |

### 🎯 Target Variable
`PAS` – *Principal Activity Status*, indicating employment category (14 possible classes such as self-employed, regular salaried, student, unemployed, homemaker, etc.).

---

## 3️⃣ Notebook 1 — Baseline and FGSM Adversarial Training

### ⚙️ Model Setup
- **Architecture:** 2-layer fully connected neural network  
- **Input Dim:** 13 features  
- **Hidden Units:** 256  
- **Output Classes:** 14  
- **Optimizer:** Adam (lr = 1e−3)  
- **Loss:** Cross-Entropy  

### ⚔️ Adversarial Training
The **Fast Gradient Sign Method (FGSM)** was applied to generate adversarial samples with varying strengths (`ε`), and the model was retrained on these perturbed inputs.

### 📊 Results (Clean vs Adversarial Accuracy)

| ε (FGSM strength) | Standard Model | Adv-Trained Model | Δ Robustness |
|-------------------|----------------|-------------------|---------------|
| 0.00 | 0.8230 | 0.8144 | −0.0086 |
| 0.05 | 0.8074 | 0.8065 | −0.0009 |
| 0.10 | 0.7889 | 0.7989 | +0.0100 |
| 0.20 | 0.7376 | 0.7834 | +0.0458 |
| 0.30 | 0.6671 | 0.7617 | +0.0946 |

**Interpretation:**  
As attack strength increases, standard models lose performance quickly, while adversarially trained models degrade much slower — demonstrating **enhanced robustness** without significant loss in clean accuracy.

---

## 4️⃣ Notebook 2 — Per-Class Robustness Analysis

To understand which employment categories benefit most from adversarial training, we analyzed robustness **per `PAS` class** at `ε = 0.25`.

| Category (PAS) | Standard Adv Acc | Adv-Trained Adv Acc | Observation |
|----------------|------------------|---------------------|--------------|
| 91 – Homemakers | 0.84 → 0.97 | Strong gain (+12.6%) |
| 92 – Domestic duties / free collection | 0.55 → 0.73 | +17% improvement |
| 21 – Regular salaried | 0.32 → 0.42 | Moderate improvement |
| 94 – Too old / disabled | 0.51 → 0.62 | +10% improvement |
| Low-support classes (Students, Unemployed, Pensioners) | <0.20 | Still fragile |

**Insights:**
- Adversarial training **stabilizes major employment classes** (high data frequency).  
- **Minority categories** remain vulnerable, revealing that data imbalance limits robustness.  
- The robustness gain correlates with sample size — showing that frequent classes learn more stable representations.

**Conclusion:**  
Adversarial training enhances stability for high-support categories but highlights fairness challenges for underrepresented groups.

---

## 5️⃣ Notebook 3 — Activation Function Study (Smoothed ReLU)

To explore architectural factors affecting robustness, we compared:
- **ReLU**
- **Softplus** (smooth exponential approximation)
- **Smoothed ReLU (q, ρ)** — a tunable smooth variant inspired by theoretical robustness literature.

### 📈 Activation Comparison Results (ε = 0.1)

| Activation | q | ρ | Clean Acc (Std) | Adv Acc (Std) | Clean Acc (Adv-Trained) | Adv Acc (Adv-Trained) |
|-------------|---|---|----------------:|---------------:|-------------------------:|-----------------------:|
| ReLU | 3.0 | 1.0 | 0.808 | 0.766 | 0.814 | 0.794 |
| Softplus | 3.0 | 1.0 | **0.817** | **0.774** | 0.811 | 0.791 |
| Smoothed ReLU | 3.0 | 1.0 | 0.813 | 0.773 | **0.814** | **0.795** |
| Smoothed ReLU | 4.0 | 0.5 | 0.815 | 0.772 | 0.812 | 0.795 |

**Findings:**
- Smooth activations achieve **comparable clean accuracy** to ReLU.
- After adversarial training, **Smoothed ReLU (q=3, ρ=1)** achieved the **highest robust accuracy (0.795)**.
- The smoother activation surface reduces gradient spikes, leading to **flatter loss landscapes** and **better stability under attack**.
- Softplus provides the best clean performance, while Smoothed ReLU offers the most balanced robustness.

**Visualization:**  
📊 `results/activation_bar_comparison.png`  
📈 `results/activation_quick_eps_sweep.png`

---

## 6️⃣ Overall Findings

✅ Adversarial training meaningfully improves model robustness against FGSM perturbations.  
✅ Class imbalance is a key factor limiting robustness — rare categories remain more fragile.  
✅ Smoother activations (Softplus, Smoothed ReLU) help further stabilize adversarially trained models.  
✅ Robustness and fairness are linked: better data balance could yield more equitable model performance.

---

## 7️⃣ Future Work

To strengthen and extend this research:
1. **Use stronger iterative attacks (PGD, AutoAttack)** for more rigorous robustness validation.  
2. **Apply class-balanced adversarial training** to address minority-class vulnerability.  
3. **Analyze feature importance** using explainable ML tools (SHAP, Integrated Gradients).  
4. **Experiment with regularized or noise-injected activations** (e.g., Gaussian ReLU, Smoothed ReLU variants).  

---

## 8️⃣ Project Artifacts

| Folder | Description |
|---------|--------------|
| `data/` | Raw and processed PLFS data (`plfs_processed_v*.csv`, train/val/test splits) |
| `notebooks/` | All analysis notebooks (`1_pipeline.ipynb`, `2_robustness.ipynb`, `3_smoothedReLU.ipynb`) |
| `results/` | Trained weights, CSV metrics, adversarial plots |
| `reports/` | Final markdown summaries (this file) |
| `Readme.md` | Quickstart and project overview |

---

## 9️⃣ Final Conclusion

This study demonstrates that **adversarially trained neural networks** can maintain accuracy under noise and attacks in real-world socio-economic classification.  
The improvement is strongest in **well-represented employment groups**, while low-support groups expose the **equity limits** of robustness.  
Finally, the **Smoothed ReLU** experiment shows that subtle architectural choices — such as activation smoothness — can further improve adversarial stability, paving the way for **fairer, more robust machine learning in labour analytics**.

---

📚 **References**
1. Goodfellow et al. (2015). *Explaining and Harnessing Adversarial Examples*. arXiv:1412.6572.  
2. Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.  
3. Santurkar et al. (2019). *How Does Batch Normalization Help Optimization?* (on loss smoothness).  
4. PLFS (Periodic Labour Force Survey), Government of India.  
