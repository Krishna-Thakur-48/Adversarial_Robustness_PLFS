# 🧠 Adversarial Robustness in Employment Classification (PLFS Dataset)

### 📍 IIT Kharagpur | Electrical Engineering – Software Systems Project  
**Author:** Krishna Thakur | **Year:** 2025  
**Tools:** PyTorch, Scikit-learn, NumPy, Pandas, Matplotlib  

---

## 🚀 Overview

This project explores **adversarial robustness** in predicting employment activity categories using the **Periodic Labour Force Survey (PLFS)** dataset — a large-scale socio-economic dataset from the Government of India.  

The objective is to understand:
- How **adversarial perturbations** (small, targeted input noise) affect employment classification models.  
- Whether **adversarial training** can make models more stable under such attacks.  
- How **activation smoothness** (e.g., Smoothed ReLU) further impacts robustness.

---

## 🧩 Project Structure
```

ADVERSARIAL_ROBUSTNESS_PLFS/
│
├── data/ # Raw + processed PLFS data
│ ├── plfs_processed_v*.csv
│ ├── train/val/test.csv
│ ├── X_all.npy, y_all.npy
│
├── notebooks/
│ ├── 1_pipeline.ipynb # Data pipeline, baseline, and FGSM ε-sweep
│ ├── 2_robustness.ipynb # Per-class robustness analysis
│ └── 3_smoothedReLU.ipynb # Activation smoothness and robustness comparison
│
├── results/
│ ├── accuracy_vs_eps.png # FGSM ε-sweep plot
│ ├── per_class_robustness.png # Per-class analysis visualization
│ ├── activation_bar_comparison.png # ReLU vs Softplus vs SmoothedReLU
│ ├── pytorch_model_std.pth / pytorch_model_adv.pth
│ ├── pytorch_summary.csv / fgsm_eps_sweep.csv
│ ├── activation_comparison_summary.csv
│ └── scaler.joblib, encoders.joblib, etc.
│
├── reports/
│ └── summary.md # Detailed multi-notebook findings and conclusions
│
└── Readme.md # Project overview (this file)
```


---

## ⚙️ Setup and Execution

### 1️⃣ Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```


2️⃣ Run the Notebooks
🧮 Notebook 1 — 1_pipeline.ipynb

Loads and processes PLFS data
Builds a logistic regression baseline
Trains a 2-layer PyTorch model
Performs FGSM adversarial attacks across multiple ε values
Compares standard vs. adversarially trained accuracy

📊 Notebook 2 — 2_robustness.ipynb

Evaluates per-class robustness (based on PAS employment categories)
Analyzes which classes gain or lose robustness under attack
Visualizes accuracy shifts across employment categories

⚡ Notebook 3 — 3_smoothedReLU.ipynb

Tests ReLU, Softplus, and Smoothed ReLU (q, ρ) activations
Evaluates the effect of activation smoothness on robustness
Finds the optimal activation setup for stable adversarial training

3️⃣ View Results

All outputs are automatically saved to the results/ folder.
A complete technical report is available at reports/summary.md

| FGSM ε | Standard Accuracy | Adv-Trained Accuracy | Δ Robustness |
| :----: | :---------------: | :------------------: | :----------: |
|  0.00  |       0.8230      |        0.8144        |    −0.0086   |
|  0.10  |       0.7889      |        0.7989        |    +0.0100   |
|  0.30  |       0.6671      |        0.7617        |    +0.0946   |


📈 Interpretation:
As perturbation strength (ε) increases, the standard model rapidly loses accuracy, while the adversarially trained model retains performance — demonstrating stronger robustness.

.

🧠 Insights Across the Three Notebooks
🔹 1️⃣ Core Observations

Adversarial training enhances robustness significantly under FGSM attacks.
Clean accuracy remains largely unaffected — robustness is gained without major trade-offs.
Data imbalance influences robustness: majority employment categories benefit more.

🔹 2️⃣ Per-Class Analysis

High-support classes (e.g., regular workers, self-employed, homemakers) show notable robustness improvements.
Low-support classes (e.g., students, unemployed, pensioners) remain fragile.
This indicates that robustness and fairness are connected — imbalance affects defense success.

🔹 3️⃣ Activation Smoothness

Smooth activations (Softplus, Smoothed ReLU) maintain comparable clean accuracy to ReLU.
Smoothed ReLU (q=3, ρ=1) achieved the highest adversarial accuracy (0.7952) after adversarial training.
Smooth activations lead to flatter loss landscapes and lower gradient magnitudes, reducing attack impact.

🔮 Future Work

To further improve both robustness and fairness:
Class-balanced adversarial training — to strengthen minority-category performance.
Stronger iterative attacks — e.g., PGD or AutoAttack for better adversarial benchmarks.
Feature sensitivity analysis — use SHAP or Integrated Gradients to explain fragile features.
Explore activation variants — e.g., Gaussian ReLU or smoother adaptive activations for enhanced stability.

Robust fairness metrics — explicitly measure robustness gaps between demographic subgroups.

🧾 References

Goodfellow et al. (2015). Explaining and Harnessing Adversarial Examples. arXiv:1412.6572
Madry et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR
Santurkar et al. (2019). How Does Batch Normalization Help Optimization? NeurIPS
PLFS (Periodic Labour Force Survey), Government of India

✅ Summary

This project shows that adversarial training significantly improves model stability in employment classification tasks using real-world data.
Moreover, architectural tweaks like Smoothed ReLU activations further enhance resilience — balancing model performance and robustness.

Together, these results provide insights for designing trustworthy, fair, and robust ML models in labour market analytics.
