# Breast Cancer Classification using Neural Network 🧠

A production-minded, end-to-end binary classification project that predicts whether a breast tumor is benign or malignant using the Breast Cancer Wisconsin Diagnostic Dataset. This repository demonstrates data loading, preprocessing, feature scaling, ANN model building, training, evaluation, and visualization — packaged for reproducibility and fast evaluation.

---

Contents
- 🔥 Highlights
- 📊 Dataset Description
- 🧠 How the Model Works
- 🧪 Project Structure
- ⚙️ Installation & Usage
- 📈 Training Visualizations
- 🧾 Evaluation & Metrics
- 🛠️ Future Improvements
- 📁 License
- 🙏 Credits

---

🔥 Highlights
- Clean, well-documented end-to-end pipeline for a tabular classification problem.
- High-performing Neural Network with test accuracy: **95.61%**.
- Reproducible artifacts: trained model, visualizations, and example scripts for training and inference.
- Designed to be recruiter- and production-friendly: clear structure, example usage, and evaluation.

---

📊 Dataset Description
- Source: Breast Cancer Wisconsin (Diagnostic) Dataset (tabular CSV).
- Samples: Classic diagnostic measurements extracted from digitized images of FNA of breast masses.
- Features: 30 real-valued input features (mean, se, worst measurements for various cell nucleus attributes).
- Target: Binary label — 0 = benign, 1 = malignant.

---

🧠 How the Model Works
This project follows an end-to-end supervised workflow:

1. Load the CSV dataset (data.csv).
2. Preprocess:
   - Handle missing values if any.
   - Encode labels (benign/malignant → 0/1).
   - Split into train / validation / test sets.
   - Standardize / normalize features (fit scaler on train → apply to val/test).
3. Build the ANN:
   - Input: 30 normalized features
   - Hidden layers: Dense layers with ReLU activation, batch normalization (optional), and dropout (optional)
   - Output: 1 neuron with Sigmoid activation
   - Loss: Binary crossentropy
   - Optimizer: Adam
4. Train the model while tracking loss & accuracy.
5. Evaluate on the test set and visualize training curves + confusion matrix.
6. Save model to models/breast_cancer_nn.h5 for inference.

Model architecture (simple diagram)
- Input (30 features)  
  ↓
- Dense (units → ReLU)  
  ↓
- Dense (units → ReLU)  
  ↓
- Dense (1 → Sigmoid)  — Binary prediction (probability)

Bullet architecture (example)
- Input layer: 30 normalized features
- Hidden layer 1: Dense(16) → ReLU
- Hidden layer 2: Dense(8) → ReLU
- Output layer: Dense(1) → Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam

---

🧪 Project Structure
```
.
├── notebooks/
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
├── models/
│   └── breast_cancer_nn.h5
├── data.csv
├── inference.py
├── training_and_evaluation.py
├── requirements.txt
└── README.md
```

---

⚙️ Installation & Usage

Prerequisites
- Python 3.8+ recommended
- Git (optional, to clone repo)

1) Create and activate virtual environment (recommended)
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run the Jupyter notebook
- Open the project notebook(s) for exploration, preprocessing code, and visualization:
```bash
jupyter lab
# or
jupyter notebook
```
Open the relevant notebook under `notebooks/` to follow the step-by-step workflow.

4) Train the model (script)
```bash
python training_and_evaluation.py
```
- This script runs preprocessing, model training, evaluation, and saves:
  - Trained model: `models/breast_cancer_nn.h5`
  - Plots to `results/` (accuracy, loss, confusion matrix)

(If your script supports flags, you can often pass `--epochs`, `--batch-size`, `--seed`, etc. Check the top of `training_and_evaluation.py` for available CLI options.)

5) Run inference (example)
- Quick example using the inference script. The script expects a preprocessed / normalized 30-feature vector or a CSV row matching `data.csv`'s feature order.

Example: run inference with a single sample (pseudo-command; adapt to script signature):
```bash
python inference.py --input "12.45,14.23,78.9,..."  # 30 comma-separated feature values
```

Example Python usage (programmatic)
```python
from inference import predict_from_list
# sample_features: list or numpy array of length 30 (preprocessed the same way as training)
sample_features = [14.2, 20.3, 95.0, 700.0, 0.1, ...]  # 30 values
probability = predict_from_list(sample_features)
print(f"Malignant probability: {probability:.4f}")
```

Notes
- Ensure the same preprocessing and scaler used during training are applied to inference inputs (the notebook or training script demonstrates and saves the scaler).
- Model file is located at `models/breast_cancer_nn.h5`.

---

📈 Training Visualizations

Model accuracy over training epochs:
![Model Accuracy Plot](results/accuracy_plot.png)

Caption: Model accuracy curve showing training and validation accuracy across epochs — used to monitor convergence and potential overfitting.

Confusion matrix on test set:
![Confusion Matrix](results/confusion_matrix.png)

Caption: Confusion matrix visualizing True/False Positives/Negatives. See Evaluation section for full interpretation.

(Additional plots, such as loss plot, ROC curve, and class probability distribution, are available in the `results/` folder.)

---

🧾 Evaluation

Overall Test Performance
- Test Accuracy: **95.61%**

Per-class classification metrics:
| Class     | Precision | Recall  | F1-Score |
|-----------|-----------:|--------:|---------:|
| Benign    | 95.71%    | 97.10% | 96.40%  |
| Malignant | 95.45%    | 93.33% | 94.38%  |

Confusion Matrix — Summary
- True Positives (TP): Correctly predicted malignant cases.
- True Negatives (TN): Correctly predicted benign cases.
- False Positives (FP): Benign cases incorrectly predicted as malignant (type I error).
- False Negatives (FN): Malignant cases incorrectly predicted as benign (type II error).

Insight from confusion matrix
- The model shows strong discrimination between benign and malignant samples, with relatively balanced precision and recall across classes.
- Recall for the benign class is slightly higher than malignant recall, meaning the model is slightly more likely to catch benign samples correctly. Conversely, malignant recall (~93.33%) indicates a small portion of malignant cases could be missed — an area to monitor carefully in clinical applications where false negatives are costly.
- Overall F1-scores are high (96.40% benign, 94.38% malignant), indicating a good balance between precision and recall.

Short performance insight
- With 95.61% test accuracy and high per-class F1-scores, this ANN is a robust baseline for this dataset. It is suitable for prototyping and research; however, for deployment in clinical scenarios, more validation, calibration, and explainability are required.

---

📋 Reproducibility Checklist
- [ ] Use the provided `data.csv` (or an identical preprocessing pipeline) to reproduce results.
- [ ] Ensure deterministic runs by setting random seeds (see `training_and_evaluation.py`).
- [ ] Save and reuse the same scaler for inference as used during training.

---

🔧 Future Improvements
- Increase model interpretability with SHAP or LIME explanations.
- Implement cross-validation to better estimate generalization performance.
- Add class calibration (Platt scaling / isotonic regression) to improve probability estimates.
- Experiment with larger / smaller architectures and regularization (dropout, weight decay).
- Convert script to a small REST API for production inference (FastAPI / Flask).
- Add unit tests and CI to validate data schemas, model predictions, and script interfaces.

---

📁 License
This project is provided under the MIT License — see LICENSE file for details. (Add a LICENSE file in your repo root if you haven't already.)

---

🙏 Credits
- Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset — widely used academic dataset for classification benchmarking.
- Author / Maintainer: Annamalai-KM
- Thanks to the open-source community for libraries and tooling used (NumPy, pandas, scikit-learn, TensorFlow / Keras, Matplotlib / Seaborn).

---

Contact
- GitHub: https://github.com/Annamalai-KM
- If you're a recruiter or engineer reviewing this project and would like a walkthrough, feel free to open an issue or reach out via GitHub. I’m happy to provide a demo and discuss production considerations.

---

Thank you for reviewing this project! 🚀
