# 📘 Dataset Availability 
The data is available via zenodo https://doi.org/10.5281/zenodo.17440661


# 📘 Baseline Classifiers for MCQ Difficulty Estimation

This repository provides **baseline implementations** for predicting the **difficulty of multiple-choice questions (MCQs)** using **textual and multimodal (image + text)** inputs.  
The baselines include:

1. 🧠 **Text-Only Baseline** — BERT, mBERT, RoBERTa, DeBERTa  
2. 🖼️ **BLIP Baseline** — Vision + Language fusion using BLIP  
3. 🧩 **ViLT Baseline** — Vision-Language Transformer (ViLT)  
4. 🎨 **CLIP Baseline** — Vision-Language alignment using CLIP  

Each baseline performs **3-fold cross-validation** and saves fold-wise classification reports.

---

## 📂 Repository Structure

```
.
├── baseline_text_classifier.py     # Text-only BERT-family baselines
├── baseline_blip_classifier.py     # BLIP-based multimodal baseline
├── baseline_vlim_classifier.py     # ViLT-based multimodal baseline
├── baseline_mm_classifier.py       # CLIP-based multimodal baseline
├── data.csv                        # CSV file with MCQs and metadata (expected)
├── images/                         # Folder containing associated images
└── README.md                       # This documentation
```

---

## 🧩 1. Dataset Format

All models expect a CSV file named **`data.csv`** with the following structure:

| Column | Description |
|:--------|:-------------|
| `filename` | Image filename (for multimodal models) |
| `MCQ` | The question text |
| `Option A`–`Option D` | Four answer choices |
| `Difficulty` | Difficulty label: `Easy`, `Medium`, or `Hard` |
| *(optional)* `description` | Textual caption or contextual information for image (used if `use_description=True`) |

### Example Row
| filename | MCQ | Option A | Option B | Option C | Option D | Difficulty |
|:--|:--|:--|:--|:--|:--|:--|
| img_0001.png | What is the process of photosynthesis? | Energy release | Energy absorption | Glucose synthesis | Protein breakdown | Medium |

---

## ⚙️ 2. Environment Setup

All scripts are written in **Python ≥ 3.8** and rely on **PyTorch + Transformers**.

### 🧱 Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers==4.40.0 simpletransformers scikit-learn pillow pandas
```

For GPU acceleration, ensure that CUDA drivers are properly installed.

---

## 🧠 3. Text-Only Baseline — `baseline_text_classifier.py`

This baseline evaluates **purely textual MCQs** using transformer encoders (BERT, mBERT, RoBERTa, DeBERTa).  
It uses **SimpleTransformers** for convenience and performs **3-fold cross-validation**.

### ✅ Features
- Converts each MCQ + options into a single text input  
- Encodes difficulty labels numerically (`Easy = 0`, `Medium = 1`, `Hard = 2`)  
- Evaluates four architectures automatically  
- Saves reports to `cv_classification_reports.txt`

### 🏃‍♂️ Run
```bash
python baseline_text_classifier.py
```

### 📊 Output
Each fold prints and saves a classification report:
```
===== BERT: 3-Fold CV =====
--- Fold 1 ---
precision  recall  f1-score  support
Easy   0.72   0.74   0.73   1422
...
✅ 3-Fold CV results saved to cv_classification_reports.txt
```

---

## 🖼️ 4. BLIP Multimodal Baseline — `baseline_blip_classifier.py`

This baseline uses the **BLIP model (`Salesforce/blip-image-captioning-base`)** to jointly process **images and text**.

### ✅ Features
- Uses `BlipProcessor` to align text and image features  
- Custom `BLIPMCQDataset` for multimodal data loading  
- Linear fusion layer combines vision + text pooled embeddings  
- 3-fold cross-validation for reproducibility  
- Saves reports to `cv_blip_classification_reports.txt`

### 🏃‍♂️ Run
```bash
python baseline_blip_classifier.py
```

### 📊 Output
```
[Fold 1] Epoch 1 Train Loss: 0.9123
...
✅ 3-Fold CV results saved to cv_blip_classification_reports.txt
```

---

## 🧩 5. ViLT Multimodal Baseline — `baseline_vlim_classifier.py`

The **ViLT baseline** leverages the **`dandelin/vilt-b32-finetuned-vqa`** model, directly fusing image and text without CNN backbones.

### ✅ Features
- Uses `ViltProcessor` and `ViltModel`  
- Resizes all images to `224×224`  
- Learns a pooled multimodal representation  
- Adds two-layer feed-forward classifier  
- Saves reports to `cv_vilt_classification_reports.txt`

### 🏃‍♂️ Run
```bash
python baseline_vlim_classifier.py
```

### 📊 Output
```
[Fold 2] Epoch 5 Train Loss: 0.7211
✅ 3-Fold CV results saved to cv_vilt_classification_reports.txt
```

---

## 🎨 6. CLIP Multimodal Baseline — `baseline_mm_classifier.py`

This baseline uses **OpenAI’s CLIP (`openai/clip-vit-base-patch32`)**, aligning text and image embeddings into a shared space.

### ✅ Features
- Uses `CLIPProcessor` for joint encoding  
- Concatenates image + text embeddings (each 512-D)  
- Passes through an MLP classifier (`Linear–ReLU–Dropout–Linear`)  
- 3-fold CV with `StratifiedKFold`  
- Saves reports to `cv_clip_classification_reports.txt`

### 🏃‍♂️ Run
```bash
python baseline_mm_classifier.py
```

### 📊 Output
```
[Fold 3] Epoch 10 Train Loss: 0.6489
✅ 3-Fold CV results saved to cv_clip_classification_reports.txt
```

---

## 🧪 7. Experimental Notes

| Aspect | Description |
|:--|:--|
| **Reproducibility** | All scripts set a fixed random seed (`42`) across NumPy, PyTorch CPU + GPU for deterministic results. |
| **Cross-Validation** | Each model performs **3-fold stratified CV** based on difficulty labels. |
| **Evaluation Metric** | `classification_report` (Precision, Recall, F1-score per class). |
| **Device Handling** | Automatically detects GPU availability and uses CUDA if possible. |
| **Batch Sizes** | Default: 16 (Text, BLIP), 64 (ViLT & CLIP). |
| **Learning Rate** | Default: 2e-5 for all models. |

---

## 📜 8. Expected Outputs

Each baseline produces a text file:
```
cv_classification_reports.txt
cv_blip_classification_reports.txt
cv_vilt_classification_reports.txt
cv_clip_classification_reports.txt
```

Each file contains fold-wise results for later aggregation into radar or bar plots.

---

## 🧾 9. Citation

If you use these baselines in your work, please cite appropriately:

```bibtex
@software{MCQBaseline2025,
  author = {Manikandan, R.},
  title  = {Baseline Classifiers for Multimodal MCQ Difficulty Estimation},
  year   = {2025},
  url    = {https://github.com/<your_repo_link>}
}
```

---

## 🧠 10. Suggested Extensions

- Add **feature-fusion variants** combining CLIP + BLIP outputs.  
- Integrate **difficulty regression** instead of classification.  
- Explore **attention-based fusion layers** for interpretability.  
- Visualize **saliency maps** to inspect modality contributions.

---

### 🏁 Summary

| Model | Modality | Pretrained Backbone | Report |
|:--|:--|:--|:--|
| `baseline_text_classifier.py` | Text | BERT / mBERT / RoBERTa / DeBERTa | `cv_classification_reports.txt` |
| `baseline_blip_classifier.py` | Image + Text | BLIP | `cv_blip_classification_reports.txt` |
| `baseline_vlim_classifier.py` | Image + Text | ViLT | `cv_vilt_classification_reports.txt` |
| `baseline_mm_classifier.py` | Image + Text | CLIP | `cv_clip_classification_reports.txt` |
