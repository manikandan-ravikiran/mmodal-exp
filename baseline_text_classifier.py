import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch, os
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)                     # Python random
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch (CPU)
    torch.cuda.manual_seed(seed)          # PyTorch (current GPU)
    torch.cuda.manual_seed_all(seed)      # All GPUs

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================
# 1. Load Data
# ============================
df = pd.read_csv("data.csv")  # your Excel file

def make_input(row):
    return f"Q: {row['MCQ']} A. {row['Option A']} B. {row['Option B']} C. {row['Option C']} D. {row['Option D']}"

df["text"] = df.apply(make_input, axis=1)

# Encode labels
label_encoder = LabelEncoder()
df["labels"] = label_encoder.fit_transform(df["Difficulty"])  # Easy=0, Medium=1, Hard=2

# ============================
# 2. Define Models
# ============================
models = {
    "BERT": "bert-base-uncased",
    "mBERT": "bert-base-multilingual-cased",
    "RoBERTa": "roberta-base",
    "DeBERTa": "microsoft/deberta-base"
}

# ============================
# 3. Cross-validation
# ============================
results = {}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for model_name, model_ckpt in models.items():
    print(f"\n===== {model_name}: 3-Fold CV =====")
    fold_reports = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(df["text"], df["labels"]), 1):
        print(f"\n--- Fold {fold} ---")

        train_data = df.iloc[train_idx][["text", "labels"]]
        eval_data  = df.iloc[test_idx][["text", "labels"]]

        model_args = ClassificationArgs(
            num_train_epochs=5,
            overwrite_output_dir=True,
            save_model_every_epoch=False,
            save_eval_checkpoints=False,
            output_dir=f"./outputs/{model_name}_fold{fold}/",
            best_model_dir=f"./outputs/{model_name}_fold{fold}/best_model/",
            train_batch_size=16,
            eval_batch_size=16,
            fp16=False,
            learning_rate=2e-5,
            logging_steps=50,
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
            process_count=1,
        )

        # Detect model type
        if "roberta" in model_ckpt.lower():
            model_type = "roberta"
        elif "deberta" in model_ckpt.lower():
            model_type = "deberta"
        else:
            model_type = "bert"

        model = ClassificationModel(
            model_type=model_type,
            model_name=model_ckpt,
            num_labels=len(label_encoder.classes_),
            args=model_args,
            use_cuda=torch.cuda.is_available()
        )

        # Train
        model.train_model(train_data, eval_df=eval_data)

        # Predict
        preds, _ = model.predict(eval_data["text"].tolist())
        report = classification_report(eval_data["labels"], preds,
                                       target_names=label_encoder.classes_,
                                       digits=3, output_dict=False)
        print(report)
        fold_reports.append(report)

    # Save all fold reports for this model
    results[model_name] = fold_reports

# ============================
# 4. Save Results
# ============================
with open("cv_classification_reports.txt", "w") as f:
    for model_name, reports in results.items():
        f.write(f"\n\n===== {model_name} (3-Fold CV) =====\n")
        for i, rep in enumerate(reports, 1):
            f.write(f"\n--- Fold {i} ---\n{rep}\n")

print("\n✅ 3-Fold CV results saved to cv_classification_reports.txt")
