import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics import classification_report
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
# ============================
# 1. Dataset
# ============================
class CLIPMCQDataset(Dataset):
    def __init__(self, df, processor, img_dir, use_description=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.img_dir = img_dir
        self.use_description = use_description

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build text
        if self.use_description and "description" in row:
            text = f"Desc: {row['description']} Q: {row['MCQ']} A. {row['Option A']} B. {row['Option B']} C. {row['Option C']} D. {row['Option D']}"
        else:
            text = f"Q: {row['MCQ']} A. {row['Option A']} B. {row['Option B']} C. {row['Option C']} D. {row['Option D']}"

        # Load image
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        # CLIP processor with fixed padding
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77  # CLIP default
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(row["labels"], dtype=torch.long)
        }

# ============================
# 2. Model
# ============================
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_labels=3, hidden_dim=256):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        hidden_size = self.clip.config.projection_dim  # 512

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        text_feat = outputs.text_embeds
        img_feat = outputs.image_embeds
        combined = torch.cat((text_feat, img_feat), dim=1)
        logits = self.fc(combined)
        return logits

# ============================
# 3. Training & Evaluation
# ============================
def train_and_eval(train_df, val_df, processor, device, num_labels, fold, batch_size=64, epochs=10, lr=2e-5):
    # Dataset & loaders
    train_dataset = CLIPMCQDataset(train_df, processor, img_dir="images/", use_description=False)
    val_dataset = CLIPMCQDataset(val_df, processor, img_dir="images/", use_description=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = CLIPClassifier(num_labels=num_labels)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device)
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Fold {fold}] Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device)
            )
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            golds.extend(batch["label"].cpu().tolist())

    report = classification_report(golds, preds, digits=3)
    print(f"\n[Fold {fold}] Validation Report:\n", report)
    return report

# ============================
# 4. Main with 3-Fold CV
# ============================
if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    # Encode difficulty
    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["Difficulty"])  # Easy=0, Medium=1, Hard=2

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["labels"]), 1):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        report = train_and_eval(train_df, val_df, processor, device, num_labels=len(label_encoder.classes_), fold=fold)
        results.append(report)

    # Save all reports
    with open("cv_clip_classification_reports.txt", "w") as f:
        for i, rep in enumerate(results, 1):
            f.write(f"\n===== Fold {i} =====\n")
            f.write(rep)

    print("\n✅ 3-Fold CV results saved to cv_clip_classification_reports.txt")