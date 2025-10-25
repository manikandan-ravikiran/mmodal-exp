import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import BlipProcessor, BlipModel
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
class BLIPMCQDataset(Dataset):
    def __init__(self, df, processor, img_dir, use_description=False, max_len=128):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.img_dir = img_dir
        self.use_description = use_description
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build text (MCQ + options, optionally add description)
        if self.use_description and "description" in row:
            text = f"Desc: {row['description']} Q: {row['MCQ']} A. {row['Option A']} B. {row['Option B']} C. {row['Option C']} D. {row['Option D']}"
        else:
            text = f"Q: {row['MCQ']} A. {row['Option A']} B. {row['Option B']} C. {row['Option C']} D. {row['Option D']}"

        # Load image
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        # Processor for multimodal input
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
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
class BLIPClassifier(nn.Module):
    def __init__(self, blip_model_name="Salesforce/blip-image-captioning-base", num_labels=3, hidden_dim=256):
        super().__init__()
        self.blip = BlipModel.from_pretrained(blip_model_name)
        
        # BLIP vision model hidden size
        vision_hidden_size = self.blip.config.vision_config.hidden_size  # 768
        # BLIP text model hidden size
        text_hidden_size = self.blip.config.text_config.hidden_size  # 768

        self.fc = nn.Sequential(
            nn.Linear(vision_hidden_size + text_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Get BLIP outputs
        outputs = self.blip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True
        )

        # Extract pooled features
        # For vision: use pooler_output from vision model
        vision_output = outputs.vision_model_output.pooler_output  # [batch, hidden]
        
        # For text: use pooler_output from text model
        text_output = outputs.text_model_output.pooler_output  # [batch, hidden]

        # Concatenate features
        combined = torch.cat((vision_output, text_output), dim=1)
        logits = self.fc(combined)
        return logits


# ============================
# 3. Training & Evaluation
# ============================
def train_and_eval(train_df, val_df, processor, device, num_labels, fold, batch_size=16, epochs=10, lr=2e-5):
    train_dataset = BLIPMCQDataset(train_df, processor, img_dir="images/", use_description=False)
    val_dataset = BLIPMCQDataset(val_df, processor, img_dir="images/", use_description=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BLIPClassifier(num_labels=num_labels)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
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

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["labels"]), 1):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        report = train_and_eval(train_df, val_df, processor, device, num_labels=len(label_encoder.classes_), fold=fold)
        results.append(report)

    # Save results
    with open("cv_blip_classification_reports.txt", "w") as f:
        for i, rep in enumerate(results, 1):
            f.write(f"\n===== Fold {i} =====\n")
            f.write(rep)

    print("\n✅ 3-Fold CV results saved to cv_blip_classification_reports.txt")