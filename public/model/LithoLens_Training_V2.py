# =============================================================
# 🪨 LithoLens V2 — Training Script for Google Colab
# =============================================================
# HOW TO USE:
# 1. Open Google Colab (colab.research.google.com)
# 2. Go to Runtime → Change runtime type → Select T4 GPU
# 3. Copy each CELL section into a separate Colab cell
# 4. Run cells one by one in order
# =============================================================


# %% CELL 1 — Install Packages
# !pip install datasets timm onnx onnxruntime torch torchvision Pillow --quiet
# print('✅ All packages installed')


# %% CELL 2 — Load MineralImage5K-98 from HuggingFace
from datasets import load_dataset
import os

print('⬇️  Downloading MineralImage5K-98 from HuggingFace...')
print('   This is ~3.9 GB — takes 5-10 minutes on Colab.')
dataset = load_dataset('Nech-C/mineralimage5K-98')

print(f'✅ Dataset loaded!')
print(f'   Train: {len(dataset["train"])} images')
print(f'   Val:   {len(dataset["validation"])} images')
print(f'   Test:  {len(dataset["test"])} images')

# Get class names from the dataset
class_names = dataset['train'].features['name'].names
print(f'   Classes: {len(class_names)}')
print(f'   Names: {class_names}')


# %% CELL 3 — Clean Class Names + Add not_mineral
import numpy as np
from collections import Counter

# Clean up unusual names
NAME_FIXES = {
    'labrador': 'labradorite',
    'nephritis': 'nephrite',
    'cancrinit': 'cancrinite',
    'scheelit': 'scheelite',
    'analcim': 'analcime',
    'elbait': 'elbaite',
    'credit': 'creedit',  # likely a dataset artifact, will check count
}

cleaned_names = []
for name in class_names:
    cleaned = NAME_FIXES.get(name, name)
    cleaned_names.append(cleaned)

print('Class name fixes applied:')
for old, new in NAME_FIXES.items():
    if old in class_names:
        print(f'  {old} → {new}')

# Count samples per class
train_labels = dataset['train']['name']
label_counts = Counter(train_labels)
print(f'\nSamples per class:')
for idx, name in enumerate(class_names):
    count = label_counts.get(idx, 0)
    bar = '█' * (count // 10)
    print(f'  {cleaned_names[idx]:20s} {count:4d} {bar}')


# %% CELL 4 — Download NOT_MINERAL images from CIFAR-100
import torchvision
import torchvision.transforms as T
from PIL import Image
import random

print('⬇️  Downloading CIFAR-100 for not_mineral class...')
cifar = torchvision.datasets.CIFAR100(
    root='/content/cifar100', train=True, download=True
)

# Pick random non-geological categories
# CIFAR has things like: bicycle, bus, baby, bear, bed, bee, etc.
random.seed(42)
not_mineral_dir = '/content/not_mineral_images'
os.makedirs(not_mineral_dir, exist_ok=True)

# Sample ~800 random images from various CIFAR classes
indices = random.sample(range(len(cifar)), 800)
for i, idx in enumerate(indices):
    img, _ = cifar[idx]
    # Upscale from 32x32 to 224x224
    img = img.resize((224, 224), Image.BICUBIC)
    img.save(f'{not_mineral_dir}/not_mineral_{i:04d}.png')

print(f'✅ Created {len(indices)} not_mineral images')


# %% CELL 5 — Build PyTorch Datasets
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import io
import os

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_MINERAL_CLASSES = len(class_names)  # 98

# Final class list = 98 minerals + not_mineral
FINAL_CLASS_NAMES = cleaned_names + ['not_mineral']
NUM_CLASSES = len(FINAL_CLASS_NAMES)  # 99
NOT_MINERAL_IDX = NUM_CLASSES - 1

print(f'Total classes: {NUM_CLASSES} ({NUM_MINERAL_CLASSES} minerals + not_mineral)')

# Augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MineralDataset(Dataset):
    """Wraps HuggingFace dataset + not_mineral images."""

    def __init__(self, hf_split, not_mineral_dir, transform, is_train=True):
        self.hf_data = hf_split
        self.transform = transform
        self.not_mineral_images = []

        if not_mineral_dir and os.path.exists(not_mineral_dir):
            all_imgs = sorted([
                os.path.join(not_mineral_dir, f)
                for f in os.listdir(not_mineral_dir)
                if f.endswith('.png')
            ])
            if is_train:
                self.not_mineral_images = all_imgs[:640]  # 80% train
            else:
                self.not_mineral_images = all_imgs[640:]  # 20% val

        self.total_len = len(self.hf_data) + len(self.not_mineral_images)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.hf_data):
            item = self.hf_data[idx]
            image = item['image'].convert('RGB')
            label = item['name']  # integer class index
        else:
            nm_idx = idx - len(self.hf_data)
            image = Image.open(self.not_mineral_images[nm_idx]).convert('RGB')
            label = NOT_MINERAL_IDX

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = MineralDataset(
    dataset['train'], not_mineral_dir, train_transform, is_train=True
)
val_dataset = MineralDataset(
    dataset['validation'], not_mineral_dir, val_transform, is_train=False
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

print(f'✅ Train: {len(train_dataset)} samples')
print(f'✅ Val:   {len(val_dataset)} samples')
print(f'✅ Batches per epoch: {len(train_loader)}')


# %% CELL 6 — Build Model + Class Weights
import timm
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# EfficientNet-B0 with pretrained weights
model = timm.create_model(
    'efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES
)
model = model.to(device)

# Class weights — handle imbalanced classes
label_counts_list = [0] * NUM_CLASSES
for idx in range(NUM_MINERAL_CLASSES):
    label_counts_list[idx] = label_counts.get(idx, 1)
label_counts_list[NOT_MINERAL_IDX] = len(train_dataset.not_mineral_images)

weights = 1.0 / (torch.tensor(label_counts_list, dtype=torch.float32) + 1)
weights = weights / weights.sum() * NUM_CLASSES  # normalize
class_weights = weights.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'✅ Model loaded — {total_params:,} parameters')
print(f'✅ Class weights computed for {NUM_CLASSES} classes')


# %% CELL 7 — Train
import torch.optim as optim
import time

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

EPOCHS = 20
best_val_acc = 0.0
best_model_path = '/content/best_model.pth'
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print('🚀 Starting training...')
print('=' * 65)

for epoch in range(EPOCHS):
    start = time.time()

    # Train
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        _, pred = outputs.max(1)
        t_total += labels.size(0)
        t_correct += pred.eq(labels).sum().item()

    # Validate
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, pred = outputs.max(1)
            v_total += labels.size(0)
            v_correct += pred.eq(labels).sum().item()

    scheduler.step()

    t_acc = 100. * t_correct / t_total
    v_acc = 100. * v_correct / v_total
    elapsed = time.time() - start

    history['train_loss'].append(t_loss / len(train_loader))
    history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss / len(val_loader))
    history['val_acc'].append(v_acc)

    saved = ''
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), best_model_path)
        saved = '💾 SAVED'

    print(f'Epoch {epoch+1:2d}/{EPOCHS} | '
          f'Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | '
          f'Time: {elapsed:.0f}s {saved}')

print('=' * 65)
print(f'✅ Done! Best val accuracy: {best_val_acc:.1f}%')


# %% CELL 8 — Plot Results
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['train_acc'], label='Train', color='#2196F3')
ax1.plot(history['val_acc'], label='Val', color='#4CAF50')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('%')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history['train_loss'], label='Train', color='#F44336')
ax2.plot(history['val_loss'], label='Val', color='#FF9800')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'LithoLens V2 — {NUM_CLASSES} classes — Best: {best_val_acc:.1f}%')
plt.tight_layout()
plt.savefig('/content/training_results.png', dpi=150, bbox_inches='tight')
plt.show()


# %% CELL 9 — Export to ONNX
import json

model.load_state_dict(torch.load(best_model_path))
model.eval()
model.cpu()

dummy = torch.randn(1, 3, 224, 224)
onnx_path = '/content/litholens_model.onnx'

torch.onnx.export(
    model, dummy, onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Save class names
class_names_data = {
    'classes': FINAL_CLASS_NAMES,
    'num_classes': NUM_CLASSES,
    'not_mineral_index': NOT_MINERAL_IDX
}
with open('/content/class_names.json', 'w') as f:
    json.dump(class_names_data, f, indent=2)

model_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f'✅ ONNX exported: {model_size:.1f} MB')
print(f'✅ Classes: {NUM_CLASSES} ({NUM_MINERAL_CLASSES} minerals + not_mineral)')
print(f'✅ Input name: "input", Output name: "output"')


# %% CELL 10 — Verify ONNX
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('/content/litholens_model.onnx')
test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {'input': test_input})
preds = outputs[0][0]

softmax = np.exp(preds - preds.max()) / np.sum(np.exp(preds - preds.max()))
top5 = np.argsort(softmax)[::-1][:5]

print('✅ ONNX works! Top 5 on random input:')
for i, idx in enumerate(top5):
    print(f'  {i+1}. {FINAL_CLASS_NAMES[idx]}: {softmax[idx]*100:.1f}%')


# %% CELL 11 — Generate CSV Template for Geology Team
import csv

csv_path = '/content/minerals_template.csv'
columns = [
    'label', 'name_english', 'name_arabic', 'category',
    'hardness_moh', 'hardness_testable', 'luster', 'streak_color',
    'cleavage', 'fracture', 'acid_reaction', 'special_property',
    'description_for_ai', 'common_locations', 'primary_color'
]

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(columns)

    for name in FINAL_CLASS_NAMES:
        if name == 'not_mineral':
            continue  # skip not_mineral — not a real mineral

        # Capitalize the english name nicely
        english_name = name.replace('_', ' ').title()
        # Special cases
        if name == 'lapis_lazuli':
            english_name = 'Lapis Lazuli'

        row = [name, english_name] + [''] * (len(columns) - 2)
        writer.writerow(row)

print(f'✅ CSV template saved: {csv_path}')
print(f'   {NUM_MINERAL_CLASSES} minerals (not_mineral excluded)')
print(f'   Columns: {columns}')
print(f'')
print(f'📋 Give this file to your geology teammates!')
print(f'   They fill in: name_arabic, category, hardness, luster, etc.')
print(f'   Then you upload the filled CSV back here.')


# %% CELL 12 — Convert Filled CSV to minerals_db.json (run AFTER team fills CSV)
# ⚠️  ONLY run this cell AFTER your team fills the CSV!
# Upload the filled CSV first, then run this cell.
import csv
import json

# Uncomment these 2 lines when ready:
# from google.colab import files
# uploaded = files.upload()  # Upload the filled CSV

csv_file = '/content/minerals_template.csv'  # Change to uploaded filename
json_output = '/content/minerals_db.json'

minerals_db = {}
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row['label']
        minerals_db[label] = {
            'name_english': row.get('name_english', ''),
            'name_arabic': row.get('name_arabic', ''),
            'category': row.get('category', ''),
            'hardness_moh': row.get('hardness_moh', ''),
            'hardness_testable': row.get('hardness_testable', ''),
            'luster': row.get('luster', ''),
            'streak_color': row.get('streak_color', ''),
            'cleavage': row.get('cleavage', ''),
            'fracture': row.get('fracture', ''),
            'acid_reaction': row.get('acid_reaction', ''),
            'special_property': row.get('special_property', ''),
            'description_for_ai': row.get('description_for_ai', ''),
            'common_locations': row.get('common_locations', ''),
            'primary_color': row.get('primary_color', ''),
        }

with open(json_output, 'w', encoding='utf-8') as f:
    json.dump(minerals_db, f, indent=2, ensure_ascii=False)

print(f'✅ minerals_db.json created with {len(minerals_db)} minerals')
print(f'   Place this file in: /public/model/minerals_db.json')


# %% CELL 13 — Download Everything
from google.colab import files

print('📥 Downloading files...')
print('You need these files for the React app:')
print('  1. litholens_model.onnx  — the AI model')
print('  2. class_names.json      — class labels + not_mineral index')
print('  3. minerals_template.csv — give to geology team')
print('  4. training_results.png  — for pitch deck')
print('')

files.download('/content/litholens_model.onnx')
files.download('/content/class_names.json')
files.download('/content/minerals_template.csv')
files.download('/content/training_results.png')

print('')
print('✅ Place model files in your React app:')
print('  public/model/litholens_model.onnx')
print('  public/model/class_names.json')
print('  public/model/minerals_db.json  (after team fills CSV)')
