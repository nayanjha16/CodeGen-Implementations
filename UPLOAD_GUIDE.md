# 📤 How to Use Your Fine-tuned Model in Colab

## Quick Summary

Your fine-tuned model `phi3-finetuned` is running locally on your Mac. To use it in Google Colab, you need to upload 2 files:

1. ✅ `phi3-finetuned-q4.gguf` (2GB) - the model weights
2. ✅ `Modelfile` - the model configuration

Both files are already in your repo!

---

## Step-by-Step Upload Process

### 1. Open the Colab Notebook

Open `deploy_colab.ipynb` in Google Colab

### 2. Locate the Files

The files are in your repo:
```
CodeGen-Implementations/
├── phi3-finetuned-q4.gguf  ← Upload this (2GB)
└── Modelfile                ← Upload this (small)
```

### 3. Upload to Colab

**In Colab:**
1. Click the **📁 Files** icon on the left sidebar
2. Click **📤 Upload to session storage** button
3. Select both files from your computer
4. Wait ~3-5 minutes for upload to complete

**You'll see:**
```
/content/
├── phi3-finetuned-q4.gguf  ✅
└── Modelfile                ✅
```

### 4. Run the Notebook

After uploading, run all cells. The notebook will automatically:
- Detect your uploaded files
- Create the model using: `ollama create phi3-finetuned -f Modelfile`
- Use `phi3-finetuned` for SQL generation ✅

---

## Alternative: Skip Upload (Use Base Model)

If upload is too slow or fails, you can skip it:

**Result:**
- Notebook will use base `phi3` model instead
- Faster setup (~2 min vs ~8 min)
- Slightly less accurate SQL generation
- Still works fine for demo!

The notebook handles both cases automatically 🎯

---

## File Size Concern?

**Q: 2GB upload is slow in Colab?**

**Solutions:**

### Option 1: Use Colab's wget (Faster!)
If you can host the file temporarily:

1. Upload `phi3-finetuned-q4.gguf` to Google Drive
2. Get shareable link
3. Use this in Colab instead of manual upload:

```python
# In Colab cell:
!gdown --id YOUR_GOOGLE_DRIVE_FILE_ID
```

### Option 2: GitHub LFS (Best for sharing)
Push the model to GitHub using Git LFS:

```bash
# On your Mac:
cd /Users/utkarshsinha/Desktop/study/PG/capstone/CodeGen-Implementations

# Install Git LFS
brew install git-lfs
git lfs install

# Track the .gguf file
git lfs track "*.gguf"
git add .gitattributes
git add phi3-finetuned-q4.gguf
git commit -m "Add fine-tuned model via LFS"
git push
```

Then in Colab:
```bash
# Colab will auto-download via LFS when cloning
# No manual upload needed!
```

### Option 3: Use Base Model
Just skip the upload - base `phi3` works fine for demos!

---

## Verification

After upload, verify files are present:

**In Colab:**
```python
!ls -lh *.gguf
!cat Modelfile
```

**Expected output:**
```
-rw-r--r-- 1 root root 2.0G phi3-finetuned-q4.gguf
FROM ./phi3-finetuned-q4.gguf
TEMPLATE """<|user|>
...
```

---

## Troubleshooting

### Upload Failed / Timeout
**Solution:** Use Google Drive method (Option 1 above)

### "Model not found" after upload
**Check:**
```python
!ollama list
```
Should show `phi3-finetuned`

**Fix:**
```bash
!ollama create phi3-finetuned -f Modelfile
```

### Out of Memory in Colab
**Solution:** 
- Use GPU runtime (Runtime → Change runtime type → T4 GPU)
- OR use base `phi3` instead (smaller)

---

## Current Status

✅ **Ready to use:**
- `phi3-finetuned-q4.gguf` exists locally (2GB)
- `Modelfile` exists and is correct
- Both are gitignored (won't auto-push to GitHub)

**Recommendation for demo:**
1. Try base `phi3` first (skip upload) - fast setup
2. If professor wants better accuracy, re-run with fine-tuned model

---

## Quick Decision Matrix

| Scenario | Upload Files? | Model Used | Setup Time |
|----------|---------------|------------|------------|
| **Quick demo** | ❌ No | phi3 (base) | ~5 min |
| **Best accuracy** | ✅ Yes | phi3-finetuned | ~10 min |
| **Final presentation** | ✅ Yes | phi3-finetuned | ~10 min |

Choose based on your time/quality tradeoff! Both will work. 🎯
