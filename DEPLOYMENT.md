# 🚀 Deployment Guide - Google Colab + ngrok

## Quick Start (5 minutes)

### Step 1: Open Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `deploy_colab.ipynb` from this repo
4. **OR** Click this direct link: [Open in Colab](https://colab.research.google.com/github/nayanjha16/CodeGen-Implementations/blob/main/deploy_colab.ipynb)

**⚠️ If Repository is Private:**
- Download this entire project folder as ZIP
- Upload the ZIP to Colab (Files tab → Upload button)
- The notebook will automatically detect and extract it

### Step 2: Enable GPU (Optional but Recommended)
1. Click **Runtime → Change runtime type**
2. Select **T4 GPU** (faster model loading)
3. Click **Save**

### Step 3: Run All Cells
1. Click **Runtime → Run all**
2. Wait 5-10 minutes for setup
3. Get your public URL from the output! 🎉

### Step 4: Share URL
Copy the ngrok URL (looks like `https://xxxx-xx-xx-xxx-xxx.ngrok-free.app`) and share with your professor!

---

## What Gets Deployed?

✅ **Full Streamlit Web UI**
- Text-to-SQL query generation
- Multiple database selection
- Interactive schema exploration
- Real-time SQL generation

✅ **Ollama LLM Running on Colab**
- phi3 model for SQL generation
- Running on Colab's GPU/CPU

✅ **Public URL via ngrok**
- Accessible from anywhere
- No authentication needed
- Expires when notebook closes

---

## Important Notes

### ⏰ Session Duration
- **Maximum**: 8 hours (Colab limit)
- **Idle timeout**: 90 minutes without interaction
- **Keep notebook tab open** to maintain session

### 💾 Data Persistence
- ❌ No data is saved after session ends
- ✅ Code pulls fresh from GitHub each time
- ✅ Models re-downloaded each session (cached by Colab)

### 🔒 Security
- Public URL = anyone with link can access
- No authentication required
- Safe for professor demo
- Don't share sensitive data through the app

---

## Troubleshooting

### Problem: "Model not found"
**Solution**: Model still downloading. Wait 2-3 more minutes or use smaller model:
```bash
# In Step 3 cell, change to:
ollama pull phi3:mini
```

### Problem: "Connection refused to Ollama"
**Solution**: Restart Ollama server
1. Runtime → Interrupt execution
2. Re-run Step 2 and Step 8

### Problem: "ngrok tunnel not found"
**Solution**: Get free ngrok account (recommended)
1. Sign up at https://dashboard.ngrok.com/signup
2. Copy your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Paste in Step 6 cell:
```python
ngrok.set_auth_token("YOUR_TOKEN_HERE")
```

### Problem: "Out of memory"
**Solution**: 
1. Runtime → Change runtime type → Select **T4 GPU**
2. OR use smaller model (phi3:mini)

### Problem: Session keeps disconnecting
**Solution**:
- Keep Colab tab active (don't minimize/close)
- Interact with UI every 30 minutes
- Use browser extension like "Auto Refresh" to keep tab active

---

## Advanced: Using Your Fine-tuned Model

If you want to use `phi3-finetuned` instead of `phi3`:

### Option 1: Upload to Hugging Face
1. Convert your fine-tuned model to GGUF format
2. Upload to Hugging Face
3. Pull in Colab:
```bash
ollama pull hf.co/yourusername/phi3-finetuned
```

### Option 2: Upload to Colab Session
1. Upload `phi3-finetuned-q4.gguf` to Colab files (left sidebar)
2. Create model:
```bash
# In Colab cell:
!ollama create phi3-finetuned -f Modelfile
```

---

## Cost Breakdown

| Resource | Cost |
|----------|------|
| Google Colab (free tier) | **$0** |
| ngrok (free tier) | **$0** |
| **Total** | **$0** ✅ |

**Limitations with free tier:**
- ngrok: Random URL each session, some rate limits
- Colab: 8-hour max session, may disconnect if idle

**Upgrade options (optional):**
- ngrok Pro ($8/month): Persistent URL, no rate limits
- Colab Pro ($10/month): Longer sessions, better GPU

---

## Demo Checklist

Before sharing with professor:

- [ ] Test the public URL yourself first
- [ ] Try 2-3 sample queries to verify it works
- [ ] Check database selection dropdown works
- [ ] Note the URL expiry time
- [ ] Have backup plan (local recording/screenshots)
- [ ] Keep Colab tab open during professor's review

---

## Alternative: Quick Local Demo

If Colab has issues, you can demo locally:

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Streamlit
streamlit run ui.py
```

Then use **ngrok locally**:
```bash
# Install ngrok: brew install ngrok
ngrok http 8501
```

This gives you a public URL from your local machine!

---

## Support

If you encounter issues:
1. Check Colab runtime status (top-right corner)
2. View logs in notebook output cells
3. Restart runtime and re-run all cells
4. Contact: Check GitHub issues

---

## Next Steps After Demo

If professor wants permanent deployment:
1. Deploy to Streamlit Cloud (free, persistent)
2. Switch to OpenAI API (more reliable)
3. Use cloud VM with Ollama (AWS/GCP)

See [README.md](README.md) for more deployment options.
