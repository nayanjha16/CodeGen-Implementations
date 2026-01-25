---
title: NL-to-NoSQL Converter
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🚀 NL-to-NoSQL Conversion System

Transform natural language questions into SQL and MongoDB queries using fine-tuned Qwen2.5-0.5B models.

## ✨ Features

- **Complete Pipeline**: Natural Language → SQL → MongoDB
- **High Accuracy**: 100% Text-to-SQL, 95.65% SQL-to-MongoDB
- **4 Operational Modes**:
  1. Complete Pipeline (NL → SQL → MongoDB)
  2. Text-to-SQL Only
  3. SQL-to-MongoDB Only
  4. Schema Translation with RAG
- **Security**: SQL injection protection, dangerous operation filtering
- **RAG-Enhanced**: Retrieval-augmented schema translation

## 🎯 Performance

- **Text-to-SQL Accuracy**: 100% exact match
- **SQL-to-MongoDB**: 95.65% semantic similarity
- **Query Time**: <2 seconds
- **Memory**: ~1.2 GB

## 🔧 Technology

- **Model**: Qwen2.5-0.5B-Instruct
- **Fine-tuning**: QLoRA (1.6% trainable parameters)
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **RAG**: Sentence-BERT embeddings

## 📚 Training Data

- **Text-to-SQL**: 2,500 examples (Spider dataset)
- **SQL-to-MongoDB**: 500 examples (BIRD dataset)

## 🎓 Citation

If you use this system in your research, please cite:

```
@software{nl_to_nosql_2025,
  author = {Bhanu Kumar},
  title = {NL-to-NoSQL Conversion System},
  year = {2025},
  url = {https://huggingface.co/spaces/YOUR_USERNAME/nl-to-nosql-converter}
}
```

## 📖 Documentation

For detailed documentation, training notebooks, and deployment guides, visit the [GitHub repository](https://github.com/YOUR_USERNAME/nl-to-nosql).

## 🚀 Local Development

```bash
# Build Docker image
docker build -t nl-to-nosql .

# Run container
docker run -p 7860:7860 nl-to-nosql
```

## 📝 License

MIT License - see LICENSE file for details
