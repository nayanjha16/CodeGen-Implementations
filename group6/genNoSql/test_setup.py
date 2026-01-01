#!/usr/bin/env python3
"""
Simple test to verify Ollama and MongoDB are working
"""
import sys

print("=" * 70)
print(" Testing Ollama + MongoDB Setup")
print("=" * 70)
print()

# Test 1: Check Ollama
print("1. Testing Ollama...")
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("   ✅ Ollama is running")
        models = response.json().get("models", [])
        for model in models:
            if "llama3" in model["name"]:
                print(f"   ✅ Found: {model['name']}")
    else:
        print(f"   ❌ Ollama returned status {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Please start Ollama: ollama serve")

print()

# Test 2: Check MongoDB Docker
print("2. Testing MongoDB Docker container...")
try:
    import subprocess
    result = subprocess.run(
        ['docker', 'ps', '--filter', 'name=text2nosql-mongo', '--format', '{{.Status}}'],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        print(f"   ✅ MongoDB container status: {result.stdout.strip()}")
    else:
        print("   ❌ MongoDB container not running")
        print("   Start with: docker run -d --name text2nosql-mongo -p 27017:27017 mongo:7.0")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 3: Test MongoDB connection
print("3. Testing MongoDB connection...")
try:
    import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    client.admin.command('ping')
    print("   ✅ MongoDB connection successful")
    client.close()
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 4: Test Ollama query
print("4. Testing Ollama query generation...")
try:
    import requests
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Say 'Hello' in one word"}],
            "stream": False
        },
        timeout=30
    )
    if response.status_code == 200:
        result = response.json()
        answer = result["message"]["content"]
        print(f"   ✅ Ollama response: {answer[:50]}")
    else:
        print(f"   ❌ Status code: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()
print("=" * 70)
print(" Setup Test Complete")
print("=" * 70)
