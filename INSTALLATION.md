# Installation Guide

This project does not automatically install requirements just because students receive the same folder.
Each student needs to install Python, install the Python packages from `requirements.txt`, and run Ollama locally for the best prompt results.

## Before You Start

- Install Python 3.10 or newer
- Make sure Python is added to your terminal path
- Install Ollama if you want the best local model results
- Keep the project files in one folder

## Recommended Runtime Order

The project now tries providers in this order:

1. Local Ollama
2. Local Hugging Face fallback

For the best classroom demo results, use Ollama with:

- `llama3.2:latest` for prompt demos
- `nomic-embed-text:latest` for embeddings used by ChromaDB

## macOS Quick Setup

Students can run these commands in order:

```bash
cd "/path/to/class 1"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
python3 main.py
```

These commands do all required setup for this project:

- move into the project folder
- create a virtual environment
- activate it
- upgrade `pip`
- install everything from `requirements.txt`
- download the local Ollama models used by the demo
- start the classroom app

## macOS Step-By-Step

### 1. Open Terminal

You can use:

- Terminal
- iTerm
- The built-in terminal inside PyCharm or VS Code

### 2. Go to the project folder

Example:

```bash
cd "/path/to/class 1"
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
```

### 4. Activate the virtual environment

```bash
source .venv/bin/activate
```

After activation, your terminal usually shows `(.venv)` at the beginning of the line.

### 5. Install the project requirements

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 6. Install Ollama models

If Ollama is installed, download the classroom models:

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

### 7. Run the project

```bash
python3 main.py
```

If `python3` does not use the virtual environment interpreter on your machine, use:

```bash
.venv/bin/python main.py
```

## Windows Quick Setup

Students can run these commands in order in Command Prompt:

```cmd
cd "C:\path\to\class 1"
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
python main.py
```

If `py` is not available, replace the second command with:

```cmd
python -m venv .venv
```

These commands do all required setup for this project:

- move into the project folder
- create a virtual environment
- activate it
- upgrade `pip`
- install everything from `requirements.txt`
- download the local Ollama models used by the demo
- start the classroom app

## Windows Step-By-Step

### 1. Open Command Prompt or PowerShell

You can use:

- Command Prompt
- PowerShell
- The built-in terminal inside PyCharm or VS Code

### 2. Go to the project folder

Example:

```powershell
cd "C:\path\to\class 1"
```

### 3. Create a virtual environment

```powershell
py -m venv .venv
```

If `py` does not work, try:

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

For Command Prompt:

```cmd
.venv\Scripts\activate
```

For PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the project requirements

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6. Install Ollama models

If Ollama is installed, download the classroom models:

```powershell
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

### 7. Run the project

```powershell
python main.py
```

## Common Commands

Install requirements again later:

```bash
python -m pip install -r requirements.txt
```

Run prompt demo only:

```bash
python run_prompt_demo.py
```

Run RAG demo only:

```bash
python run_rag_demo.py
```

Run both demos:

```bash
python run_both_demos.py
```

Force Ollama provider:

```bash
LLM_PROVIDER=ollama python run_prompt_demo.py
```

## Troubleshooting

### `pip` not found

Try:

```bash
python -m pip install -r requirements.txt
```

On macOS, you can also try:

```bash
python3 -m pip install -r requirements.txt
```

### PowerShell blocks activation

Run this in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

### Ollama command not found

Install Ollama first, then restart the terminal and run:

```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```

### First model download takes time

The first Ollama pull can take several minutes because the model files must download to the student's machine.

### Ollama is not running

If the project does not use Ollama automatically, start the Ollama app or server and run the demo again.

If Ollama is not available, the project falls back to the local Hugging Face model.

### ChromaDB creates local files

The RAG demo stores vector database files in the `chroma_storage/` folder.
That is expected.
