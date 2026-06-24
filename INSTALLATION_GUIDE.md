# 📥 Installation Guide — AI Road Damage Detection

---

## ✅ System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / Ubuntu 20.04 / macOS 11 | Windows 11 / Ubuntu 22.04 |
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 4 GB | 8 GB+ |
| GPU | Not required | NVIDIA CUDA (for YOLO training) |
| Disk | 2 GB | 5 GB+ |

---

## 🪟 Windows Installation

### Step 1 — Install Python

1. Go to https://www.python.org/downloads/
2. Download Python **3.11.x** (Windows installer 64-bit)
3. Run installer → **CHECK** "Add Python to PATH"
4. Click "Install Now"
5. Verify:
```cmd
python --version
pip --version
```

### Step 2 — Download Project

```cmd
# Option A: Download ZIP from GitHub and extract
# Option B: Git clone
git clone <your-repo-url>
cd road_damage_ai
```

### Step 3 — Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command line.

### Step 4 — Install Dependencies

```cmd
pip install -r requirements.txt
```

This installs:
- streamlit (dashboard)
- opencv-python-headless (computer vision)
- Pillow, numpy, pandas (data processing)
- plotly (charts)

### Step 5 — Run

```cmd
streamlit run app.py
```

Browser opens automatically at **http://localhost:8501**

---

## 🐧 Ubuntu / Linux Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & pip
sudo apt install python3.11 python3.11-venv python3-pip -y

# Install system OpenCV dependencies
sudo apt install libgl1-mesa-glx libglib2.0-0 ffmpeg -y

# Clone project
git clone <your-repo-url>
cd road_damage_ai

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 🍎 macOS Installation

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Clone project
git clone <your-repo-url>
cd road_damage_ai

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 🐳 Docker Installation (Any OS)

### Prerequisites
Install Docker Desktop from https://www.docker.com/products/docker-desktop/

```bash
# Build image
docker build -t road-damage-ai .

# Run
docker run -p 8501:8501 road-damage-ai

# Open browser: http://localhost:8501
```

---

## 💻 VS Code Setup

### 1. Install VS Code
Download from https://code.visualstudio.com/

### 2. Install Extensions
Open VS Code → Extensions (Ctrl+Shift+X) → Install:
- **Python** (Microsoft)
- **Pylance** (Microsoft)
- **Python Indent**
- **GitLens**

### 3. Open Project
```
File → Open Folder → select road_damage_ai/
```

### 4. Select Python Interpreter
- Press `Ctrl+Shift+P`
- Type: "Python: Select Interpreter"
- Choose: `./venv/Scripts/python.exe` (Windows) or `./venv/bin/python` (Mac/Linux)

### 5. Create launch config
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit: app.py",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "app.py"],
            "console": "integratedTerminal"
        }
    ]
}
```

Now press **F5** to launch the app.

### 6. Integrated Terminal
Press `Ctrl+\`` (backtick) to open terminal in VS Code, then:
```bash
venv\Scripts\activate   # Windows
streamlit run app.py
```

---

## ❌ Troubleshooting

### "streamlit: command not found"
```bash
# Make sure venv is activated
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Try explicit path
python -m streamlit run app.py
```

### "No module named cv2"
```bash
pip install opencv-python-headless
```

### "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

### "pip install is slow"
```bash
pip install -r requirements.txt -i https://pypi.org/simple/ --no-cache-dir
```

### App shows blank page
- Hard refresh browser: `Ctrl+Shift+R`
- Clear Streamlit cache: `Ctrl+C` → restart

### Video upload fails
- Check file size < 500 MB
- Convert to MP4 format: `ffmpeg -i input.avi output.mp4`

---

## 🔁 Update Project

```bash
git pull origin main
pip install -r requirements.txt --upgrade
streamlit run app.py
```