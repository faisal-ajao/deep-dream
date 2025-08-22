# 🌌 DeepDream with TensorFlow & Keras

This project implements **DeepDream** using a pretrained **InceptionV3** model in TensorFlow/Keras.  
DeepDream enhances patterns in images by maximizing neuron activations, resulting in surreal, dream-like visuals.  

---

## 🚀 Features
- Generate DeepDream images from your own photos.  
- Multi-octave processing for richer details.  
- Configurable parameters (layers, iterations, step size, max loss).  
- Pre-included examples (`cat`, `elephant`, `human`, `golden_gate_bridge`).  

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/faisal-ajao/deep-dream.git
cd deep-dream

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main script to generate a DeepDream image:

```bash
python main.py
```

Or experiment interactively with the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

---

## 📊 Output Example (Image)  

Example with **cat.jpg**:  

<p>
  <img src="assets/cat_dream.png" alt="Cat DeepDream Output" width="400"/>
</p>

---

## 📂 Project Structure
```
deep-dream/
├── assets/                    # Generated dream images
│   ├── cat_dream.png
│   ├── elephant_dream.png
│   ├── golden_gate_bridge_dream.png      
│   └── human_dream.png
├── main.ipynb           # Notebook version
├── main.py              # Script version
├── inputs/                    # Input images
│   ├── cat.jpg
│   ├── elephant.jpg
│   ├── golden_gate_bridge.jpg
│   └── human.jpg
├── outputs/                   # Auto-saved results
│   └── .gitkeep
├── README.md
└── requirements.txt
```

---

## 🧠 Tech Stack
- Python 3.10 
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
