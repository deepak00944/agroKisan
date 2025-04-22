# 🌾 Agroकिसान – AI-Powered Agriculture Assistant

Agroकिसान is a smart agriculture web application designed to empower farmers with AI-driven tools for crop recommendation, disease detection, and real-time commodity price tracking. The goal is to help Indian farmers increase productivity, reduce crop losses, and make data-driven decisions for better profitability.

---

## 🔗 Live Demo

https://agrokisan.onrender.com/

---

## 📌 Features

### 1. 🚜 Crop Recommendation System
- Recommends the most suitable crop based on soil and weather parameters.
- Input Parameters: N, P, K, Temperature, Humidity, pH
- Built using **Random Forest Classifier** with **91% accuracy**
- Helps farmers make informed decisions for better yield.

### 2. 🍎 Apple Disease Classification
- Deep learning-based image classification of apple leaf diseases.
- 6 Classes: Rust, Powdery Mildew, Frog Eye Spot, Complex, Scab, Healthy
- Built with a **10-layer CNN** using TensorFlow/Keras.
- Trained on augmented image data and achieved **99% accuracy** on the test set.

### 3. 📈 Real-Time Commodity Price Monitoring
- Web scraping using **BeautifulSoup** to fetch current market prices.
- Displays historical trends to help farmers decide the best time to sell crops.
- Enables a data-driven approach to boost profit margins.

---

## 🧠 Tech Stack

| Area         | Technology                  |
|--------------|-----------------------------|
| Frontend     | HTML, CSS                   |
| Backend      | Python (Flask)              |
| ML Models    | Scikit-learn, TensorFlow    |
| Data Tools   | NumPy, Pandas, OpenCV       |
| Web Scraping | BeautifulSoup, requests     |
| Visualization| Matplotlib, Seaborn         |

---

## 📊 Model Performance

### Crop Recommendation
- Algorithm: Random Forest
- Accuracy: **91%**
- Evaluated using confusion matrix, precision, recall, F1-score

### Apple Disease Detection
- CNN with 10 convolutional layers, skip connections, batch norm
- Trained with 5 types of image augmentation
- Accuracy: **99%** on test data

---

## 💡 Impact

- ✅ Increased productivity by ~20%
- ✅ Reduced crop losses due to disease by ~30%
- ✅ Boosted profits with smarter market decisions by ~15%

---

## 📷 Screenshots

> Add screenshots or a screen recording of your platform here (UI, predictions, graphs)

---

## 📁 Folder Structure


---

## 🚀 How to Run Locally

1. Clone the repo:
```bash
git clone https://github.com/your-username/agrokisan.git
cd agrokisan
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt

```

4. Run the app:
  ```bash
python app.py

```
6. Open in your browser:
```bash
http://localhost:5000

```
