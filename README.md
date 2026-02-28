# 🛡️ FinSafe: AI-Based Fraud Detection & Risk Segmentation
Analyze large-scale financial transactions for fraud prevention strategy.

**GitHub Repository**: [transactional-fraud-detection-model](https://github.com/YadavSourabhGH/transactional-fraud-detection-model.git)

## 🏢 Business Problem Statement
Financial fraud and revenue leakage are critical risks in digital payments. This project applies AI techniques to minimize fraud loss while optimizing revenue through risk-based segmentation.

### Economic Concepts Applied
- **Asymmetry of Information**: Identifying hidden intent through data patterns.
- **Risk Premium**: Tailoring strategies for high-risk segments.
- **Revenue Optimization**: Balancing security and customer friction.

## 🤖 AI Techniques Used
- **Data Preprocessing**: Handling currency symbols, negative values, and time features.
- **K-Means Clustering**: Segmenting transactions into Low, Medium, and High risk.
- **Logistic Regression**: Predicting fraudulent transactions with high accuracy.

## 📁 Project Structure
- `fraud_detection.ipynb`: Complete Google Colab notebook for the ML pipeline.
- `app.py`: Interactive Streamlit dashboard for real-time prediction.
- `requirements.txt`: Python package dependencies.
- `fraud_model.pkl` & `scaler.pkl`: Trained model artifacts (generated from notebook).

## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YadavSourabhGH/transactional-fraud-detection-model.git
   cd transactional-fraud-detection-model
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🧪 Interactive Test Cases
Try these values in the Streamlit app to see the AI in action:

| Case | Amount | Hour | Method | Expected Result |
| :--- | :--- | :--- | :--- | :--- |
| **High Risk** | $4,500.00 | 3 AM | Online Transaction | ⚠️ Fraud Detected |
| **Medium Risk** | $1,200.00 | 2 PM | Online Transaction | ✅ Safe (Secondary Review) |
| **Low Risk** | $25.00 | 10 AM | Swipe Transaction | ✅ Safe (Instant Approval) |
| **Refund** | -$150.00 | 11 AM | Chip Transaction | ✅ Safe (Low Risk) |

## 📊 Model Performance
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99% |
| **Precision** | 0.95 |
| **Recall** | 0.90 |
| **F1-Score** | 0.92 |

---
*Note: This project is part of a Business AI Mini-Project. Results are based on labels generated for strategy demonstration.*
