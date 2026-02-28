# AI-Based Fraud Detection and Risk Segmentation

Analyze large-scale financial transactions for fraud prevention strategy.

## Business Problem Statement
Financial fraud and revenue leakage are critical risks in digital payments. This project applies AI techniques to minimize fraud loss while optimizing revenue through risk-based segmentation.

### Economic Concepts Applied
- **Asymmetric Information**: Identifying hidden intent through data patterns.
- **Risk Premium**: Tailoring strategies for high-risk segments.
- **Revenue Optimization**: Balancing security and customer friction.

## AI Techniques Used
- **Data Preprocessing**: Handling currency, time features, and missing values.
- **Exploratory Data Analysis**: Visualizing fraud patterns and anomalies.
- **K-Means Clustering**: Segmenting transactions into Low, Medium, and High risk.
- **Logistic Regression**: Predicting fraudulent transactions with high precision and recall.

## Dataset Link
[Kaggle Dataset: Financial Transaction Fraud](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)

## Project Structure
- `fraud_detection.ipynb`: Complete Google Colab notebook.
- `streamlit_app.py`: Interactive dashboard for deployment.
- `fraud_model.pkl`: Trained Logistic Regression model.
- `scaler.pkl`: StandardScaler for feature transformation.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn seaborn matplotlib streamlit joblib`.
3. Run the Streamlit app: `streamlit run streamlit_app.py`.
4. Open the notebook in Google Colab to see the full analysis.

## Model Results
| Metric | Score |
| --- | --- |
| Accuracy | 99% |
| Precision | 0.95 |
| Recall | 0.90 |
| F1-Score | 0.92 |

*Note: Results are based on synthetic labels generated for demonstration purposes.*
