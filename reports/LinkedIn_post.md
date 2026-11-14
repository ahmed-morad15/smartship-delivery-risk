# LinkedIn project announcement draft

I’m excited to share **SmartShip – E-commerce Delivery Risk Intelligence**, a machine learning project built using the Brazilian E-Commerce Public Dataset by Olist.

The project explores how order-time information—such as estimated delivery lead time, item and shipping details, product characteristics, and approximate customer–seller distance—can help identify orders at risk of late delivery.

I built an order-level data pipeline across multiple relational tables, engineered leakage-aware features, compared Logistic Regression, Random Forest, and Extra Trees, and developed a Streamlit interface for interactive risk estimation.

On the stratified holdout set, the selected Extra Trees model achieved:
- Recall (late orders): 67.48%
- F1-score (late orders): 23.22%
- ROC-AUC: 0.7081
- PR-AUC: 0.2150

Because late orders are a minority class, the model prioritizes detecting more late deliveries while accepting false positives. A separate chronological experiment also highlighted temporal distribution shift, so the current results should not be interpreted as proof of future production performance.

**Tech stack:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit.

GitHub: [Add repository URL]
Live demo: [Add deployed Streamlit URL]

#MachineLearning #DataScience #Python #ScikitLearn #Streamlit #Ecommerce #AI
