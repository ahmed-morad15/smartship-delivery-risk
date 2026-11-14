# Deploy SmartShip to Streamlit Community Cloud

The app is designed to run from the repository root and uses the saved artifact `models/smartship_pipeline.joblib`. The raw Olist CSV files are not needed by the deployed app; they are needed only to rebuild the dataset and retrain the model.

1. Create a GitHub repository and push the SmartShip project.
2. Ensure `models/smartship_pipeline.joblib` is committed (it is intentionally not ignored).
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app/main.py`.
5. Deploy. Streamlit Cloud installs dependencies from `requirements.txt`.
6. Open the generated app URL and test at least two different input combinations.

The model artifact was serialized with scikit-learn 1.8.0. Keep the pinned scikit-learn, NumPy, pandas, SciPy, and joblib versions in `requirements.txt` aligned with the environment used to generate the artifact.

## Local run

```powershell
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Do not claim the app is publicly deployed until the Streamlit Cloud deployment has completed and the URL has been tested.
