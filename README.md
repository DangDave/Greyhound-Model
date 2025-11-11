# Greyhound Racing Model Dashboard 🐾

This Streamlit web app is designed for professional Betfair traders and data analysts focused on **greyhound racing performance**.  
It predicts and displays **Lay** and **Back** selections per race based on historical form data, early speed metrics, and custom score modelling.

---

## 🚀 Features
- Upload daily greyhound runner data (Excel `.xlsx` or `.csv`)
- Automatically clean and prepare the dataset
- Calculate **LayScore** and **BackScore** for each race
- View selections, strike rate, and PnL summaries
- Filter by track, odds range, and race time
- Export results to CSV for backtesting

---

## 🧠 How it Works
The app uses your advanced model logic:
- Weighted factors for `Win%`, `Place%`, `Average(5)`, `Days Last Run`, and early-speed performance  
- Z-score normalization by race  
- Combined probability weighting to rank the top 1 Lay and Back selections per race  
- Visual summary of strike rates and ROI across odds bands  

---

## 🧰 Requirements
All dependencies are listed in `requirements.txt`.  
Main packages:
- Streamlit  
- Pandas  
- NumPy  
- SciPy  
- OpenPyXL  
- Matplotlib / Plotly  

---

## 🕹️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

---

## ☁️ Deployment
To deploy on [Streamlit Cloud](https://share.streamlit.io):
1. Push this repo to GitHub.
2. Click **Deploy** from your Streamlit app or go to [share.streamlit.io](https://share.streamlit.io).
3. Select your repo and main app file.
4. Wait for the build — your app will be live at  
   `https://your-app-name.streamlit.app`.

---

### 🐕 Author
Developed by **David Dangs** — professional Betfair trader & data analyst.  
Optimized for **greyhound racing analytics** and **automated selections**.
