# Introduction:
Streamlit app for predicting activity for PDE5A. Developed with Python 3.11

# Requirements:
streamlit==1.24.1  
matplotlib==3.7.1  
numpy==1.23.5  
pandas==1.5.3  
scikit-learn==1.2.1  
flask==1.1.2  
flask_wtf==1.0.1  
jinja2==2.11.3  
click==8.1.7  
molvs==0.1.1  
rdkit==2023.03.2  

# Installation:
git clone https://github.com/ncordeirfcup/PDE5A_Streamlit.git (or download the zip file and extract).

# Running:
Open Anaconda prompt and navigate to PDE5A_Streamlit directory. Run the following commands:  
conda create --name myenv python=3.11.3  
conda activate myenv  
pip install -r requirements.txt  
streamlit run app.py  

