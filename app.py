"""Streamlit demonstration: customer churn risk analytics, not a production retention system."""
import pandas as pd
import streamlit as st
from train import predict_dataframe, synthetic_telecom_data

st.set_page_config(page_title='ChurnSense | Telecom AI',layout='wide')
st.title('ChurnSense · Telecom customer churn analytics')
st.caption('Portfolio demo: risk scoring to prioritize exploratory retention analysis. Not a production decision system.')
source=st.file_uploader('Upload an IBM Telco-style CSV for predictions',type=['csv'])
df=pd.read_csv(source) if source else synthetic_telecom_data(600,seed=17).drop(columns=['Churn'])
try:
    prob=predict_dataframe(df)
    output=df.copy();output['churn_probability']=prob
    output['risk_segment']=pd.cut(prob,bins=[-1,.35,.65,1.001],labels=['Low','Medium','High'])
    col1,col2,col3=st.columns(3)
    col1.metric('Customers analyzed',len(output));col2.metric('High-risk customers',int((prob>=.65).sum()))
    col3.metric('Average estimated risk',f'{prob.mean():.1%}')
    st.subheader('Risk distribution')
    st.bar_chart(output['risk_segment'].value_counts().reindex(['Low','Medium','High']).fillna(0))
    st.subheader('Customer prioritization (illustrative)')
    st.dataframe(output.sort_values('churn_probability',ascending=False).head(50),hide_index=True)
    st.download_button('Download scored CSV',output.to_csv(index=False),file_name='telecom_churn_scored.csv',mime='text/csv')
except (FileNotFoundError,ValueError) as exc:
    st.error(f'{exc}. Run `python train.py` before starting the dashboard.')
