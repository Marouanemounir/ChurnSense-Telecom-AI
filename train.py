"""ChurnSense: reproducible deep-learning baseline for telecom customer churn.
By default trains on clearly-labelled synthetic data for an executable demonstration.
Use --csv /path/to/IBM-Telco-Customer-Churn.csv for a public IBM sample dataset.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SEED = 42


def synthetic_telecom_data(n=6000, seed=SEED):
    """Simulation for reproducibility; NOT actual telecom subscriber data."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, size=n)
    contract = rng.choice(['Month-to-month', 'One year', 'Two year'], size=n, p=[.55,.26,.19])
    internet = rng.choice(['DSL','Fiber optic','No'],size=n,p=[.40,.47,.13])
    pay = rng.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],n,p=[.34,.20,.24,.22])
    support = rng.choice(['Yes','No','No internet service'], n, p=[.30,.57,.13])
    security = rng.choice(['Yes','No','No internet service'], n, p=[.28,.59,.13])
    charges = np.clip(rng.normal(65,22,n),18,120)
    age_flag = rng.binomial(1,.16,n)
    paperless = rng.choice(['Yes','No'], n,p=[.65,.35])
    score = (-1.4 + 1.20*(contract=='Month-to-month') - .9*(contract=='Two year')
             - .030*tenure + .70*(internet=='Fiber optic') + .73*(support=='No')
             + .50*(security=='No') + .60*(pay=='Electronic check')
             + .014*(charges-65) + .20*(paperless=='Yes') + .18*age_flag)
    probs = 1 / (1+np.exp(-score))
    churn = np.where(rng.random(n)<probs,'Yes','No')
    df=pd.DataFrame({
      'customerID':[f'DEMO-{i:05d}' for i in range(n)],
      'gender':rng.choice(['Male','Female'],n), 'SeniorCitizen':age_flag,
      'Partner':rng.choice(['Yes','No'],n), 'Dependents':rng.choice(['Yes','No'],n),
      'tenure':tenure, 'PhoneService':rng.choice(['Yes','No'],n,p=[.9,.1]),
      'MultipleLines':rng.choice(['Yes','No'],n), 'InternetService':internet,
      'OnlineSecurity':security, 'OnlineBackup':rng.choice(['Yes','No'],n),
      'DeviceProtection':rng.choice(['Yes','No'],n), 'TechSupport':support,
      'StreamingTV':rng.choice(['Yes','No'],n), 'StreamingMovies':rng.choice(['Yes','No'],n),
      'Contract':contract, 'PaperlessBilling':paperless, 'PaymentMethod':pay,
      'MonthlyCharges':np.round(charges,2),
      'TotalCharges':np.round(np.maximum(0,tenure*charges+rng.normal(0,100,n)),2),
      'Churn':churn
    })
    return df


def make_model(n_features:int):
    return nn.Sequential(nn.Linear(n_features,64),nn.ReLU(),nn.Dropout(.20),
                         nn.Linear(64,24),nn.ReLU(),nn.Linear(24,1))


def load_data(csv=None):
    df=pd.read_csv(csv) if csv else synthetic_telecom_data()
    if 'Churn' not in df.columns:
        raise ValueError('The dataset must contain the Churn target column.')
    df=df.copy()
    if 'TotalCharges' in df: df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df=df.drop_duplicates(subset=['customerID']) if 'customerID' in df.columns else df.drop_duplicates()
    y=df['Churn'].map({'Yes':1,'No':0}) if df['Churn'].dtype=='object' else df['Churn'].astype(int)
    if y.isna().any(): raise ValueError('Churn must contain Yes/No or 0/1 labels.')
    x=df.drop(columns=['Churn','customerID'],errors='ignore')
    return x,y.astype(int),df


def train(csv=None,epochs=60,outdir=ROOT/'artifacts'):
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    x,y,df=load_data(csv)
    x_train,x_tmp,y_train,y_tmp=train_test_split(x,y,test_size=.30,random_state=SEED,stratify=y)
    x_val,x_test,y_val,y_test=train_test_split(x_tmp,y_tmp,test_size=.50,random_state=SEED,stratify=y_tmp)
    nums=x_train.select_dtypes(include=['number']).columns.tolist()
    cats=[c for c in x_train.columns if c not in nums]
    preprocess=ColumnTransformer([
        ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
        ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)
    ],sparse_threshold=0)
    tr=preprocess.fit_transform(x_train).astype('float32')
    va=preprocess.transform(x_val).astype('float32')
    te=preprocess.transform(x_test).astype('float32')
    model=make_model(tr.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    npositive=int(y_train.sum());nnegative=len(y_train)-npositive
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nnegative/max(npositive,1)],dtype=torch.float32))
    loader=DataLoader(TensorDataset(torch.from_numpy(tr),torch.tensor(y_train.to_numpy(),dtype=torch.float32).unsqueeze(1)),batch_size=128,shuffle=True)
    bestloss=float('inf'); beststate=None; patience=0
    for epoch in range(epochs):
        model.train()
        for xb,yb in loader:
            opt.zero_grad();loss=lossfn(model(xb),yb);loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            val_loss=float(lossfn(model(torch.from_numpy(va)),torch.tensor(y_val.to_numpy(),dtype=torch.float32).unsqueeze(1)))
        if val_loss<bestloss-1e-5:
            bestloss=val_loss;beststate={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
        else:
            patience+=1
            if patience>=10:break
    model.load_state_dict(beststate);model.eval()
    with torch.no_grad():
        preds=torch.sigmoid(model(torch.from_numpy(te))).flatten().numpy()
    pred_label=(preds>=.5).astype(int)
    metrics={'source':'IBM Telco public sample' if csv else 'SIMULATED DATA (not real customers)',
             'samples':len(df),'train_count':len(x_train),'validation_count':len(x_val),'test_count':len(x_test),
             'epochs_used':epoch+1,'roc_auc':float(roc_auc_score(y_test,preds)),
             'pr_auc':float(average_precision_score(y_test,preds)),
             'f1':float(f1_score(y_test,pred_label)),'precision':float(precision_score(y_test,pred_label)),
             'recall':float(recall_score(y_test,pred_label)),
             'accuracy':float(accuracy_score(y_test,pred_label))}
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),outdir/'model.pt')
    joblib.dump({'preprocessor':preprocess,'features':x.columns.tolist(),'n_features':tr.shape[1]},outdir/'pipeline.joblib')
    (outdir/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metrics,indent=2,ensure_ascii=False))
    return metrics


def predict_dataframe(data:pd.DataFrame,artifacts=ROOT/'artifacts'):
    artifacts=Path(artifacts)
    bundle=joblib.load(artifacts/'pipeline.joblib')  # trusted local artifact only
    df=data.copy()
    if 'TotalCharges' in df: df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    missing=[f for f in bundle['features'] if f not in df]
    if missing: raise ValueError('Missing expected columns: '+', '.join(missing))
    arr=bundle['preprocessor'].transform(df[bundle['features']]).astype('float32')
    net=make_model(bundle['n_features'])
    net.load_state_dict(torch.load(artifacts/'model.pt',map_location='cpu',weights_only=True))
    net.eval()
    with torch.no_grad():probas=torch.sigmoid(net(torch.from_numpy(arr))).flatten().numpy()
    return probas

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Train a PyTorch telecom churn MLP (synthetic demo by default).')
    p.add_argument('--csv',type=str,default=None,help='Path to IBM Telco customer churn CSV')
    p.add_argument('--epochs',type=int,default=60)
    args=p.parse_args();train(args.csv,args.epochs)
