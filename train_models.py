"""Train ensemble models for the Kaudulla elephant movement project."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'kaudulla_elephants_clean.csv'
MODELS=ROOT/'models'; MODELS.mkdir(exist_ok=True)
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
RANDOM_STATE=42
LABELS=['Low Movement','Moderate Movement','High Movement']

warnings.filterwarnings('ignore')

def haversine_kmh(lat1, lon1, lat2, lon2, hours):
    r=6371.0088
    p1=np.radians(lat1); p2=np.radians(lat2)
    dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arctan2(np.sqrt(a),np.sqrt(1-a))/hours

def build_dataset():
    raw=pd.read_csv(DATA)
    raw['datetime']=pd.to_datetime(raw['datetime'],utc=True,errors='coerce')
    raw=raw.dropna(subset=['datetime','lat','lon']).copy()
    raw=raw[raw['lat'].between(5,11) & raw['lon'].between(79,82)].copy()
    raw=raw.sort_values(['collar_id','datetime'])
    grouped=raw.groupby('collar_id',sort=False)
    raw['next_lat']=grouped['lat'].shift(-1); raw['next_lon']=grouped['lon'].shift(-1)
    raw['next_time']=grouped['datetime'].shift(-1)
    raw['elapsed_hours']=(raw['next_time']-raw['datetime']).dt.total_seconds()/3600
    valid=raw['elapsed_hours'].between(0.25,2.5) & raw['next_lat'].notna() & raw['next_lon'].notna()
    raw=raw.loc[valid].copy()
    raw['speed_kmh']=haversine_kmh(raw['lat'].to_numpy(),raw['lon'].to_numpy(),raw['next_lat'].to_numpy(),raw['next_lon'].to_numpy(),raw['elapsed_hours'].to_numpy())
    # Remove implausible GPS jumps before creating the movement classes.
    raw=raw[raw['speed_kmh'].between(0,20)].copy()
    q1,q2=raw['speed_kmh'].quantile([1/3,2/3]).to_numpy()
    raw['movement_class']=pd.cut(raw['speed_kmh'],bins=[-0.001,q1,q2,20],labels=LABELS,include_lowest=True).astype(str)
    dt=raw['datetime'].dt.tz_convert('Asia/Colombo')
    raw['hour']=dt.dt.hour; raw['day_of_week']=dt.dt.dayofweek; raw['month']=dt.dt.month
    raw['is_weekend']=(dt.dt.dayofweek>=5).astype(int)
    raw['day_sin']=np.sin(2*np.pi*raw['hour']/24); raw['day_cos']=np.cos(2*np.pi*raw['hour']/24)
    raw['month_sin']=np.sin(2*np.pi*raw['month']/12); raw['month_cos']=np.cos(2*np.pi*raw['month']/12)
    features=['collar_id','name','sex','lat','lon','hour','day_of_week','month','is_weekend','day_sin','day_cos','month_sin','month_cos']
    data=raw[features+['movement_class','speed_kmh']].replace([np.inf,-np.inf],np.nan).dropna(subset=['movement_class'])
    return raw,data,features

def make_preprocessor():
    categorical=['name','sex']
    numeric=['collar_id','lat','lon','hour','day_of_week','month','is_weekend','day_sin','day_cos','month_sin','month_cos']
    num=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
    cat=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('numeric',num,numeric),('categorical',cat,categorical)]),numeric,categorical

def main():
    raw,data,features=build_dataset()
    q1,q2=data['speed_kmh'].quantile([1/3,2/3]).to_numpy()
    X=data[features]; y=pd.Series(pd.Categorical(data['movement_class'],categories=LABELS).codes,index=data.index,name='movement_class')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE,stratify=y)
    pre,numeric,categorical=make_preprocessor()
    # Lightweight GridSearchCV on a reproducible stratified sample keeps tuning practical.
    tune_n=min(5000,len(X_train))
    tune_X=X_train.sample(n=tune_n,random_state=RANDOM_STATE)
    tune_y=y_train.loc[tune_X.index]
    tuning_pipeline=Pipeline([('preprocessor',make_preprocessor()[0]),('model',RandomForestClassifier(n_estimators=90,class_weight='balanced_subsample',n_jobs=-1,random_state=RANDOM_STATE))])
    grid=GridSearchCV(tuning_pipeline,{'model__max_depth':[12,18],'model__min_samples_leaf':[1,2]},scoring='f1_weighted',cv=3,n_jobs=1,verbose=0)
    grid.fit(tune_X,tune_y)
    tuned_depth=grid.best_params_['model__max_depth']; tuned_leaf=grid.best_params_['model__min_samples_leaf']
    models={
      'Random Forest (Bagging)':RandomForestClassifier(n_estimators=220,max_depth=tuned_depth,min_samples_leaf=tuned_leaf,class_weight='balanced_subsample',n_jobs=-1,random_state=RANDOM_STATE),
      'XGBoost':XGBClassifier(n_estimators=180,max_depth=6,learning_rate=.06,subsample=.85,colsample_bytree=.85,objective='multi:softprob',eval_metric='mlogloss',n_jobs=-1,random_state=RANDOM_STATE),
      'LightGBM':LGBMClassifier(n_estimators=180,max_depth=-1,learning_rate=.06,num_leaves=31,class_weight='balanced',verbosity=-1,n_jobs=-1,random_state=RANDOM_STATE),
    }
    results=[]; fitted={}; predictions={}
    for name,est in models.items():
        pipe=Pipeline([('preprocessor',make_preprocessor()[0]),('model',est)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test)
        results.append({'Model':name,'Family':'Bagging' if 'Random Forest' in name else 'Boosting','Accuracy':accuracy_score(y_test,pred),'Precision (weighted)':precision_score(y_test,pred,average='weighted',zero_division=0),'Recall (weighted)':recall_score(y_test,pred,average='weighted',zero_division=0),'F1-score (weighted)':f1_score(y_test,pred,average='weighted',zero_division=0)})
        fitted[name]=pipe; predictions[name]=pred
        print(name,results[-1]['Accuracy'])
    # Stacking: heterogeneous tree learners with a logistic-regression meta learner.
    stack_est=StackingClassifier(estimators=[('rf',RandomForestClassifier(n_estimators=120,max_depth=16,n_jobs=-1,class_weight='balanced_subsample',random_state=RANDOM_STATE)),('extra',ExtraTreesClassifier(n_estimators=120,max_depth=16,n_jobs=-1,class_weight='balanced',random_state=RANDOM_STATE)),('xgb',XGBClassifier(n_estimators=100,max_depth=5,learning_rate=.07,objective='multi:softprob',eval_metric='mlogloss',n_jobs=-1,random_state=RANDOM_STATE))],final_estimator=LogisticRegression(max_iter=1200,class_weight='balanced'),cv=3,n_jobs=-1,passthrough=False)
    stack=Pipeline([('preprocessor',make_preprocessor()[0]),('model',stack_est)])
    stack.fit(X_train,y_train); pred=stack.predict(X_test)
    results.append({'Model':'Stacking Ensemble','Family':'Stacking','Accuracy':accuracy_score(y_test,pred),'Precision (weighted)':precision_score(y_test,pred,average='weighted',zero_division=0),'Recall (weighted)':recall_score(y_test,pred,average='weighted',zero_division=0),'F1-score (weighted)':f1_score(y_test,pred,average='weighted',zero_division=0)})
    fitted['Stacking Ensemble']=stack; predictions['Stacking Ensemble']=pred
    print('Stacking',results[-1]['Accuracy'])
    # Hybrid: probability-level blend of Random Forest and a scaled neural network.
    hybrid=VotingClassifier(estimators=[('rf',RandomForestClassifier(n_estimators=180,max_depth=18,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=RANDOM_STATE)),('ann',MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',alpha=.002,learning_rate_init=.001,max_iter=180,early_stopping=True,random_state=RANDOM_STATE))],voting='soft',weights=[2,1],n_jobs=-1)
    hybrid_pipe=Pipeline([('preprocessor',make_preprocessor()[0]),('model',hybrid)])
    hybrid_pipe.fit(X_train,y_train); pred=hybrid_pipe.predict(X_test)
    results.append({'Model':'RF + Neural Network (Hybrid)','Family':'Hybrid','Accuracy':accuracy_score(y_test,pred),'Precision (weighted)':precision_score(y_test,pred,average='weighted',zero_division=0),'Recall (weighted)':recall_score(y_test,pred,average='weighted',zero_division=0),'F1-score (weighted)':f1_score(y_test,pred,average='weighted',zero_division=0)})
    fitted['RF + Neural Network (Hybrid)']=hybrid_pipe; predictions['RF + Neural Network (Hybrid)']=pred
    print('Hybrid',results[-1]['Accuracy'])
    # Save the strongest model for interactive prediction.
    metrics=pd.DataFrame(results).sort_values('F1-score (weighted)',ascending=False)
    best_name=metrics.iloc[0]['Model']
    # Save the selected deployment model; all model metrics remain available for comparison.
    joblib.dump(fitted[best_name],MODELS/'best_model.joblib')
    metrics.to_csv(OUT/'model_metrics.csv',index=False)
    cm=confusion_matrix(y_test,predictions[best_name],labels=[0,1,2])
    pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(OUT/'best_confusion_matrix.csv')
    # Feature importance from best available tree estimator (prefer RF for interpretable baseline).
    rf=fitted['Random Forest (Bagging)']
    names=rf.named_steps['preprocessor'].get_feature_names_out()
    imp=rf.named_steps['model'].feature_importances_
    pd.DataFrame({'Feature':names,'Importance':imp}).sort_values('Importance',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
    report=classification_report(y_test,predictions[best_name],labels=[0,1,2],target_names=LABELS,zero_division=0,output_dict=True)
    with open(OUT/'run_summary.json','w') as f:
        json.dump({'rows_in_original_dataset':int(pd.read_csv(DATA).shape[0]),'rows_after_valid_movement_filter':int(len(data)),'class_distribution':{LABELS[int(k)]:int(v) for k,v in pd.Series(y).value_counts().items()},'best_model':best_name,'best_f1_weighted':float(metrics.iloc[0]['F1-score (weighted)']),'labels':LABELS,'features':features,'speed_thresholds_kmh':{'low_to_moderate':float(q1),'moderate_to_high':float(q2)},'random_forest_grid_search':{'best_params':grid.best_params_,'best_cv_weighted_f1':float(grid.best_score_),'tuning_sample_size':int(tune_n)},'classification_report':report},f,indent=2)
    # Small sample for dashboard exploration and map visualisation.
    raw[['name','sex','datetime','lat','lon','speed_kmh','movement_class']].sample(min(4000,len(raw)),random_state=RANDOM_STATE).to_csv(OUT/'movement_sample.csv',index=False)
    print('\nRESULTS\n',metrics.to_string(index=False))
    print('Best:',best_name)

if __name__=='__main__': main()
