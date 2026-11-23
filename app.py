from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import base64
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'kaudulla_elephants_clean.csv'
RESULTS=ROOT/'results'; MODELS=ROOT/'models'
LABELS=['Low Movement','Moderate Movement','High Movement']
COLORS={'Low Movement':'#8da99a','Moderate Movement':'#d8a64b','High Movement':'#df765c'}

st.set_page_config(page_title='Kaudulla | Elephant Movement Lab',page_icon='🐘',layout='wide',initial_sidebar_state='collapsed')
# Real Kaudulla elephant photograph (Wikimedia Commons, CC BY 4.0)
HERO_PHOTO='https://thumb.wikimedia.org/wikipedia/commons/thumb/f/f9/Parc_Nacional_de_Kaudulla_-_Elefants.jpg/960px-Parc_Nacional_de_Kaudulla_-_Elefants.jpg'

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#20372d;--muted:#586b60;--paper:#f4f7f2;--green:#315b47;--line:#dce6dc}
.stApp{background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif}
.block-container{max-width:1400px;padding-top:1.15rem;padding-bottom:2rem}
.stApp,.stApp p,.stApp label,.stApp li{color:#344b3e}
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stCaptionContainer"]{color:#344b3e}
.stApp [data-testid="stPlotlyChart"] text{fill:#263f32!important}
.stApp [data-testid="stPlotlyChart"] .xtick text,
.stApp [data-testid="stPlotlyChart"] .ytick text,
.stApp [data-testid="stPlotlyChart"] .legend text{fill:#263f32!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Manrope',sans-serif!important;color:#20372d!important;letter-spacing:-.025em}
.stApp [data-testid="stCaptionContainer"] p,.stApp small{color:#617267!important}
.stApp [data-testid="stWidgetLabel"] p,.stApp [data-testid="stWidgetLabel"] label{color:#263f32!important;font-weight:650!important}
.stApp input,.stApp textarea,.stApp [data-baseweb="select"]>div{color:#20372d!important;background:#fff!important}
/* Wildlife-green styling for selected multiselect chips across Streamlit/BaseWeb versions */
.stApp [data-baseweb="tag"],
.stApp span[data-baseweb="tag"],
.stApp div[data-baseweb="tag"],
.stApp .stMultiSelect [data-baseweb="tag"]{
    background:#315b47!important;
    background-color:#315b47!important;
    border:1px solid #244a38!important;
    border-radius:8px!important;
    color:#ffffff!important;
    box-shadow:none!important;
}
.stApp [data-baseweb="tag"] *,
.stApp span[data-baseweb="tag"] *,
.stApp div[data-baseweb="tag"] *{
    color:#ffffff!important;
    fill:#ffffff!important;
}
.stApp [data-baseweb="tag"] svg{
    color:#ffffff!important;
    fill:#ffffff!important;
}
.stApp [data-baseweb="select"] [role="combobox"]{
    background:#ffffff!important;
    color:#20372d!important;
}

.stApp [role="radiogroup"] label,.stApp [role="radiogroup"] label p{color:#294436!important;font-weight:650!important}
.stApp [role="radiogroup"]{gap:8px}
.stApp [role="radiogroup"] label{background:#fff;border:1px solid #dce6dc;border-radius:999px;padding:8px 15px}
.stApp [role="radiogroup"] label:has(input:checked){background:#e2eee4;border-color:#789b82}
.hero{background:linear-gradient(90deg,rgba(19,53,39,.96) 0%,rgba(25,67,48,.88) 43%,rgba(22,55,41,.18) 100%),url('https://thumb.wikimedia.org/wikipedia/commons/thumb/f/f9/Parc_Nacional_de_Kaudulla_-_Elefants.jpg/960px-Parc_Nacional_de_Kaudulla_-_Elefants.jpg') center 48%/cover no-repeat;border-radius:26px;padding:42px 42px;color:#fff;position:relative;overflow:hidden;min-height:285px;box-shadow:0 18px 45px #203b2c18}
.hero h1{color:#fff!important;font-size:2.75rem!important;line-height:1.08;margin:.45rem 0 .85rem;max-width:590px}
.hero p{color:#f1f6f0!important;font-size:1.02rem;line-height:1.7;max-width:600px;margin:0}
.eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#d9e9d9!important;font-weight:800}
.hero-credit{position:absolute;right:18px;bottom:13px;color:#fff!important;background:#102b20a8;padding:5px 9px;border-radius:8px;font-size:.65rem}
.metric{background:#fff;border:1px solid var(--line);border-radius:18px;padding:17px 19px;min-height:108px;box-shadow:0 5px 18px #203b2c08}
.metric-label{font-size:.76rem;color:#52675a!important;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.metric-value{font-family:'Manrope';font-size:1.65rem;font-weight:800;color:#20372d!important;margin-top:5px}
.metric-note{font-size:.75rem;color:#65786b!important;margin-top:1px}
.panel{background:#fff;border:1px solid var(--line);border-radius:20px;padding:20px 22px}
.coordinate-note{background:#e3efe5;border:1px solid #b8d0bd;border-left:5px solid #477a58;border-radius:14px;padding:15px 18px;margin:14px 0 18px;color:#244b34!important;font-size:.91rem;line-height:1.65;font-weight:500}
.coordinate-note strong{color:#183d29!important}

.small-note{font-size:.83rem;color:#617267!important}
.elephant-card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:16px;text-align:center;transition:all .25s ease;color:#20372d!important}
.elephant-card:hover{transform:translateY(-5px);box-shadow:0 14px 30px #203b2c12;border-color:#b8cbbd}
div[data-testid="stTabs"] button{font-weight:650!important;color:#294436!important}
.stButton>button,.stFormSubmitButton>button{border-radius:12px;border:1px solid #d5e1d6;background:#fff;color:#315b47;font-weight:700;min-height:42px}
.stButton>button:hover,.stFormSubmitButton>button:hover{border-color:#315b47;color:#1e4033;background:#edf5ed}
[data-testid="stAlert"]{color:#294436!important}
[data-testid="stDataFrame"]{color:#20372d!important}
footer,[data-testid="stFooter"],#MainMenu{visibility:hidden!important;height:0!important}

/* Final override: selected chips in both multiselect widgets */
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"],
.stApp div[data-testid="stMultiSelect"] span[data-baseweb="tag"],
.stApp div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
.stApp [data-testid="stWidgetLabel"] + div [data-baseweb="tag"],
.stApp [data-baseweb="tag"] {
    background: #315b47 !important;
    background-color: #315b47 !important;
    border-color: #244a38 !important;
    color: #ffffff !important;
}
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] [role="button"]:hover {
    background-color: #244a38 !important;
}

/* LAST-PRIORITY OVERRIDE for Streamlit multiselect selected chips */
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"],
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] > span,
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] > div,
.stApp div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    --tag-background: #315b47 !important;
    background: #315b47 !important;
    background-color: #315b47 !important;
    border: 1px solid #244a38 !important;
    color: #ffffff !important;
}
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
.stApp div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
.stApp div[data-testid="stMultiSelect"] [data-baseweb="st-tag"] * {
    color: #ffffff !important;
    fill: #ffffff !important;
}
</style>''',unsafe_allow_html=True)

def show_chart(fig, **kwargs):
    # Apply readable typography and light-green hover tooltips to every Plotly chart.
    hover_style = dict(
        bgcolor='#E8F1E8',
        bordercolor='#789B82',
        font=dict(family='DM Sans, sans-serif', color='#20372d', size=13)
    )
    fig.update_layout(
        template='plotly_white',
        font=dict(family='DM Sans, sans-serif', color='#263f32', size=13),
        title_font=dict(color='#20372d'),
        legend=dict(font=dict(color='#263f32')),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        hoverlabel=hover_style,
        xaxis=dict(tickfont=dict(color='#344b3e'), title_font=dict(color='#344b3e'), gridcolor='#e8eee8'),
        yaxis=dict(tickfont=dict(color='#344b3e'), title_font=dict(color='#344b3e'), gridcolor='#e8eee8')
    )
    # Trace-level settings override layout defaults in some Plotly chart types.
    fig.update_traces(hoverlabel=hover_style)
    return st.plotly_chart(fig, **kwargs)

@st.cache_data
def load_data():
    raw=pd.read_csv(DATA)
    return raw
@st.cache_data
def load_results():
    metrics=pd.read_csv(RESULTS/'model_metrics.csv')
    imp=pd.read_csv(RESULTS/'feature_importance.csv')
    cm=pd.read_csv(RESULTS/'best_confusion_matrix.csv',index_col=0)
    sample=pd.read_csv(RESULTS/'movement_sample.csv')
    summary=json.loads((RESULTS/'run_summary.json').read_text())
    return metrics,imp,cm,sample,summary
@st.cache_resource
def load_model(): return joblib.load(MODELS/'best_model.joblib')

raw=load_data(); metrics,importance,cm,sample,summary=load_results()

st.markdown('''<div class="hero"><div class="eyebrow">WILDLIFE INTELLIGENCE · SRI LANKA</div><h1>Kaudulla Elephant<br>Movement Lab</h1><p>Understanding movement patterns through GPS telemetry and ensemble machine learning. Explore model performance, discover movement signals, and test predictions.</p><div class="hero-credit">Elephants in Kaudulla · Photo: Josep Renalias · CC BY 4.0</div></div>''',unsafe_allow_html=True)
st.write('')
nav=st.radio('Explore the project',['Overview','Model arena','Movement explorer','Live prediction'],horizontal=True,label_visibility='collapsed')
st.write('')

if nav=='Overview':
    st.subheader('A closer look at elephant movement')
    st.caption('A compact overview of the cleaned telemetry data and the trained ensemble suite.')
    c1,c2,c3,c4=st.columns(4)
    vals=[('GPS records',f'{len(raw):,}','Original uploaded dataset'),('Usable movement pairs',f"{summary['rows_after_valid_movement_filter']:,}",'After GPS and time-gap filtering'),('Tracked elephants',str(raw['name'].nunique()),'Unique recorded names'),('Best test F1',f"{summary['best_f1_weighted']:.1%}",summary['best_model'])]
    for col,(label,value,note) in zip([c1,c2,c3,c4],vals):
        with col: st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
    st.write('')
    left,right=st.columns([1.15,.85],gap='large')
    with left:
        st.markdown('#### Movement profile')
        dist=pd.DataFrame({'Movement':list(summary['class_distribution'].keys()),'Records':list(summary['class_distribution'].values())})
        fig=px.pie(dist,names='Movement',values='Records',hole=.68,color='Movement',color_discrete_map=COLORS)
        fig.update_traces(textposition='outside',textinfo='percent+label',marker=dict(line=dict(color='white',width=3)))
        fig.update_layout(height=330,margin=dict(t=12,b=12,l=12,r=12),legend_title=None,paper_bgcolor='white',plot_bgcolor='white')
        show_chart(fig,width='stretch')
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('#### Meet the movement classes')
        movement_cards = [
            ('Low Movement', '0 – 0.090 km/h', 'Lower third of observed GPS displacement speeds', '🐘'),
            ('Moderate Movement', 'Above 0.090 – 0.238 km/h', 'Middle third of observed GPS displacement speeds', '🐘'),
            ('High Movement', 'Above 0.238 km/h', 'Upper third of observed GPS displacement speeds', '🐘'),
        ]
        for label, speed_range, desc, emoji in movement_cards:
            st.markdown(
                f'<div class="elephant-card" style="margin:10px 0">'
                f'<div style="font-size:2.3rem;transition:transform .3s">{emoji}</div>'
                f'<b>{label}</b><div style="font-weight:700;color:#315b47;margin:5px 0">{speed_range}</div>'
                f'<div class="small-note">{desc}</div></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>',unsafe_allow_html=True)
    st.info('Movement labels are engineered from consecutive GPS fixes using dataset-specific speed tertiles. They are estimates of displacement speed, not direct observations of an elephant’s behaviour.')

elif nav=='Model arena':
    st.subheader('Ensemble model comparison')
    st.caption('All models use the same stratified 80/20 train-test split and weighted evaluation metrics.')
    sort_metric=st.selectbox('Rank models by',['F1-score (weighted)','Accuracy','Recall (weighted)','Precision (weighted)'])
    view=metrics.sort_values(sort_metric,ascending=False)
    best=view.iloc[0]
    k1,k2,k3=st.columns(3)
    for col,(lab,val) in zip([k1,k2,k3],[('Top model',best['Model']),('Test accuracy',f"{best['Accuracy']:.1%}"),('Weighted F1',f"{best['F1-score (weighted)']:.1%}")]):
        with col: st.markdown(f'<div class="metric"><div class="metric-label">{lab}</div><div class="metric-value" style="font-size:1.25rem">{val}</div></div>',unsafe_allow_html=True)
    st.write('')
    fig=px.bar(view,x='Model',y=sort_metric,color='Family',text=view[sort_metric].map(lambda x:f'{x:.1%}'),barmode='group',color_discrete_sequence=['#789783','#d4a65b','#8c9bb5','#d77d67'])
    fig.update_traces(textposition='outside',cliponaxis=False)
    fig.update_layout(height=420,margin=dict(t=25,b=10,l=10,r=10),yaxis=dict(tickformat='.0%',range=[0,1.08]),xaxis_title='',legend_title='',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white')
    show_chart(fig,width='stretch')
    st.dataframe(metrics.style.format({c:'{:.3f}' for c in ['Accuracy','Precision (weighted)','Recall (weighted)','F1-score (weighted)']}),width='stretch',hide_index=True)
    

elif nav=='Movement explorer':
    st.subheader('Explore GPS movement observations')
    st.caption('A sampled subset of valid movement pairs is shown for responsiveness.')
    names=sorted(sample['name'].dropna().unique())
    chosen=st.multiselect('Filter elephant names',names,default=names[:min(5,len(names))])
    classes=st.multiselect('Movement class',LABELS,default=LABELS)
    d=sample[sample['name'].isin(chosen)&sample['movement_class'].isin(classes)].copy()
    a,b,c=st.columns(3)
    a.metric('Visible observations',f'{len(d):,}')
    b.metric('Elephants selected',d['name'].nunique())
    c.metric('Median speed',f"{d['speed_kmh'].median():.2f} km/h" if len(d) else '—')
    if len(d):
        fig=px.scatter_map(
            d, lat='lat', lon='lon', color='movement_class',
            hover_name='name',
            hover_data={'speed_kmh':':.2f','datetime':True,'lat':False,'lon':False},
            center={'lat': 7.95, 'lon': 80.75},
            zoom=8.4, height=520,
            color_discrete_map=COLORS,
            map_style='open-street-map'
        )
        fig.update_layout(margin=dict(t=0,b=0,l=0,r=0),legend_title='Movement')
        show_chart(fig,width='stretch')
        by_name=d.groupby('name',as_index=False)['speed_kmh'].median().sort_values('speed_kmh',ascending=False)
        st.markdown('#### Median observed speed by elephant')
        show_chart(px.bar(by_name,x='name',y='speed_kmh',color='speed_kmh',color_continuous_scale=['#9ab6a0','#315b47']),width='stretch')
    else: st.warning('No observations match these filters. Select at least one elephant and movement class.')
    st.caption('Map points are sampled GPS observations, not a continuous route. Missing coordinates and long time gaps are excluded from the movement-pair dataset.')

elif nav=='Live prediction':
    st.subheader('Try a movement prediction')
    st.caption('Enter a hypothetical observation context. The saved best-performing model returns a class estimate.')
    st.markdown('''<div class="coordinate-note"><strong>Understanding the coordinates</strong><br>Latitude tells us how far north or south a location is (Kaudulla is around 8° N). Longitude tells us how far east or west it is (around 80–81° E). The values below are editable example coordinates near the Kaudulla area — they are not a live GPS location.</div>''', unsafe_allow_html=True)
    elephants=raw[['collar_id','name','sex']].drop_duplicates().sort_values('name')
    selected_name=st.selectbox('Elephant',sorted(elephants['name'].unique()))
    row=elephants[elephants['name']==selected_name].iloc[0]
    with st.form('prediction_form'):
        c1,c2,c3=st.columns(3)
        with c1: lat=st.number_input('Latitude (°N)',min_value=5.0,max_value=11.0,value=8.1,step=.001,format='%.5f',help='North–south coordinate. Positive values are north of the Equator.')
        with c2: lon=st.number_input('Longitude (°E)',min_value=79.0,max_value=82.0,value=80.9,step=.001,format='%.5f',help='East–west coordinate. Positive values are east of the Prime Meridian.')
        with c3: hour=st.slider('Local hour',0,23,12)
        c4,c5=st.columns(2)
        with c4: day=st.selectbox('Day of week',['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        with c5: month=st.selectbox('Month',list(range(1,13)),index=6)
        submitted=st.form_submit_button('Predict movement class',width='stretch')
    if submitted:
        day_num=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(day)
        x=pd.DataFrame([{'collar_id':int(row['collar_id']),'name':row['name'],'sex':row['sex'],'lat':lat,'lon':lon,'hour':hour,'day_of_week':day_num,'month':month,'is_weekend':int(day_num>=5),'day_sin':np.sin(2*np.pi*hour/24),'day_cos':np.cos(2*np.pi*hour/24),'month_sin':np.sin(2*np.pi*month/12),'month_cos':np.cos(2*np.pi*month/12)}])
        try:
            model = load_model()
        except Exception as exc:
            st.error("The saved model could not be loaded. This is usually caused by a Python or scikit-learn version mismatch, or a missing model dependency.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.info("Use Python 3.11 or 3.12, reinstall requirements.txt in a fresh .venv, then run: python train_models.py")
            st.stop()
        code=int(np.asarray(model.predict(x)).ravel()[0]); label=LABELS[code]
        proba=model.predict_proba(x)[0]
        st.markdown(f'<div class="panel" style="margin-top:18px;text-align:center"><div style="font-size:3rem">🐘</div><div class="eyebrow" style="color:#789783">MODEL ESTIMATE</div><h2>{label}</h2><div class="small-note">Predicted movement class for the entered context</div></div>',unsafe_allow_html=True)
        prob_df=pd.DataFrame({'Movement':LABELS,'Probability':proba})
        fig=px.bar(prob_df,x='Movement',y='Probability',color='Movement',text=prob_df['Probability'].map(lambda v:f'{v:.1%}'),color_discrete_map=COLORS)
        fig.update_layout(yaxis_tickformat='.0%',yaxis_range=[0,1],showlegend=False,height=300,margin=dict(t=20,b=10,l=10,r=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white')
        fig.update_traces(textposition='outside')
        show_chart(fig,width='stretch')
        st.warning('This is a model estimate based on historical patterns, not a real-time GPS reading or a wildlife safety recommendation.')

elif nav=='Methodology':
    st.subheader('From GPS points to movement classes')
    st.markdown('''<div class="panel"><h4>01 · Prepare the data</h4><p>Parse timestamps, remove rows without valid GPS coordinates, retain plausible Sri Lankan coordinates, and sort records by collar and time.</p><hr><h4>02 · Engineer the target</h4><p>Pair each GPS fix with the next fix for the same collar. Keep time gaps between 15 minutes and 2.5 hours, calculate great-circle distance with the Haversine formula, and convert it to km/h. Remove implausible speeds above 20 km/h. Define Low, Moderate, and High Movement using the 33rd and 67th percentiles of observed speed to avoid an almost-empty high-speed class.</p><hr><h4>03 · Create predictors</h4><p>Use collar ID, elephant name, sex, current latitude/longitude, local hour, weekday, month, weekend indicator, and cyclical time features. The next GPS coordinates and calculated speed are not model inputs.</p><hr><h4>04 · Train and evaluate</h4><p>Use a stratified 80/20 split, impute missing feature values, standardise numeric features, one-hot encode categorical features, and compare the ensemble families using weighted metrics.</p></div>''',unsafe_allow_html=True)
    st.markdown('#### Feature importance · Random Forest')
    top=importance.head(12).sort_values('Importance')
    fig=px.bar(top,x='Importance',y='Feature',orientation='h',color='Importance',color_continuous_scale=['#c8d9ca','#315b47'])
    fig.update_layout(height=430,margin=dict(t=15,b=10,l=10,r=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='white',coloraxis_showscale=False)
    show_chart(fig,width='stretch')
    st.markdown('#### Confusion matrix · best model')
    fig=px.imshow(cm,text_auto=True,aspect='auto',color_continuous_scale='Greens',labels=dict(x='Predicted',y='Actual',color='Count'))
    fig.update_layout(height=380,margin=dict(t=20,b=10,l=10,r=10))
    show_chart(fig,width='stretch')
    st.markdown('#### Research reflection')
    st.markdown('''Tree ensembles are often effective on structured tabular data because they capture nonlinear feature interactions and do not require image-like spatial representations. Neural networks can model more complex interactions but generally need careful scaling and tuning. These experiments concern engineered tabular telemetry features; they do not test raw image or text inputs. The high scores should be interpreted cautiously because the random split may place observations from the same elephant in both training and test sets. A collar-wise holdout would be a stronger test of generalisation to previously unseen elephants.''')
    st.markdown('#### Limitations')
    st.markdown('''- GPS fixes can be sparse or missing; movement classes are derived from observed displacement rather than continuous motion.
- The speed thresholds are dataset-specific tertiles and may not match validated ethological categories.
- GPS noise, terrain, and time gaps can influence estimated speed.
- A random split can produce optimistic estimates for known collars and individuals.''')


