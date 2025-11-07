
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, RocCurveDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Insurance Policy Status — Analytics & Prediction", layout="wide")

def find_column(df, candidates, contains_any=None):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if contains_any:
        for c in cols:
            lc = c.lower()
            if any(sub in lc for sub in contains_any):
                return c
    return None

def detect_target(df):
    for c in df.columns:
        if c.lower().replace(" ", "") in ["policystatus", "policy_status", "policy-status", "policystat", "policystatusflag"]:
            return c
    return find_column(df, ["policy status", "PolicyStatus", "Policy Status"])

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

def split_xy(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def build_preprocess(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocess = ColumnTransformer(
        transformers=[('num', numeric_transformer, num_cols),
                      ('cat', categorical_transformer, cat_cols)],
        remainder='drop'
    )
    return preprocess, num_cols, cat_cols

def get_feature_names(preprocess, num_cols, cat_cols):
    names = []
    names.extend(num_cols)
    if len(cat_cols):
        ohe = preprocess.named_transformers_['cat'].named_steps['onehot']
        names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    return names

def models_dict():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced_subsample'),
        "Gradient Boosted": GradientBoostingClassifier(random_state=42)
    }

def pos_label_from_y(y):
    unique = pd.Series(y.unique())
    if "Active" in list(unique): return "Active"
    if "Lapsed" in list(unique): return "Lapsed"
    if "Yes" in list(unique): return "Yes"
    if 1 in list(unique): return 1
    return unique.iloc[min(1, len(unique)-1)]

def plot_confusion(cm, labels, title, cmap):
    fig, ax = plt.subplots(figsize=(5,4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format='d', cmap=cmap, ax=ax, colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig

def run_training(df, target):
    X, y = split_xy(df, target)
    preprocess, num_cols, cat_cols = build_preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos_label = pos_label_from_y(y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    figs_conf = []
    roc_fig = plt.figure(figsize=(7,5))
    roc_ax = roc_fig.gca()
    feat_imps = {}

    for name, clf in models_dict().items():
        pipe = Pipeline(steps=[('preprocess', preprocess), ('model', clf)])
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
            cv_metric = ("ROC-AUC (CV)", float(np.mean(cv_scores)), float(np.std(cv_scores)))
        except Exception:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
            cv_metric = ("Accuracy (CV)", float(np.mean(cv_scores)), float(np.std(cv_scores)))

        pipe.fit(X_train, y_train)
        y_pred_tr = pipe.predict(X_train)
        y_pred_te = pipe.predict(X_test)

        if hasattr(pipe.named_steps['model'], "predict_proba"):
            y_prob_te = pipe.predict_proba(X_test)[:,1]
        elif hasattr(pipe.named_steps['model'], "decision_function"):
            from sklearn.preprocessing import MinMaxScaler
            y_prob_te = MinMaxScaler().fit_transform(pipe.decision_function(X_test).reshape(-1,1)).ravel()
        else:
            y_prob_te = None

        tr_acc = accuracy_score(y_train, y_pred_tr)
        te_acc = accuracy_score(y_test, y_pred_te)
        precision = precision_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0)
        auc = roc_auc_score((y_test==pos_label).astype(int), y_prob_te) if y_prob_te is not None else np.nan

        metrics.append({
            "Algorithm": name,
            "Training Accuracy": round(tr_acc,4),
            "Testing Accuracy": round(te_acc,4),
            "Precision (Test)": round(precision,4),
            "Recall (Test)": round(recall,4),
            "F1 Score (Test)": round(f1,4),
            "AUC (Test)": round(auc,4) if not np.isnan(auc) else None,
            f"{cv_metric[0]} Mean": round(cv_metric[1],4),
            f"{cv_metric[0]} Std": round(cv_metric[2],4)
        })

        labels_display = sorted(y.unique().tolist())[:2] if len(y.unique())>=2 else y.unique().tolist()
        cm_tr = confusion_matrix(y_train, y_pred_tr, labels=labels_display)
        cm_te = confusion_matrix(y_test, y_pred_te, labels=labels_display)
        figs_conf.append(("train", name, plot_confusion(cm_tr, labels_display, f"{name} - Training Confusion Matrix", "Blues")))
        figs_conf.append(("test", name, plot_confusion(cm_te, labels_display, f"{name} - Test Confusion Matrix", "Greens")))

        if y_prob_te is not None:
            RocCurveDisplay.from_predictions((y_test==pos_label).astype(int), y_prob_te, name=name, ax=roc_ax)

        importances = getattr(pipe.named_steps['model'], "feature_importances_", None)
        if importances is not None:
            feature_names = get_feature_names(pipe.named_steps['preprocess'], num_cols, cat_cols)
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15).reset_index()
            fi.columns = ["Feature","Importance"]
            feat_imps[name] = fi

    roc_ax.set_title("ROC Curves — All Models")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.grid(True, linestyle="--", alpha=0.4)
    roc_fig.tight_layout()
    return pd.DataFrame(metrics).set_index("Algorithm"), figs_conf, roc_fig, feat_imps

def get_feature_names(preprocess, num_cols, cat_cols):
    names = []
    names.extend(num_cols)
    if len(cat_cols):
        ohe = preprocess.named_transformers_['cat'].named_steps['onehot']
        names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    return names

def charts_section(df, target):
    st.subheader("📊 Insight Dashboard")

    def find_col_contains(any_list):
        for c in df.columns:
            lc = c.lower()
            if any(s in lc for s in any_list):
                return c
        return None

    role_like = find_col_contains(["policytype","policy type","product","plan","segment"])
    sat_like = find_col_contains(["satisfaction","score","rating"]) or find_col_contains(["premium","age","tenure","duration"])

    with st.expander("Filters", expanded=True):
        c1, c3 = st.columns([1,2])
        selected_roles = []
        if role_like:
            roles = sorted(df[role_like].dropna().astype(str).unique().tolist())
            selected_roles = c1.multiselect(f"Filter by {role_like}", roles, default=roles)
        sat_min = sat_max = None
        if sat_like and pd.api.types.is_numeric_dtype(df[sat_like]):
            smin, smax = float(np.nanmin(df[sat_like])), float(np.nanmax(df[sat_like]))
            sat_min, sat_max = c3.slider(f"Range for {sat_like}", min_value=smin, max_value=smax, value=(smin, smax))

    dff = df.copy()
    if role_like and selected_roles:
        dff = dff[dff[role_like].astype(str).isin(selected_roles)]
    if sat_like and pd.api.types.is_numeric_dtype(df[sat_like]):
        dff = dff[(dff[sat_like] >= sat_min) & (dff[sat_like] <= sat_max)]

    # 1 Bar: positive rate by role_like
    if role_like:
        rate = dff.groupby(role_like)[target].apply(lambda s: (s.astype(str).str.lower().isin(['active','yes','1','true','approved','open'])).mean()).reset_index(name='PositiveRate')
        st.altair_chart(alt.Chart(rate).mark_bar().encode(
            x=alt.X(role_like, sort='-y'),
            y=alt.Y('PositiveRate:Q', axis=alt.Axis(format='%')),
            tooltip=[role_like, alt.Tooltip('PositiveRate:Q', format='.1%')]
        ).properties(title=f"{target} Positive Rate by {role_like}"), use_container_width=True)

    # 2 Stacked: region vs status
    region_like = find_col_contains(["region","state","city","area","zone"])
    if region_like:
        g = dff.groupby([region_like, target]).size().reset_index(name='Count')
        st.altair_chart(alt.Chart(g).mark_bar().encode(
            x=f'{region_like}:N', y='Count:Q', color=f'{target}:N', tooltip=[region_like, 'Count', target]
        ).properties(title=f"{region_like} vs {target} (Counts)"), use_container_width=True)

    # 3 Scatter: premium vs age
    prem = find_col_contains(["annualpremium","monthlypremium","premium"])
    age = find_col_contains(["age"])
    if prem and age:
        st.altair_chart(alt.Chart(dff).mark_circle(opacity=0.6).encode(
            x=f'{age}:Q', y=f'{prem}:Q', color=f'{target}:N', tooltip=[target, age, prem]
        ).properties(title=f"{prem} vs {age} by {target}"), use_container_width=True)

    # 4 Line: tenure vs rate
    tenure = find_col_contains(["tenure","duration","policetenure","customertenure","yearswithcompany"])
    if tenure:
        cohort = dff.groupby(tenure)[target].apply(lambda s: (s.astype(str).str.lower().isin(['active','yes','1','true','approved','open'])).mean()).reset_index(name='PositiveRate')
        st.altair_chart(alt.Chart(cohort).mark_line(point=True).encode(
            x=f'{tenure}:Q', y=alt.Y('PositiveRate:Q', axis=alt.Axis(format='%')), tooltip=[tenure,'PositiveRate']
        ).properties(title=f"{target} Positive Rate by {tenure}"), use_container_width=True)

    # 5 Heatmap: coverage vs payment
    def get_col(any_list):
        for c in dff.columns:
            lc = c.lower()
            if any(s in lc for s in any_list):
                return c
        return None
    coverage = get_col(["coverage","plan"])
    payment = get_col(["paymentfrequency","paymentmode","billingfrequency","billing"])
    if coverage and payment:
        tmp = dff.copy()
        tmp['is_pos'] = tmp[target].astype(str).str.lower().isin(['active','yes','1','true','approved','open']).astype(int)
        grid = tmp.groupby([coverage, payment])['is_pos'].mean().reset_index()
        st.altair_chart(alt.Chart(grid).mark_rect().encode(
            x=f'{coverage}:N', y=f'{payment}:N',
            color=alt.Color('is_pos:Q', scale=alt.Scale(scheme="greens"), legend=alt.Legend(title='Positive Rate')),
            tooltip=[coverage, payment, alt.Tooltip('is_pos:Q', format='.1%')]
        ).properties(title=f"Heatmap: {coverage} × {payment} → {target} rate"), use_container_width=True)

def metrics_tab(df, target):
    st.subheader("🧠 Train Models & Metrics")
    run = st.button("Run 5-Fold CV + Train All Models")
    if not run:
        st.info("Click to train Decision Tree, Random Forest, and Gradient Boosted and view metrics/plots.")
        return
    metrics_df, conf_figs, roc_fig, feat_imps = run_training(df, target)
    st.markdown("### Metrics Table")
    st.dataframe(metrics_df)
    st.markdown("### Confusion Matrices")
    col_tr, col_te = st.columns(2)
    for split, name, fig in conf_figs:
        if split == "train": col_tr.pyplot(fig)
        else: col_te.pyplot(fig)
    st.markdown("### ROC Curves (All Models)")
    st.pyplot(roc_fig)
    st.markdown("### Feature Importances (Top 15)")
    for name, fi in feat_imps.items():
        st.markdown(f"**{name}**")
        st.bar_chart(fi.set_index('Feature')['Importance'])

def predict_tab(df, target):
    st.subheader("📥 Upload New Dataset to Predict Policy Status")
    st.caption("Upload a CSV with the same schema (target optional). Nulls are handled automatically.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None: return
    new_df = pd.read_csv(uploaded)
    X, y = split_xy(df, target)
    preprocess, _, _ = build_preprocess(X)
    model = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
    pipe.fit(X, y)
    if hasattr(pipe.named_steps['model'], "predict_proba"):
        proba = pipe.predict_proba(new_df)[:,1]
    elif hasattr(pipe.named_steps['model'], "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        proba = MinMaxScaler().fit_transform(pipe.decision_function(new_df).reshape(-1,1)).ravel()
    else:
        proba = None
    preds = pipe.predict(new_df)
    out = new_df.copy()
    out['PredictedPolicyStatus'] = preds
    if proba is not None: out['PolicyStatusProbability'] = proba
    st.markdown("### Preview")
    st.dataframe(out.head(50))
    st.download_button("⬇️ Download Predictions CSV", data=out.to_csv(index=False).encode('utf-8'), file_name="policy_status_predictions.csv", mime="text/csv")

def main():
    st.title("Insurance — Policy Status Analytics & Prediction")
    st.write("Upload **Insurance.csv** below or keep it next to `app.py` in the repo.")
    data_src = st.file_uploader("Upload Insurance.csv (optional if present locally)", type=["csv"])
    if data_src is not None:
        df = pd.read_csv(data_src)
    else:
        try:
            df = load_data("Insurance.csv")
        except Exception:
            st.warning("Insurance.csv not found. Please upload the dataset above."); return

    target = detect_target(df)
    if target is None:
        # if user named exactly "policy status"
        for c in df.columns:
            if c.lower() == "policy status":
                target = c
                break
    if target is None:
        st.error("Dataset must include a 'policy status' (or PolicyStatus) column."); return

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧪 Models & Metrics", "🔮 Predict New Data"])
    with tab1: charts_section(df, target)
    with tab2: metrics_tab(df, target)
    with tab3: predict_tab(df, target)

if __name__ == "__main__":
    main()
