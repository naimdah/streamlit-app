import streamlit as st
import pandas as pd
import requests
import re
import hmac
import os
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from datetime import timezone as ZoneInfo
from mistralai import Mistral

# ==========================================
# 1. CONFIGURATION & SÉCURITÉ
# ==========================================
st.set_page_config(page_title="Eulerian Analyst Pro", page_icon="🔒", layout="wide")

def check_password():
    """Vérifie le mot de passe pour accéder à l'app."""
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], "SNCF_TEAM_2024"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Accès Sécurisé - SNCF Connect")
    st.text_input("Mot de passe équipe", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("Mot de passe incorrect")
    return False

if not check_password():
    st.stop()

# --- VOS CLÉS ---
try:
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    EULERIAN_TOKEN = st.secrets["EULERIAN_TOKEN"]
    EULERIAN_USER_TOKEN = st.secrets["EULERIAN_USER_TOKEN"]
except:
    MISTRAL_API_KEY = "4RYLP7nnLh8BsoCRaaHA4pryfyLgIaxt" 
    EULERIAN_TOKEN = "PC3qOHIpm.Vp0nKozLS_XfY50IDyOEOXb8EFi07MAsFPmw--"
    EULERIAN_USER_TOKEN = "QV.tEbjPNGLjgHPax6VWH_oe8sU7fhKeoQV7eaAVymv2erwLJQ--"

EULERIAN_HOST = "sncfc.api.eulerian.com"
EULERIAN_SITE = "sncf-connect"

# ==========================================
# 2. MOTEUR DE DONNÉES (FIABLE)
# ==========================================
@st.cache_data(ttl=3600)
def get_data():
    METRICS = [
        ("visit", "Visites"),
        ("dvisitor", "dvisitor"),
        ("hit", "hit"),
        ("realscartvalidamount", "realscartvalidamount"),
        ("realscartvalid", "realscartvalid")
    ]
    metrics_payload = [{"name": name, "field": field} for name, field in METRICS]
    
    url = f"https://{EULERIAN_HOST}/ea/v2/ea/{EULERIAN_SITE}/report/batch/query.json"
    headers = {"Authorization": f"Bearer {EULERIAN_TOKEN}", "Content-Type": "application/json", "Accept": "application/json"}
    
    payload = {
        "async": False, "userToken": EULERIAN_USER_TOKEN,
        "reports": [{
            "kind": "rt#insummary", "path": "mcMEDIAINCOMING[?].*",
            "dimensions": [{"name": "media_key", "field": "media_key"}],
            "metrics": metrics_payload,
            "dateRanges": [{"range": "LAST_7_DAYS"}], "dateScale": "D", "dateRangeSplitPerScale": True,
            "segmentFilterClauses": [{"field": "attributionrule", "operator": "IN", "value": [8]}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        res = response.json()
        if response.status_code != 200 or "data" not in res:
            st.error(f"Erreur API : {res.get('error_msg', response.text)}")
            return pd.DataFrame()

        rows = []
        report = res["data"]["reports"][0]
        date_epochs = [v["epoch"] for v in report["columnHeader"]["dateRanges"][0]["values"]]
        try: dates = [datetime.fromtimestamp(e, tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d") for e in date_epochs]
        except: dates = [datetime.fromtimestamp(e).strftime("%Y-%m-%d") for e in date_epochs]
        metrics_names = [m["name"] for m in report["columnHeader"]["metrics"]]

        if "data" in report:
            for row in report["data"]:
                media_key = row["dimensions"][0]
                for i, date_str in enumerate(dates):
                    metrics_values = []
                    for m in row["metrics"]:
                        val = 0
                        if m and len(m) > 0:
                            if "values" in m[0] and i < len(m[0]["values"]):
                                val = m[0]["values"][i]
                            elif "value" in m[0]:
                                val = m[0]["value"]
                        metrics_values.append(float(val))
                    rows.append({"Date": date_str, "Levier": media_key, **dict(zip(metrics_names, metrics_values))})

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.rename(columns={
                "visit": "Visites",
                "realscartvalidamount": "VA",
                "realscartvalid": "Commandes",
                "media_key": "Levier"
            })
            df["VA"] = df["VA"].astype(float)
            df["Commandes"] = df["Commandes"].astype(int)
            df["Visites"] = df["Visites"].astype(int)
            df["Tx_Conv"] = df["Commandes"] / df["Visites"].replace(0, 1)
            df["Panier_Moyen"] = df["VA"] / df["Commandes"].replace(0, 1)
            return df
    except Exception as e:
        st.error(f"Erreur technique : {e}")
        return pd.DataFrame()
    return pd.DataFrame()

# ==========================================
# 3. CERVEAU INTELLIGENT
# ==========================================
def analyser_contexte(question, df):
    df_filtered = df.copy()
    filtres = []
    q = question.lower()

    for lev in df["Levier"].unique():
        if str(lev).lower() in q:
            df_filtered = df_filtered[df_filtered["Levier"] == lev]
            filtres.append(f"Levier : {lev}")
            break

    match = re.search(r'\b([0-2]?[0-9]|3[0-1])\b', q)
    if match and not any(k in q for k in ["top", "les"]): 
        jour = match.group(1).zfill(2)
        df_temp = df_filtered[df_filtered["Date"].str.endswith(f"-{jour}")]
        if not df_temp.empty:
            df_filtered = df_temp
            filtres.append(f"Date : le {jour}")

    msg = f" ({' | '.join(filtres)})" if filtres else ""
    return df_filtered, msg

def detect_intent(question):
    q = question.lower()
    if any(k in q for k in ["top", "classement", "meilleur", "pire"]): return "TOP"
    if any(k in q for k in ["record", "max", "pic", "jour", "quand"]): return "MAX"
    if any(k in q for k in ["taux", "conversion", "panier", "moyen", "combien", "total", "va", "visite"]): return "METRIC"
    return "ANALYSE"

def reponse_hybride(question, df, historique):
    q = question.lower()
    df_context, info_msg = analyser_contexte(question, df)
    
    if df_context.empty: return "⚠️ Pas de données avec ces filtres."
    
    intent = detect_intent(question)
    tech_context = ""

    if intent == "METRIC":
        va = df_context["VA"].sum()
        vis = df_context["Visites"].sum()
        cmd = df_context["Commandes"].sum()
        if "panier" in q: tech_context = f"Panier Moyen {info_msg}: {(va/cmd if cmd else 0):.2f} €"
        elif "taux" in q: tech_context = f"Taux Conv {info_msg}: {(cmd/vis if vis else 0):.2%}"
        elif "visite" in q: tech_context = f"Total Visites {info_msg}: {int(vis):,}"
        else: tech_context = f"Total VA {info_msg}: {va:,.2f} €"

    elif intent == "TOP":
        grp = df_context.groupby("Levier")[["VA", "Visites", "Commandes"]].sum().reset_index()
        col = "VA"
        if "visite" in q: col = "Visites"
        elif "commande" in q: col = "Commandes"
        
        sens = True if any(x in q for x in ["pire", "moins"]) else False
        top = grp.sort_values(col, ascending=sens).head(5)
        
        tech_context = f"CLASSEMENT PYTHON ({col}) :\n"
        for i, r in enumerate(top.itertuples(), 1):
            val = r._asdict()[col]
            fmt = f"{val:,.0f} €" if col=="VA" else f"{int(val)}"
            tech_context += f"{i}. {r.Levier} ({fmt})\n"

    elif intent == "MAX":
        col = "VA"
        if "visite" in q: col = "Visites"
        idx = df_context[col].idxmax()
        row = df_context.loc[idx]
        tech_context = f"RECORD {info_msg} : Le {row['Date']} ({row['Levier']}) avec {row[col]:,.0f}."

    else:
        tech_context = f"DONNÉES BRUTES :\n{df_context.head(50).to_csv(index=False, sep=';')}"

    client = Mistral(api_key=MISTRAL_API_KEY)
    prompt = f"""Tu es Data Analyst expert.
    Voici la RÉPONSE CALCULÉE EXACTE (Utilise-la impérativement) :
    ----------------
    {tech_context}
    ----------------
    Question : "{question}"
    Consigne : Fais une phrase naturelle. Ne recalcule rien. Parle de "Volume d'Affaires" (VA).
    """
    
    try:
        msgs = [{"role":"system", "content": prompt}]
        for m in historique[-2:]:
            if "Je suis prêt" not in str(m["content"]): msgs.append({"role":m["role"], "content":m["content"]})
        return client.chat.complete(model="open-mistral-nemo", messages=msgs).choices[0].message.content
    except Exception as e: return f"Erreur IA : {e}"

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.title("📊 Eulerian Dashboard & Analyst")

if st.button("🔄 Actualiser"):
    st.cache_data.clear()
    st.rerun()

df = get_data()

if not df.empty:
    k1, k2, k3, k4 = st.columns(4)
    total_va = df['VA'].sum()
    k1.metric("Visites", f"{int(df['Visites'].sum()):,}".replace(",", " "))
    k2.metric("Volume d'Affaires", f"{total_va:,.0f} €".replace(",", " "))
    k3.metric("Commandes", int(df['Commandes'].sum()))
    k4.metric("Taux Conv.", f"{(df['Commandes'].sum()/df['Visites'].sum()):.2%}")
    
    st.divider()

    # SECTION DÉTAIL RESTAURÉE
    with st.expander("🔎 Voir toutes les données brutes (Jours x Leviers)", expanded=False):
        st.dataframe(df, use_container_width=True)

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Tendances")
        metrique = st.radio("Courbe :", ["VA", "Visites", "Commandes"], horizontal=True)
        chart = df.pivot_table(index="Date", columns="Levier", values=metrique, aggfunc="sum").fillna(0)
        st.line_chart(chart)
    
    with c2:
        st.subheader("Détail (Cumul)")
        st.dataframe(
            df.groupby("Levier")[["VA", "Visites"]].sum().sort_values("VA", ascending=False),
            use_container_width=True,
            column_config={"VA": st.column_config.NumberColumn(format="%.0f €")}
        )

    st.divider()
    st.subheader("🤖 Assistant IA")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if q := st.chat_input("Posez une question..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Analyse..."):
                rep = reponse_hybride(q, df, st.session_state.messages)
                st.markdown(rep)
                st.session_state.messages.append({"role": "assistant", "content": rep})
else:
    st.warning("⚠️ Pas de données.")