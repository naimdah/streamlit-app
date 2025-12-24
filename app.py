import streamlit as st
import pandas as pd
import requests
import re
import os
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from datetime import timezone as ZoneInfo
from mistralai import Mistral

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Eulerian Analyst Pro", page_icon="📊", layout="wide")

# ⚠️ VOS CLÉS
MISTRAL_API_KEY = "4RYLP7nnLh8BsoCRaaHA4pryfyLgIaxt" 
EULERIAN_TOKEN = "PC3qOHIpm.Vp0nKozLS_XfY50IDyOEOXb8EFi07MAsFPmw--"
EULERIAN_USER_TOKEN = "QV.tEbjPNGLjgHPax6VWH_oe8sU7fhKeoQV7eaAVymv2erwLJQ--"
EULERIAN_HOST = "sncfc.api.eulerian.com"
EULERIAN_SITE = "sncf-connect"

# ==========================================
# 2. MOTEUR DE DONNÉES (FIABLE 100%)
# ==========================================
@st.cache_data(ttl=3600)
def get_data_v3():
    # CORRECTION : On utilise UNIQUEMENT les codes techniques ici pour que l'API comprenne
    # (Nom technique, Nom technique) -> On ne met pas de français ici !
    METRICS = [
        ("visit", "visit"),
        ("dvisitor", "dvisitor"),
        ("hit", "hit"),
        ("realscartvalidamount", "realscartvalidamount"),
        ("realscartvalid", "realscartvalid")
    ]
    
    metrics_payload = [{"name": name, "field": field} for name, field in METRICS]
    
    url = f"https://{EULERIAN_HOST}/ea/v2/ea/{EULERIAN_SITE}/report/batch/query.json"
    headers = {
        "Authorization": f"Bearer {EULERIAN_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "async": False,
        "userToken": EULERIAN_USER_TOKEN,
        "reports": [{
            "kind": "rt#insummary",
            "path": "mcMEDIAINCOMING[?].*",
            "dimensions": [{"name": "media_key", "field": "media_key"}],
            "metrics": metrics_payload,
            "dateRanges": [{"range": "LAST_7_DAYS"}],
            "dateScale": "D",
            "dateRangeSplitPerScale": True,
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
        
        # Gestion des dates
        date_epochs = [v["epoch"] for v in report["columnHeader"]["dateRanges"][0]["values"]]
        try:
            dates = [datetime.fromtimestamp(e, tz=ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d") for e in date_epochs]
        except:
            dates = [datetime.fromtimestamp(e).strftime("%Y-%m-%d") for e in date_epochs]

        metrics_names = [m["name"] for m in report["columnHeader"]["metrics"]]

        if "data" in report:
            for row in report["data"]:
                media_key = row["dimensions"][0]
                for i, date_str in enumerate(dates):
                    metrics_values = []
                    for m in row["metrics"]:
                        # Correction : gestion sécurisée des valeurs
                        if m and len(m) > 0:
                            if "values" in m[0] and i < len(m[0]["values"]):
                                val = m[0]["values"][i]
                            elif "value" in m[0]:
                                val = m[0]["value"]
                            else:
                                val = 0
                        else:
                            val = 0
                        metrics_values.append(float(val))

                    row_dict = {
                        "Date": date_str,
                        "Levier": media_key,
                        **dict(zip(metrics_names, metrics_values))
                    }
                    rows.append(row_dict)

        df = pd.DataFrame(rows)
        
        if not df.empty:
            # C'EST ICI QU'ON RENOMME EN FRANÇAIS (CA -> VA)
            df = df.rename(columns={
                "visit": "Visites",
                "realscartvalidamount": "VA", # On renomme le champ technique en VA
                "realscartvalid": "Commandes",
                "media_key": "Levier"
            })
            
            # Nettoyage et Calculs
            if "VA" in df.columns: df["VA"] = df["VA"].astype(float)
            if "Commandes" in df.columns: df["Commandes"] = df["Commandes"].astype(int)
            if "Visites" in df.columns: df["Visites"] = df["Visites"].astype(int)
            
            # KPI Métier
            df["Tx_Conv"] = df["Commandes"] / df["Visites"].replace(0, 1)
            df["Panier_Moyen"] = df["VA"] / df["Commandes"].replace(0, 1)
            
            return df

    except Exception as e:
        st.error(f"Erreur technique : {e}")
        return pd.DataFrame()

# ==========================================
# 3. CERVEAU D'ANALYSE HYBRIDE (PYTHON + MISTRAL)
# ==========================================

def analyser_contexte(question, df):
    """ Filtre le tableau selon la question (Date / Levier) """
    df_filtered = df.copy()
    filtres_appliques = []
    q_lower = question.lower()

    # 1. Filtre Levier (Recherche textuelle)
    tous_leviers = df["Levier"].unique()
    leviers_trouves = []
    for levier in tous_leviers:
        if str(levier).lower() in q_lower:
            leviers_trouves.append(levier)
    if leviers_trouves:
        df_filtered = df_filtered[df_filtered["Levier"].isin(leviers_trouves)]
        filtres_appliques.append(f"Levier : {', '.join(leviers_trouves)}")

    # 2. Filtre Date (Regex Avancé)
    match_date = re.search(r'(?:le|du|au)\s+([0-2]?[0-9]|3[0-1])', q_lower)
    if match_date:
        jour = match_date.group(1).zfill(2)
        df_temp = df_filtered[df_filtered["Date"].str.endswith(f"-{jour}")]
        if not df_temp.empty:
            df_filtered = df_temp
            filtres_appliques.append(f"Date : le {jour}")
    
    msg_contexte = f" ({' | '.join(filtres_appliques)})" if filtres_appliques else ""
    return df_filtered, msg_contexte

def detect_intent(question):
    q = question.lower()
    if any(k in q for k in ["top", "classement", "meilleur", "pire", "plus grand", "plus gros", "premiers", "derniers", "liste", "les 3", "les 5", "les 10"]): return "TOP"
    if any(k in q for k in ["record", "max", "pic", "jour", "quand"]): return "MAX"
    if any(k in q for k in ["taux", "conversion", "panier", "moyen", "combien", "total", "va", "ca", "visite", "commande"]): return "METRIC"
    return "ANALYSE"

def reponse_hybride(question, df, historique_chat):
    q = question.lower()
    
    # 1. FILTRAGE
    df_context, info_msg = analyser_contexte(question, df)
    
    if df_context.empty:
        return "⚠️ Je ne trouve aucune donnée correspondant à vos filtres. Vérifiez la date ou le nom du levier."

    intent = detect_intent(question)
    
    # 2. CALCULS SYSTÉMATIQUES (L'ANTISÈCHE)
    total_va = df_context["VA"].sum()
    total_vis = df_context["Visites"].sum()
    total_cmd = df_context["Commandes"].sum()
    
    contexte_tech = ""

    # --- ROUTAGE INTENTIONS ---

    # CAS 1 : CLASSEMENT (TOP / FLOP)
    if intent == "TOP":
        # Agrégation par Levier
        synthese = df_context.groupby("Levier")[["VA", "Visites", "Commandes"]].sum().reset_index()
        synthese["Tx_Conv"] = synthese["Commandes"] / synthese["Visites"].replace(0, 1)
        synthese["Panier_Moyen"] = synthese["VA"] / synthese["Commandes"].replace(0, 1)

        # Choix de la colonne de tri
        if "commande" in q: col, label = "Commandes", "Cmds"
        elif "visite" in q or "trafic" in q: col, label = "Visites", "Visites"
        elif "panier" in q: col, label = "Panier_Moyen", "€ PM"
        elif "taux" in q or "conv" in q: col, label = "Tx_Conv", "% Conv"
        else: col, label = "VA", "€" # Par défaut

        # Sens du tri (Croissant / Décroissant)
        ascending = False # Par défaut (Meilleur en premier)
        mot_ordre = "meilleurs"
        
        if any(x in q for x in ["pire", "moins", "faible", "petit", "bas", "croissant", "flop"]):
            ascending = True
            mot_ordre = "moins performants"

        # Nombre d'éléments (Top N)
        match_n = re.search(r'(?:top|classement|les)\s*(\d+)', q)
        top_n = int(match_n.group(1)) if match_n else 5 # Par défaut 5
        
        top = synthese.sort_values(col, ascending=ascending).head(top_n)
        
        contexte_tech = f"CLASSEMENT CALCULÉ PAR PYTHON ({mot_ordre} sur {col}) :\n"
        for i, r in enumerate(top.itertuples(), 1):
            val = r._asdict()[col]
            if col == "Tx_Conv": val_fmt = f"{val:.2%}"
            elif col in ["VA", "Panier_Moyen"]: val_fmt = f"{val:,.0f} €"
            else: val_fmt = f"{int(val)}"
            contexte_tech += f"{i+1}. {r.Levier} ({val_fmt} {label})\n"

    # CAS 2 : MÉTRIQUES (Total / Moyenne)
    elif intent == "METRIC":
        # Détection Moyenne
        is_avg = "moyenne" in q or "moyen" in q
        nb_jours = df_context["Date"].nunique()
        diviseur = nb_jours if (is_avg and "panier" not in q) else 1
        prefixe = "Moyenne journalière" if diviseur > 1 else "Total"
        
        if "panier" in q:
            pm = total_va / total_cmd if total_cmd > 0 else 0
            contexte_tech = f"Panier Moyen Global : {pm:.2f} €"
        elif "taux" in q:
            tc = total_cmd / total_vis if total_vis > 0 else 0
            contexte_tech = f"Taux de Conversion Global : {tc:.2%}"
        elif "commande" in q:
            val = total_cmd / diviseur
            contexte_tech = f"{prefixe} Commandes : {int(val)}"
        elif "visite" in q:
            val = total_vis / diviseur
            contexte_tech = f"{prefixe} Visites : {int(val):,}"
        else:
            val = total_va / diviseur
            contexte_tech = f"{prefixe} Volume d'Affaires (VA) : {val:,.2f} €"

    # CAS 3 : RECORD TEMPOREL
    elif intent == "MAX":
        col = "VA"
        if "commande" in q: col = "Commandes"
        elif "visite" in q: col = "Visites"
        
        idx = df_context[col].idxmax()
        row = df_context.loc[idx]
        val_fmt = f"{int(row[col])}" if col != "VA" else f"{row[col]:,.0f} €"
        contexte_tech = f"RECORD TROUVÉ PAR PYTHON : Le {row['Date']} ({row['Levier']}) avec {val_fmt} ({col})."

    # CAS 4 : ANALYSE GÉNÉRALE
    else:
        contexte_tech = f"""
        CHIFFRES CLÉS (Période complète) :
        - VA Total : {total_va:,.0f} €
        - Visites : {int(total_vis)}
        - Commandes : {int(total_cmd)}
        """

    # 3. DONNÉES DÉTAILLÉES (CSV)
    try:
        df_display = df_context.sort_values("VA", ascending=False).head(60)
        df_display["VA"] = df_display["VA"].apply(lambda x: f"{x:.0f}")
        csv_data = df_display.to_markdown(index=False)
    except:
        csv_data = df_context.head(60).to_csv(index=False, sep=";")

    # 4. APPEL MISTRAL
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    prompt = f"""
    Tu es un Data Analyst Expert pour SNCF Connect.
    
    L'utilisateur pose une question.
    Pour répondre, base-toi PRIORITAIREMENT sur la "Fiche Technique" ci-dessous car elle contient les calculs exacts faits par le système (Python).
    
    ====================
    {contexte_tech}
    ====================
    
    Détail des données (si besoin) :
    {csv_data}
    
    Question utilisateur : "{question}"
    
    Consignes : 
    1. Reformule la réponse de la section "RÉPONSE EXACTE" de manière naturelle et pro.
    2. Ne recalcule rien si la réponse est déjà dans la fiche technique.
    3. Parle de "Volume d'Affaires (VA)" et non de CA.
    """
    
    try:
        msgs = [{"role":"system", "content": prompt}]
        for m in historique_chat[-2:]:
            if "Je suis prêt" not in str(m["content"]):
                msgs.append({"role": m["role"], "content": m["content"]})
        
        resp = client.chat.complete(model="open-mistral-nemo", messages=msgs)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Erreur IA : {e}"

# ==========================================
# 4. INTERFACE UTILISATEUR "DASHBOARD"
# ==========================================
st.title("📊 Eulerian Dashboard & Analyst")
st.markdown("### Suivi des Performances Marketing (7 derniers jours)")

if st.button("🔄 Actualiser les données"):
    st.cache_data.clear()
    st.rerun()

df = get_data_v3()

if not df.empty:
    # --- A. BLOC KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    total_va = df['VA'].sum()
    total_visites = df['Visites'].sum()
    total_commandes = df['Commandes'].sum()
    global_tx = total_commandes / total_visites if total_visites > 0 else 0

    k1.metric("Visites", f"{int(total_visites):,}".replace(",", " "))
    k2.metric("Volume d'Affaires", f"{total_va:,.0f} €".replace(",", " "))
    k3.metric("Commandes", int(total_commandes))
    k4.metric("Taux de Conv.", f"{global_tx:.2%}")
    
    st.divider()

    # --- B. GRAPHIQUE & TABLEAU ---
    col_graph, col_table = st.columns([1, 2])

    with col_graph:
        st.subheader("Répartition du VA par Levier")
        df_chart = df.groupby("Levier")["VA"].sum().reset_index().sort_values("VA", ascending=False)
        st.bar_chart(df_chart, x="Levier", y="VA", color="#ffcd00")

    with col_table:
        st.subheader("Détail des données")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Date": st.column_config.TextColumn("Date"),
                "Levier": st.column_config.TextColumn("Levier"),
                "Visites": st.column_config.NumberColumn("Visites", format="%d"),
                "VA": st.column_config.NumberColumn("Volume d'Affaires", format="%.2f €"),
                "Commandes": st.column_config.NumberColumn("Cmds"),
                "Tx_Conv": st.column_config.ProgressColumn("Taux Conv.", format="%.2f%%", min_value=0, max_value=max(df["Tx_Conv"].max(), 0.05)),
                "Panier_Moyen": st.column_config.NumberColumn("Panier Moyen", format="%.2f €"),
            },
            hide_index=True
        )

    st.divider()

    # --- C. CHATBOT INTELLIGENT ---
    st.subheader("🤖 Assistant IA")
    st.info("Exemples : 'Top 3 VA', 'Moyenne des visites', 'Combien de VA pour le Mailing le 21 ?'")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Je suis prêt à analyser ce tableau pour vous."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Votre question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                rep = reponse_hybride(user_input, df, st.session_state.messages)
                st.markdown(rep)
                st.session_state.messages.append({"role": "assistant", "content": rep})
else:
    st.warning("⚠️ Pas de données. Vérifiez la connexion API.")