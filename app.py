import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# IMPORT CORRETTO E STABILE (Quello che ha funzionato nel terminale)
from langchain.agents import create_agent

# ==========================================
# 1. SETUP E CONFIGURAZIONE UI
# ==========================================
load_dotenv()

st.set_page_config(page_title="AI Bando Advisor", page_icon="🏢", layout="centered")

# ==========================================
# 2. SISTEMA DI AUTENTICAZIONE
# ==========================================
PASSWORD_ACCESSO = os.environ.get("APP_PASSWORD", "a")

def verifica_login():
    if "autenticato" not in st.session_state:
        st.session_state["autenticato"] = False

    if not st.session_state["autenticato"]:
        st.title("🔒 Accesso Riservato")
        password = st.text_input("Inserisci la password aziendale:", type="password")
        if st.button("Accedi"):
            if password == PASSWORD_ACCESSO:
                st.session_state["autenticato"] = True
                st.rerun()
            else:
                st.error("Credenziali non valide.")
        st.stop()

verifica_login()

# ==========================================
# 3. TOOL & AGENTE
# ==========================================
@tool
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=False # Non fa crashare Streamlit, ma restituisce l'errore come stringa all'LLM
)
def interroga_database_bando(categoria: str = None, punteggio_minimo: float = 0):
    """Cerca le aziende nel database cloud di Google Sheets in tempo reale."""
    # Dependency Injection dell'URL, nessuna hardcoding
    sheet_url = os.environ.get("GOOGLE_SHEET_URL")
    
    if not sheet_url:
        return "Errore Critico: Manca l'URL del database cloud nelle variabili d'ambiente."
        
    try:
        # I/O di Rete: Legge lo stream CSV direttamente da Google
        df = pd.read_csv(sheet_url)
        
        # Normalizzazione dello schema: rimuove eventuali spazi vuoti lasciati dall'utente su Google Sheets
        df.columns = df.columns.str.strip()
        
        if categoria:
            df = df[df['Categoria'].str.lower() == categoria.lower()]
        if punteggio_minimo > 0:
            df = df[df['Punteggio_Totale'] >= punteggio_minimo]
            
        if df.empty:
            return "Nessun risultato trovato nel database cloud."
            
        return df[['Azienda', 'Categoria', 'Dipendenti', 'Investimento', 'Punteggio_Totale']].to_string(index=False)
        
    except pd.errors.EmptyDataError:
        return "Errore Dati: Il foglio Google è vuoto."
    except Exception as e:
        # Cattura errori di parsing o di rete
        raise RuntimeError(f"Fallimento della connessione a Google Sheets: {str(e)}")

@st.cache_resource
def inizializza_agente():
    """Istanzia l'agente usando la sintassi V1 unificata."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    istruzioni = (
        "Sei il Financial Advisor dell'azienda. Rispondi in italiano professionale. "
        "Usa sempre il tool a disposizione per interrogare i dati dei bandi."
    )
    # Usiamo create_agent esattamente come nel main.py funzionante
    return create_agent(
        model=llm, 
        tools=[interroga_database_bando], 
        system_prompt=istruzioni
    )

agente_ia = inizializza_agente()

# ==========================================
# 4. INTERFACCIA CHAT
# ==========================================
st.title("Enterprise AI - Bando Advisor")
st.markdown("Interroga il database aziendale in linguaggio naturale.")
st.divider()

if "cronologia_chat" not in st.session_state:
    st.session_state["cronologia_chat"] = []

for msg in st.session_state["cronologia_chat"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if domanda := st.chat_input("Es: Trovami i ristoranti con più di 100 punti..."):
    with st.chat_message("user"):
        st.markdown(domanda)
    
    st.session_state["cronologia_chat"].append(HumanMessage(content=domanda))
    
    with st.chat_message("assistant"):
        with st.spinner("Analisi database in corso..."):
            try:
                # Passiamo l'intera cronologia per mantenere il contesto
                risposta = agente_ia.invoke({"messages": st.session_state["cronologia_chat"]})
                messaggio_finale = risposta["messages"][-1].content
                
                st.markdown(messaggio_finale)
                st.session_state["cronologia_chat"].append(AIMessage(content=messaggio_finale))
            except Exception as e:
                st.error(f"Fallimento del servizio IA: {e}")