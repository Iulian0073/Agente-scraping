import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# IMPORT STABILI PER LANGCHAIN 0.2.1 (Versione Cloud)
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==========================================
# 1. SETUP E CONFIGURAZIONE UI
# ==========================================
load_dotenv()

st.set_page_config(page_title="AI Bando Advisor", page_icon="🏢", layout="centered")

# ==========================================
# 2. SISTEMA DI AUTENTICAZIONE
# ==========================================
PASSWORD_ACCESSO = os.environ.get("APP_PASSWORD", "admin123")

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
# 3. TOOL & AGENTE (Architettura Stabile)
# ==========================================
@tool
def interroga_database_bando(categoria: str = None, punteggio_minimo: float = 0):
    """Cerca le aziende nel database cloud di Google Sheets in tempo reale."""
    sheet_url = os.environ.get("GOOGLE_SHEET_URL")
    if not sheet_url:
        return "Errore Critico: Manca l'URL del database cloud."
        
    try:
        df = pd.read_csv(sheet_url)
        df.columns = df.columns.str.strip()
        
        if categoria:
            df = df[df['Categoria'].str.lower() == categoria.lower()]
        if punteggio_minimo > 0:
            df = df[df['Punteggio_Totale'] >= punteggio_minimo]
            
        if df.empty:
            return "Nessun risultato trovato nel database cloud."
            
        return df[['Azienda', 'Categoria', 'Dipendenti', 'Investimento_Necessario', 'Punteggio_Totale']].to_string(index=False)
        
    except pd.errors.EmptyDataError:
        return "Errore Dati: Il foglio Google è vuoto."
    except Exception as e:
        return f"Fallimento della connessione a Google Sheets: {str(e)}"

@st.cache_resource
def inizializza_agente():
    """Istanzia l'agente usando l'architettura AgentExecutor consolidata."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = [interroga_database_bando]
    
    # Prompt strutturato richiesto dalla versione 0.2.1
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei il Financial Advisor dell'azienda. Rispondi in italiano professionale. Usa sempre il tool a disposizione per interrogare i dati dei bandi."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agente = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agente, tools=tools, verbose=False)

agente_ia = inizializza_agente()

# ==========================================
# 4. INTERFACCIA CHAT
# ==========================================
st.title("🏢 Enterprise AI - Bando Advisor")
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
    
    with st.chat_message("assistant"):
        with st.spinner("Analisi database in corso..."):
            try:
                # Invocazione compatibile con AgentExecutor
                risposta = agente_ia.invoke({
                    "input": domanda,
                    "chat_history": st.session_state["cronologia_chat"]
                })
                
                messaggio_finale = risposta["output"]
                st.markdown(messaggio_finale)
                
                # Salvataggio in memoria post-esecuzione
                st.session_state["cronologia_chat"].append(HumanMessage(content=domanda))
                st.session_state["cronologia_chat"].append(AIMessage(content=messaggio_finale))
            except Exception as e:
                st.error(f"Fallimento del servizio IA: {e}")
