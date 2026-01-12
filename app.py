import streamlit as st
from streamlit_local_storage import LocalStorage
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from typing import List, Dict, Any
import concurrent.futures
from pydantic import SecretStr

import uuid

import yfinance as yf
from tradingview_ta import TA_Handler
from ddgs import DDGS
from simpleeval import simple_eval
import requests

from datetime import datetime, timezone, timedelta
from langchain.messages import HumanMessage  #, AIMessage as LCHMessage

localStorage = LocalStorage()

# Store checkpointer in session state to persist across reruns


# =========================
# SESSION STATE
# =========================
if "response" not in st.session_state:
    st.session_state.response = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "base_url" not in st.session_state:
    st.session_state.base_url = localStorage.getItem("base_url") or "https://api.openai.com/v1"
if "model" not in st.session_state:
    st.session_state.model = localStorage.getItem("model") or "gpt-4o-mini"
if "exchange_rate" not in st.session_state:
    st.session_state.exchange_rate = ""
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"stock_chat_{uuid.uuid4().hex}"
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Settings")
    model = st.text_input("Model", value=st.session_state.model, autocomplete="off")
    api_key = st.text_input("API Key", type="password",  autocomplete="off")
    base_url = st.text_input("Base URL", value=st.session_state.base_url, autocomplete="off")

    if st.button("Save Credentials",icon="📌",width="stretch"):
        st.session_state.model = model 
        st.session_state.api_key = api_key 
        st.session_state.base_url = base_url 

        st.success("Credentials saved to sessions storage")

    if st.session_state.base_url and st.session_state.model:
        localStorage.setItem("base_url", st.session_state.base_url, key="set_base_url")
        localStorage.setItem("model", st.session_state.model, key="set_model")

    st.divider()
    if st.button("New Chat", icon="✨", width="stretch"):
        st.session_state.response = []
        st.session_state.thread_id = f"stock_chat_{uuid.uuid4().hex}"
        st.rerun()


# =========================
# UI
# =========================
st.title("💬 FinSight AI")
st.caption("NSE Equity Research Agent")

with st.expander("ℹ️ About, Disclaimer & Privacy"):
    st.markdown("""
### 🤖 What this AI Agent does
FinSight AI is an institutional-grade Equity Research Analyst specializing in **NSE-listed Indian equities**. It automates data retrieval and analysis to provide concise, fact-based insights.

### 🛠️ Tools Available
- **Stock Data Tool**: Fetches real-time fundamental (valuation, margins) and technical (RSI, MACD) data.
- **News Tool**: Searches for recent market news and sentiment.
- **Calculator**: Performs precise arithmetic for financial metrics.

### 💡 How & When to Use
Use this agent when you need a quick, data-backed overview of an Indian NSE stock.

---
### ⚠️ Disclaimer
- **Model Dependency**: The accuracy of the analysis depends on the AI model you select in the settings.
- **Not Financial Advice**: This tool is for informational purposes only. **Do not use this for financial decision-making.**

### 🔒 Privacy
- **Serverless Frontend**: This application runs as a frontend interface.
- **Local Storage**: Your API Key, Base URL, and Model preferences are stored in your browser's **LocalStorage** for convenience.
""")

if not all([st.session_state.api_key, st.session_state.base_url, st.session_state.model]):
    st.warning("Add API credentials in the sidebar.")
    st.stop()

for msg in st.session_state.response:
    with st.chat_message(msg["content"]["role"]):
        st.markdown(msg["content"]["content"])
        if len(msg["tool_calls"]) > 0:
            with st.expander("Tool Used."):
                for tool_call in msg["tool_calls"]:
                    st.write(f"Tool Name: {tool_call['name']}")
                    st.write(tool_call["args"])
        if msg["tolen_usage"] != 0:
            st.markdown(f":small[Token usage: {msg['tolen_usage']}]")


# =========================
# HELPERS
# =========================
def normalize_symbol(symbol: str) -> str:
    return symbol.upper().replace(".NS", "").strip()

def usd_inr():
    url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.min.json"
    try:
        r = requests.get(url)
        r.raise_for_status()
        conversion = r.json()
        date = conversion.get("date", "2026-01-11")
        rate = conversion["usd"]["inr"]
        return f"1 USD = {rate} INR as of date {date}"
    except:
        return "1 USD = 90.27 INR as of date 2026-01-11"


def threaded_executor(func, items, max_workers=3):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({"error": str(e)})
    return results


# =========================
# DATA FETCHING (CACHED)
# =========================
@st.cache_data(ttl=1800, show_spinner=False)  # 30 min
def fetch_fundamentals(symbol: str) -> Dict[str, Any]:
    symbol = normalize_symbol(symbol)
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        if not isinstance(info, dict):
            return {}

        metrics = {}

        keys = [
            "marketCap", "bookValue", "trailingPE", "forwardPE",
            "dividendYield", "fiveYearAvgDividendYield",
            "debtToEquity", "currentRatio",
            "returnOnEquity", "returnOnAssets",
            "freeCashflow", "revenueGrowth", "earningsGrowth",
            "profitMargins", "operatingMargins",
            "enterpriseToEbitda", "payoutRatio"
        ]

        for k in keys:
            if isinstance(info.get(k), (int, float)):
                metrics[k] = info[k]

        summary = info.get("longBusinessSummary")
        if isinstance(summary, str) and len(summary) > 300:
            summary = summary[:300] + "..."

        return {
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "summary": summary,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "metrics": metrics
        }
    except Exception:
        return {}


def fetch_technicals(symbol: str) -> Dict[str, float]:
    symbol = normalize_symbol(symbol)
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange="NSE",
            screener="india",
            interval="1d",
            timeout=10
        )
        analysis = handler.get_analysis()
        indicators = analysis.indicators if analysis else {}

        keys = ["close",
            "RSI", "ADX", "ADX+DI", "ADX-DI", "Mom", "Stoch.K", "Stoch.D","BB.Upper", "BB.Lower",
            "MACD.macd", "MACD.signal", "EMA20", "EMA50", "EMA100", "EMA200",
        ]

        return {
            k: round(float(indicators[k]), 2)
            for k in keys
            if isinstance(indicators.get(k), (int, float))
        }
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)  # 10 min
def stock_snapshot(symbol: str) -> Dict[str, Any]:
    symbol = normalize_symbol(symbol)
    fund = fetch_fundamentals(symbol)
    tech = fetch_technicals(symbol)

    result = {
        "symbol": symbol,
        "company name": fund.get("name"),
        "sector": fund.get("sector"),
        "industry": fund.get("industry"),
        "businessSummary": fund.get("summary"),
        "price": fund.get("price") or tech.get("close"),
        "fundamental": fund.get("metrics", {}),
        "technical": tech
    }

    if not fund:
        result["error"] = "Fundamental data unavailable"
    if not tech:
        result["warning"] = "Technical indicators unavailable"

    return result


def get_stocks_data_tool(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches fundamental and technical data for a list of NSE stock symbols.
    
    Args:
        symbols: List of stock symbols (e.g. ['RELIANCE', 'TCS']).
        
    Returns:
        List of dictionaries containing price, valuation metrics, and technical indicators.
    """
    symbols = [normalize_symbol(s) for s in symbols if isinstance(s, str)]
    return threaded_executor(stock_snapshot, symbols)


@st.cache_data(ttl=900, show_spinner=False)  # 15 min
def web_search_tool(query: str) -> Dict[str, Any]:
    """
    Performs a real-time web-search for a query.
    
    Args:
        query: Search query string.
        
    Returns:
        Dictionary containing search results on success or error on failure.
    """
    try:
        results = DDGS(timeout=5).text(
            query,
            region="in-en",
            max_results=10,
            safesearch="on",
            timelimit="w"
        )
        if not results:
            return {"error": "No news found"}

        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


def calculator_tool(expression):
    """
    Evaluate a mathematical expression safely using `simple_eval`.
        +   : addition
        -   : subtraction
        *   : multiplication
        /   : division (float)
        //  : floor division (integer)
        %   : modulus (remainder)
        **  : exponentiation (power)

    Args:
        expression : str
            A string containing a mathematical expression to evaluate.
            Examples:
                - "22 * 43"
                - "(3.55 + 7.34) / 6"
                - "56 // 3"
                - "2 ** 5"
                - "10 % 3"

    Returns
        dict
            On success:
                {"result": <evaluated_value>}
            On failure:
                {"error": <error_message>}
    """
    try:
        result = simple_eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# =========================
# SYSTEM PROMPT
# =========================

if not st.session_state.exchange_rate:
    st.session_state.exchange_rate = usd_inr()

SYSTEM_PROMPT = f"""
You are **FinSight AI**, an institutional-grade Equity Research Analyst specializing in **NSE-listed Indian equities**.

Today's Date: {(datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")}
{st.session_state.exchange_rate}

### Rules
- Always call **get_stocks_data_tool** before discussing price, valuation, fundamentals, or technical indicators.  
  - Use Indian currency (INR) for price, market cap, and financial figures unless explicitly requested otherwise.
- Use **web_search_tool** for recent events, corporate actions, regulatory updates, macro developments, or market sentiment.  
  - When using web_search_tool, always include: **source title, publisher/domain, and URL**.
- For any arithmetic or numeric calculation, always call **calculator_tool**. Do not calculate manually.

### Style
- Maintain a **neutral, institutional tone** suitable for professional equity research.  
- Highlight **tickers** and **key metrics** in bold for clarity.  
- Present insights using **bullet points** and **short paragraphs** only.  
- Do **not** use tables, charts, or complex Markdown formatting.

### Output
- Provide concise, fact-based analysis.  
- Focus on clarity, accuracy, and professional readability.
"""



# =========================
# AGENT
# =========================
def stock_agent(model: str, base_url: str, api_key: str, thread_id: str, message: str) -> Dict[str, Any]:
    try:
        llm = ChatOpenAI(
            model=model.strip(),
            base_url=base_url.strip(),
            api_key=SecretStr(api_key.strip()),
            temperature=0.0
        )

        agent = create_agent(
            model=llm,
            tools=[get_stocks_data_tool, web_search_tool, calculator_tool],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=st.session_state.checkpointer
        )

        messages = []
        messages.append(HumanMessage(content=message))

        response = agent.invoke({"messages": messages}, config={"configurable": {"thread_id": thread_id}})

        content = response["messages"][-1].content.strip()

        total_usage = 0
        all_tool_calls = []
        for message in response["messages"]:
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    for tool_call_item in message.tool_calls:
                        all_tool_calls.append({
                            "name": tool_call_item["name"],
                            "args": tool_call_item["args"]
                        })
                if message.usage_metadata:
                    total_usage += message.usage_metadata.get('total_tokens', 0)



        return {
            "content": content,
            "tool_calls": all_tool_calls,
            "tolen_usage": total_usage
        }

    except Exception as e:
        return {"error": f"Unable to generate ai response due to - {str(e)}"}


# =========================
# CHAT INPUT
# =========================
if prompt := st.chat_input("Ask about NSE stocks..."):
    st.session_state.response.append({
        "content": {"role": "user", "content": prompt},
        "tool_calls": [],
        "tolen_usage": 0
    })
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = stock_agent(
                model=st.session_state.model,
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url,
                thread_id=st.session_state.thread_id,
                message=prompt
            )

        if "error" in result:
            st.error(result["error"])
            if st.session_state.response and st.session_state.response[-1]["content"]["role"] == "user":
                st.session_state.response.pop()
        else:
            st.markdown(result["content"])
            if result["tool_calls"]:
                with st.expander("Tool Used."):
                    for tool_call in result["tool_calls"]:
                        st.write(f"Tool Name: {tool_call['name']}")
                        st.write(tool_call["args"])
            if result["tolen_usage"] != 0:
                st.markdown(f":small[Token usage: {result['tolen_usage']}]")
            st.session_state.response.append({
                "content": {"role": "assistant", "content": result["content"]},
                "tool_calls": result["tool_calls"],
                "tolen_usage": result["tolen_usage"]
            })
