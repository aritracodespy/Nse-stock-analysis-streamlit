import streamlit as st
from streamlit_local_storage import LocalStorage
from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from typing import List, Dict, Any
import concurrent.futures
from pydantic import SecretStr

import yfinance as yf
from tradingview_ta import TA_Handler
from ddgs import DDGS
from simpleeval import simple_eval
import requests

from datetime import datetime, timezone, timedelta

localStorage = LocalStorage()



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



# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Settings")
    if st.button("New Chat", icon="✨", width="stretch"):
        st.session_state.response = []
        st.rerun()
    st.divider()

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





# =========================
# UI
# =========================
st.title("💬 FinBuddy")
st.caption("Your Friendly AI Stock Market Assistant")

with st.expander("ℹ️ About, Disclaimer & Privacy"):
    st.markdown("""
### 🤖 What this AI Agent does
FinBuddy is a friendly and chill AI-powered stock market assistant specializing in **NSE-listed Indian equities**. It helps you understand stocks through fundamentals, technical indicators, and recent market developments in a simple, unbiased, and data-driven way.

### 🛠️ Tools Available
- **Stock Data Tool**: Fetches real-time fundamental (valuation, margins) and technical (RSI, MACD) data.
- **Web Search Tool**: Searches the web real-time.
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

def safe_trim(text: str, limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[:limit] + ("..." if len(text) > limit else "")

# =========================
# DATA FETCHING (CACHED)
# =========================

def build_llm_stock_payload(info: dict | None, tv: dict | None) -> dict:
    payload = {}

    if info:
        payload["company overview"] = {
            "company": {
                "name": info.get("longName"),
                "symbol": info.get("symbol"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "business_summary": (
                    safe_trim(info.get("longBusinessSummary", ""),300)
                ),
            },
        }
        payload["fundamentals"] = {
            "market_snapshot": {
                "current_price": info.get("currentPrice"),
                "previous_close": info.get("previousClose"),
                "day_range": {
                    "low": info.get("dayLow"),
                    "high": info.get("dayHigh"),
                },
                "52_week_range": {
                    "low": info.get("fiftyTwoWeekLow"),
                    "high": info.get("fiftyTwoWeekHigh"),
                },
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta"),
                "volume": info.get("volume"),
                "avg_volume_3m": info.get("averageDailyVolume3Month"),
            },

            "valuation": {
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("trailingPegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "enterprise_value": info.get("enterpriseValue"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
            },

            "profitability": {
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "ebitda_margin": info.get("ebitdaMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
            },

            "growth": {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "quarterly_earnings_growth": info.get("earningsQuarterlyGrowth"),
                "eps_ttm": info.get("epsTrailingTwelveMonths"),
                "eps_forward": info.get("epsForward"),
            },

            "financial_health": {
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "free_cash_flow": info.get("freeCashflow"),
                "operating_cash_flow": info.get("operatingCashflow"),
            },

            "dividends": {
                "dividend_rate": info.get("dividendRate"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "five_year_avg_yield": info.get("fiveYearAvgDividendYield"),
                "last_dividend": info.get("lastDividendValue"),
            },

            "ownership_governance": {
                "insider_holding_pct": info.get("heldPercentInsiders"),
                "institutional_holding_pct": info.get("heldPercentInstitutions"),
                "overall_risk": info.get("overallRisk"),
                "board_risk": info.get("boardRisk"),
                "compensation_risk": info.get("compensationRisk"),
                "shareholder_rights_risk": info.get("shareHolderRightsRisk"),
            },

            "analyst_sentiment": {
                "recommendation": info.get("recommendationKey"),
                "recommendation_mean": info.get("recommendationMean"),
                "analyst_count": info.get("numberOfAnalystOpinions"),
                "target_price": {
                    "low": info.get("targetLowPrice"),
                    "mean": info.get("targetMeanPrice"),
                    "high": info.get("targetHighPrice"),
                },
            },
        }

    if tv:
        payload["technicals"] = {
            "market_bias": {
                "recommend_all": tv.get("Recommend.All"),
                "recommend_ma": tv.get("Recommend.MA"),
                "recommend_other": tv.get("Recommend.Other"),
            },

            "momentum": {
                "rsi": tv.get("RSI"),
                "stochastic_k": tv.get("Stoch.K"),
                "williams_r": tv.get("W.R"),
                "cci": tv.get("CCI20"),
            },

            "trend_strength": {
                "adx": tv.get("ADX"),
                "plus_di": tv.get("ADX+DI"),
                "minus_di": tv.get("ADX-DI"),
            },

            "trend_direction": {
                "price": tv.get("close"),
                "ema_20": tv.get("EMA20"),
                "ema_50": tv.get("EMA50"),
                "ema_200": tv.get("EMA200"),
            },

            "momentum_shift": {
                "macd": tv.get("MACD.macd"),
                "signal": tv.get("MACD.signal"),
            },

            "volatility": {
                "bollinger_lower": tv.get("BB.lower"),
                "bollinger_upper": tv.get("BB.upper"),
            },

            "key_levels": {
                "support": tv.get("Pivot.M.Fibonacci.S1"),
                "pivot": tv.get("Pivot.M.Fibonacci.Middle"),
                "resistance": tv.get("Pivot.M.Fibonacci.R1"),
            },
        }

    return payload

@st.cache_data(ttl=900, show_spinner=False)
def stock_snapshot(symbol: str) -> dict:
    clean_symbol = symbol.strip().upper().replace(".NS", "")
    full_symbol = f"{clean_symbol}.NS"

    fundamentals = None
    technicals = None
    errors = {"fundamentals": {}, "technicals": {}}

    # --- Yahoo Finance ---
    try:
        info = yf.Ticker(full_symbol).info
        if isinstance(info, dict) and info:
            fundamentals = info
        else:
            errors["fundamentals"] = {
                "type": "YFINANCE_NO_DATA",
                "message": "Empty response from Yahoo Finance"
            }
    except Exception as e:
        errors["fundamentals"] = {
            "type": "YFINANCE_ERROR",
            "message": str(e)
        }

    # --- TradingView ---
    try:
        handler = TA_Handler(
            symbol=clean_symbol,
            exchange="NSE",
            screener="india",
            interval="1d",
            timeout=10
        )
        analysis = handler.get_analysis()
        technicals = analysis.indicators if analysis and analysis.indicators else None

        if technicals is None:
            errors["technicals"] = {
                "type": "TRADINGVIEW_NO_DATA",
                "message": "No indicators returned"
            }

    except Exception as e:
        errors["technicals"] = {
            "type": "UNEXPECTED_TECHNICAL_ERROR",
            "message": str(e)
        }

    # --- Hard fail only if both missing ---
    if not fundamentals and not technicals:
        return {
            "ok": False,
            "symbol": clean_symbol,
            "data": None,
            "availability": {
                "fundamentals": False,
                "technicals": False
            },
            "errors": errors
        }

    payload = build_llm_stock_payload(fundamentals, technicals)

    return {
        "ok": True,
        "symbol": clean_symbol,
        "data": payload,
        "availability": {
            "fundamentals": fundamentals is not None,
            "technicals": technicals is not None
        },
        "errors": errors
    }


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


@st.cache_data(ttl=900, show_spinner=False)
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

        return {"results": [{"title": result["title"], "body": result["body"]} for result in results if result["title"] and result["body"]]}
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
You are **FinBuddy**, a friendly and chill AI-powered stock market assistant focused on **NSE-listed Indian equities**.

Your role is to help users understand stocks using fundamentals, technical indicators, and recent market developments in a simple, unbiased, and data-driven manner.

Today's Date: {(datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")}
{st.session_state.exchange_rate}

### Core Rules
- Always call **get_stocks_data_tool** before discussing price, valuation, fundamentals, or technical indicators.
- Use **recommendation metrics, market_bias, and target_price** from `get_stocks_data_tool` to provide a clear verdict (strong-buy / buy / neutral / sell / strong-sell) with price levels.
- If the tool returns an error or `ok: false`, inform the user that data could not be retrieved and ask for a valid NSE symbol. Do not proceed further.
- Use **INR** for all prices and financial figures unless stated otherwise.
- Do not repeat raw data; add interpretation and context.
- If any metric is missing or null, state **“Data not available”**. Never assume or hallucinate values.
- Use **web_search_tool** for recent news, corporate actions, regulations, macro trends, or sentiment. Always include **source title / publisher / domain**.
- If no relevant news is found, state this explicitly.
- For sector or stock recommendations, rely only on data from the **web search tool**.
- All numeric calculations must use **calculator_tool**.
- Keep your response under 250 words.

### Style & Output
- Friendly, calm, neutral, and easy to understand.
- Highlight **tickers** and **key metrics** in bold.
- Use bullet points and short paragraphs only.
- No tables, charts, or complex formatting.
- Prioritize clarity, accuracy, and concise analysis.

"""

# =========================
# AGENT
# =========================
def stock_agent(model: str, base_url: str, api_key: str, message: str) -> Dict[str, Any]:
    try:
        llm = ChatOpenAI(
            model=model.strip(),
            base_url=base_url.strip(),
            api_key=SecretStr(api_key.strip()),
            temperature=0.0
        )
    except Exception as e:
        return {"error": f"Failed to initialize language model!- {str(e)}"}
    try:
        agent = create_agent(
            model=llm,
            tools=[get_stocks_data_tool, web_search_tool, calculator_tool],
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as e:
        return {"error": f"Failed to create agent!- {str(e)}"}
    try:
        messages = []
        for msg in st.session_state.response[-4:]:
            if msg["content"]["role"] == "user":
                messages.append(HumanMessage(content=safe_trim(msg["content"]["content"], 1500)))
            elif msg["content"]["role"] == "assistant":
                messages.append(AIMessage(content=safe_trim(msg["content"]["content"], 1500)))
        messages.append(HumanMessage(content=message))

        response = agent.invoke({"messages": messages})

        content = response["messages"][-1].content.strip()

        total_usage = 0
        num_requests = 0
        all_tool_calls = []
        for message in response["messages"]:
            if isinstance(message, AIMessage):
                num_requests += 1
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
            "tolen_usage": total_usage,
            "num_requests": num_requests
        }
    except Exception as e:
        return {"error": f"Agent failed during reasoning or tool execution - {str(e)}"}

# =========================
# CHAT INPUT
# =========================
if prompt := st.chat_input("Ask about NSE stocks..."):
    st.session_state.response.append({
        "content": {"role": "user", "content": prompt},
        "tool_calls": [],
        "tolen_usage": 0,
    })
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = stock_agent(
                model=st.session_state.model,
                api_key=st.session_state.api_key,
                base_url=st.session_state.base_url,
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
            
            if result.get("tolen_usage", 0) != 0 and result.get("num_requests", 0) != 0:
                st.markdown(f":small[Api request: {result.get("num_requests", 0)}, Token usage: {result.get("tolen_usage", 0)}]") 
            
            st.session_state.response.append({
                "content": {"role": "assistant", "content": result["content"]},
                "tool_calls": result["tool_calls"],
                "tolen_usage": result["tolen_usage"]
            })
