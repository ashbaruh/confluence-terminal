"""Which assets are scanned. US = S&P 500 (auto), IL = TASE leaders, CRYPTO = majors.
Add your own tickers to state/extra_tickers.json (the on-demand button does this)."""
import csv
import io
import json
import urllib.request

from config import STATE

SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Tel Aviv leaders (Yahoo symbols). Symbols that fail to download are skipped automatically.
IL = {
    "LUMI.TA": ("בנק לאומי", "Financials"), "POLI.TA": ("בנק הפועלים", "Financials"),
    "DSCT.TA": ("בנק דיסקונט", "Financials"), "MZTF.TA": ("מזרחי טפחות", "Financials"),
    "FIBI.TA": ("הבינלאומי", "Financials"), "TEVA.TA": ("טבע", "Health Care"),
    "NICE.TA": ("נייס", "Information Technology"), "ICL.TA": ("כיל", "Materials"),
    "ESLT.TA": ("אלביט מערכות", "Industrials"), "BEZQ.TA": ("בזק", "Communication Services"),
    "AZRG.TA": ("עזריאלי", "Real Estate"), "NVMI.TA": ("נובה", "Information Technology"),
    "TSEM.TA": ("טאואר", "Information Technology"), "ORA.TA": ("אורמת", "Utilities"),
    "PHOE.TA": ("הפניקס", "Financials"), "HARL.TA": ("הראל", "Financials"),
    "CLIS.TA": ("כלל ביטוח", "Financials"), "MGDL.TA": ("מגדל", "Financials"),
    "BIG.TA": ("ביג", "Real Estate"), "DLEKG.TA": ("קבוצת דלק", "Energy"),
    "NWMD.TA": ("ניומד", "Energy"), "ENLT.TA": ("אנלייט", "Utilities"),
    "SPEN.TA": ("שפיר", "Industrials"), "ALHE.TA": ("אלוני חץ", "Real Estate"),
    "MLSR.TA": ("מליסרון", "Real Estate"), "AMOT.TA": ("אמות", "Real Estate"),
    "SAE.TA": ("שופרסל", "Consumer Staples"), "STRS.TA": ("שטראוס", "Consumer Staples"),
    "OPCE.TA": ("או.פי.סי", "Utilities"), "CAMT.TA": ("קמטק", "Information Technology"),
    "ENOG.TA": ("אנרג'יאן", "Energy"), "FTAL.TA": ("פתאל", "Consumer Discretionary"),
    "ELAL.TA": ("אל על", "Industrials"), "MTAV.TA": ("מיטב", "Financials"),
    "ISRA.TA": ("ישראמקו", "Energy"), "DANE.TA": ("דנאל", "Health Care"),
}

CRYPTO = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "XRP",
    "BNB-USD": "BNB", "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink", "DOT-USD": "Polkadot", "LTC-USD": "Litecoin", "TRX-USD": "TRON",
}


def market_of(sym: str) -> str:
    if sym.endswith(".TA") or sym.startswith("^TA"):
        return "IL"
    if sym.endswith("-USD"):
        return "CRYPTO"
    return "US"


def load_extra():
    p = STATE / "extra_tickers.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def build_universe(include_us=True):
    """Return {symbol: {"name", "sector", "market"}}."""
    u = {}
    if include_us:
        try:
            raw = urllib.request.urlopen(SP500_CSV, timeout=30).read().decode("utf-8")
            for row in csv.DictReader(io.StringIO(raw)):
                sym = row["Symbol"].replace(".", "-")
                u[sym] = {"name": row["Security"], "sector": row["GICS Sector"], "market": "US"}
        except Exception as e:  # keep running with IL + crypto
            print("S&P list failed:", e)
    for s, (n, sec) in IL.items():
        u[s] = {"name": n, "sector": sec, "market": "IL"}
    for s, n in CRYPTO.items():
        u[s] = {"name": n, "sector": "Crypto", "market": "CRYPTO"}
    for s in load_extra():
        s = s.upper()
        if s not in u:
            u[s] = {"name": s, "sector": "Custom", "market": market_of(s)}
    return u
