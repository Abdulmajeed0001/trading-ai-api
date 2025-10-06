# app.py
import os, json
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS

# OpenAI (SDK الحديث)
try:
    from openai import OpenAI
    OPENAI_OK = True
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    OPENAI_OK = False
    client = None

app = Flask(__name__)
CORS(app)

def _pick_frame(expire_iso: str):
    # يحاول حساب الفرق بالأيام
    try:
        d = datetime.fromisoformat(expire_iso)
        days = (d - datetime.now()).days
    except Exception:
        days = 10  # افتراضي
    if days <= 2:
        return "5m", "5d"
    if days <= 7:
        return "15m", "7d"
    if days <= 14:
        return "1h", "1mo"
    return "4h", "3mo"

def _ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(period).mean()
    roll_down = pd.Series(loss).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=close.index)

def _tech_snapshot(df: pd.DataFrame):
    # df: OHLCV من yfinance
    close = df["Close"]
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi14 = _rsi(close, 14)

    info = {
        "last_close": float(close.iloc[-1]),
        "ema20": float(ema20.iloc[-1]) if not ema20.isna().iloc[-1] else None,
        "ema50": float(ema50.iloc[-1]) if not ema50.isna().iloc[-1] else None,
        "rsi14": float(rsi14.iloc[-1]) if not rsi14.isna().iloc[-1] else None,
        "above_ema20": bool(close.iloc[-1] > ema20.iloc[-1]) if not ema20.isna().iloc[-1] else None,
        "above_ema50": bool(close.iloc[-1] > ema50.iloc[-1]) if not ema50.isna().iloc[-1] else None,
        "rsi_zone": "overbought" if rsi14.iloc[-1] >= 70 else "oversold" if rsi14.iloc[-1] <= 30 else "neutral"
    }
    return info

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    JSON IN:
    {
      "symbol": "AAPL",
      "contract": {
        "Symbol": "...", "Type": "CALL/PUT", "Strike": "...",
        "Expire": "2025-10-10", "Volume": "...", "Premium": "...",
        "Delta": "...", "Theta": "...", "IV": "...", "Side": "BUY/SELL", ...
      }
    }
    """
    data = request.json or {}
    symbol = (data.get("symbol") or "").upper().strip()
    contract = data.get("contract") or {}
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    expire = contract.get("Expire", "")
    interval, period = _pick_frame(expire)

    # جلب الشموع
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise ValueError("No data")
        df = df.dropna().tail(120)
    except Exception as e:
        return jsonify({"error": f"failed to fetch candles: {e}"}), 500

    # لقطة مؤشرات
    tech = _tech_snapshot(df)
    # سنرسل آخر 50 شمعة فقط لتقليل الحجم
    candles = df.reset_index().tail(50)
    candles_dict = candles.to_dict(orient="records")

    # تحليل بديل (في حال عدم وجود مفتاح)
    fallback_text = _fallback_comment(symbol, interval, contract, tech)

    # لو مفتاح OpenAI موجود: اطلب تحليلاً لغوياً أفضل بالعربية
    if OPENAI_OK and os.getenv("OPENAI_API_KEY"):
        prompt = f"""
حلّل هذا العقد في ضوء سلوك السعر الحالي للسهم {symbol} على إطار {interval}.
بيانات العقد (كما هي):
{json.dumps(contract, ensure_ascii=False)}

ملخص المؤشرات:
- آخر إغلاق: {tech['last_close']:.4f}
- EMA20: {tech['ema20']}
- EMA50: {tech['ema50']}
- RSI14: {tech['rsi14']} ({tech['rsi_zone']})
- السهم فوق EMA20؟ {tech['above_ema20']}
- السهم فوق EMA50؟ {tech['above_ema50']}

اعتبارات:
- لو العقد أسبوعي (≤7 أيام)، ركّز على زخم 15 دقيقة وإشارات الانعكاس السريعة.
- لو أسبوعين (≤14 يوم)، اعتمد 1h لقياس الاتجاه اللحظي والتصحيحات.
- غير ذلك، 4h لاتجاه أوسع.

مطلوب:
- اتجاه متوقّع قصير الأجل مدعوم بسببين فنيين على الأقل.
- كيف يتقاطع ذلك مع نوع العقد (CALL/PUT)، السترايك، والـIV.
- مخاطر محتملة وخطة بديلة (invalidations).
- نقاط دخول/خروج مقترحة (تقديرية) مع توضيح أنها ليست توصية.
- اكتب بإيجاز ووضوح بالعربية الاحترافية.
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "أنت محلل فني محترف في أسواق الأسهم الأمريكية والاختيارات."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()
        except Exception:
            text = fallback_text
    else:
        text = fallback_text

    return jsonify({
        "frame_used": interval,
        "tech": tech,
        "analysis": text,
        "candles_preview": candles_dict[-10:],  # آخر 10 فقط كمعاينة
    })

def _fallback_comment(symbol, interval, contract, tech):
    side = (contract.get("Side") or "").upper()
    typ = (contract.get("Type") or "").upper()
    iv = contract.get("IV") or "-"
    strike = contract.get("Strike") or "-"
    bias = "محايد"
    if tech["above_ema20"] and tech["above_ema50"]:
        bias = "صعودي"
    elif not tech["above_ema20"] and not tech["above_ema50"]:
        bias = "هبوطي"

    rsi_note = {
        "overbought": "RSI مرتفع (تشبّع شرائي) → احتمال تصحيح.",
        "oversold": "RSI منخفض (تشبّع بيعي) → احتمال ارتداد.",
        "neutral": "RSI حيادي."
    }[tech["rsi_zone"]]

    return (
        f"تحليل سريع ({symbol} | {interval})\n"
        f"- انحياز فني: {bias}. آخر إغلاق {tech['last_close']:.2f}.\n"
        f"- EMA20/50: فوقهما؟ {tech['above_ema20']}/{tech['above_ema50']}.\n"
        f"- {rsi_note}\n"
        f"- العقد: {typ} {side}, سترايك {strike}, IV={iv}.\n"
        f"- الملاحظة: لو {typ=='CALL' and bias=='صعودي'} فالتوافق إيجابي، ولعكسه الحذر.\n"
        f"هذا تلخيص آلي بديل عند غياب نموذج اللغة."
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
