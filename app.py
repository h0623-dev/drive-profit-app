# app.py
import streamlit as st
import streamlit.components.v1 as components
import sqlite3
from datetime import datetime, date, timedelta
import requests
import re
import hashlib
import hmac
import secrets
import os
import pandas as pd

DB_PATH = "drive_profit.db"

# ============================================================
# UI (CSS)
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
        h1,h2,h3 {letter-spacing:-0.2px;}
        section[data-testid="stSidebar"] {background: #fbfbfd;}
        section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
        .card {
          background: white;
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 16px;
          padding: 14px 16px;
          box-shadow: 0 1px 10px rgba(0,0,0,0.04);
          margin-bottom: 12px;
        }
        .muted {color: rgba(0,0,0,0.55);}
        .pill {
          display:inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          background: rgba(0,0,0,0.05);
          font-size: 12px;
          margin-left: 6px;
        }
        .stButton>button {border-radius: 12px; padding: 0.55rem 0.9rem; font-weight: 700;}
        .stTextInput>div>div>input, .stNumberInput input, .stSelectbox>div>div {border-radius: 12px !important;}
        div[data-testid="stDataFrame"] {border-radius: 14px; overflow:hidden; border: 1px solid rgba(0,0,0,0.06);}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Helpers: format/parse (NO decimals)
# ============================================================
def iround(x) -> int:
    try:
        return int(round(float(str(x).replace(",", "").strip())))
    except Exception:
        return 0

def fmt_won(x) -> str:
    return f"{iround(x):,}원"

def fmt_km(x) -> str:
    return f"{iround(x):,}KM"

def fmt_l(x) -> str:
    return f"{iround(x):,}L"

def fmt_won_per_l(x) -> str:
    return f"{iround(x):,}원/L"

def fmt_pct(x) -> str:
    return f"{iround(x)}%"

def parse_int_from_text(s: str | None) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    s = re.sub(r"[^\d\-]", "", s)
    try:
        return int(s)
    except Exception:
        return 0

def fmt_comma_int(x) -> str:
    return f"{iround(x):,}"

# ============================================================
# Security
# ============================================================
def _pbkdf2_hash(password: str, salt_hex: str | None = None, iterations: int = 200_000) -> str:
    if salt_hex is None:
        salt = secrets.token_bytes(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt_hex}${dk.hex()}"

def _verify_pbkdf2(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, _hash = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        recomputed = _pbkdf2_hash(password, salt_hex=salt_hex, iterations=int(iters))
        return hmac.compare_digest(recomputed, stored)
    except Exception:
        return False

def _normalize_recovery_code(code: str) -> str:
    return re.sub(r"\D+", "", code or "")

# ============================================================
# Kakao key (NOT shown in UI)
# ============================================================
def get_kakao_key() -> str:
    key = (os.getenv("KAKAO_REST_API_KEY", "") or "").strip()
    if key:
        return key
    try:
        return (st.secrets.get("KAKAO_REST_API_KEY", "") or "").strip()  # type: ignore
    except Exception:
        return ""

def _kakao_headers():
    k = get_kakao_key()
    return {"Authorization": f"KakaoAK {k}"} if k else {}

# ============================================================
# DB helpers
# ============================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cur.fetchall()}

def col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return col in table_columns(conn, table)

# ============================================================
# OPINET national daily fuel cache
# ============================================================
def fetch_opinet_national_prices():
    out = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    # 휘발유/경유: 전국 행 heuristic
    try:
        url = "https://www.opinet.co.kr/user/dopospdrg/dopOsPdrgAreaView.do"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            html = r.text
            m = re.search(r"전국.*?(\d{3,4}\.\d+).*?(\d{3,4}\.\d+)", html, re.DOTALL)
            if m:
                block = m.group(0)
                nums = [float(x) for x in re.findall(r"(\d{3,4}\.\d+)", block)]
                nums = [n for n in nums if 500 < n < 5000]
                if len(nums) >= 2:
                    g, d = nums[-2], nums[-1]
                    if d > g:
                        g, d = d, g
                    out["휘발유"] = g
                    out["경유"] = d
    except Exception:
        pass

    # LPG
    try:
        url = "https://www.opinet.co.kr/user/dopvsavsel/dopVsAvselSelect.do"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            html = r.text
            m = re.search(r"(\d{3,4}\.\d+)", html)
            if m:
                val = float(m.group(1))
                if 300 < val < 5000:
                    out["LPG"] = val
    except Exception:
        pass

    return out

def upsert_fuel_price_daily(price_date: date, fuel_type: str, price: float, source: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fuel_prices_daily (price_date, fuel_type, price_krw_per_l, source, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(price_date, fuel_type) DO UPDATE SET
            price_krw_per_l=excluded.price_krw_per_l,
            source=excluded.source,
            fetched_at=excluded.fetched_at
    """, (
        price_date.isoformat(), fuel_type, float(price), source,
        datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()

def get_fuel_price_daily_latest(fuel_type: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT price_krw_per_l, price_date, source
        FROM fuel_prices_daily
        WHERE fuel_type=?
        ORDER BY price_date DESC
        LIMIT 1
    """, (fuel_type,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None, None, None
    return float(row[0]), row[1], row[2]

def refresh_fuel_prices_daily_if_needed():
    today = date.today().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fuel_prices_daily WHERE price_date=?", (today,))
    cnt = cur.fetchone()[0]
    conn.close()
    if cnt >= 2:
        return
    prices = fetch_opinet_national_prices()
    for ft, p in prices.items():
        upsert_fuel_price_daily(date.today(), ft, p, "OPINET(전국 평균)")

# ============================================================
# DB init + migrations (role 포함)
# ============================================================
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            pw_hash TEXT NOT NULL,
            recovery_hash TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            fuel_type TEXT NOT NULL,
            fuel_eff_km_per_l REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            trip_date TEXT NOT NULL,
            vehicle_id INTEGER NOT NULL,

            trip_type TEXT NOT NULL,
            paid_distance_km REAL NOT NULL,
            empty_distance_km REAL NOT NULL,
            total_distance_km REAL NOT NULL,

            fare_krw REAL NOT NULL,

            fuel_type TEXT NOT NULL DEFAULT '휘발유',
            region TEXT NOT NULL DEFAULT '전국',
            fuel_price_krw_per_l REAL NOT NULL,
            fuel_used_l REAL NOT NULL,
            fuel_cost_krw REAL NOT NULL,

            toll_krw REAL NOT NULL DEFAULT 0,
            parking_krw REAL NOT NULL DEFAULT 0,
            other_cost_krw REAL NOT NULL DEFAULT 0,

            total_cost_krw REAL NOT NULL,
            profit_krw REAL NOT NULL,
            profit_margin_pct REAL NOT NULL,

            origin_text TEXT,
            dest_text TEXT,
            route_mode TEXT,

            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fuel_prices_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            price_date TEXT NOT NULL,
            fuel_type TEXT NOT NULL,
            price_krw_per_l REAL NOT NULL,
            source TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(price_date, fuel_type)
        )
    """)

    if not col_exists(conn, "users", "role"):
        try:
            cur.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        except Exception:
            pass

    conn.commit()

    # ensure admin exists
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO users (username, pw_hash, recovery_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", _pbkdf2_hash("admin1234"), _pbkdf2_hash(_normalize_recovery_code("000000")), "admin",
             datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
    else:
        cur.execute("UPDATE users SET role='admin' WHERE username='admin'")
        conn.commit()

    conn.close()

# ============================================================
# Auth (users)
# ============================================================
def get_user_by_username(username: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, pw_hash, recovery_hash, role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row

def get_user_info(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, role FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return ("user", "user")
    return (row[0], row[1])

def get_user_pw_hash(user_id: int) -> str | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pw_hash FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def set_user_password(user_id: int, new_password: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET pw_hash=? WHERE id=?", (_pbkdf2_hash(new_password), user_id))
    conn.commit()
    conn.close()

def user_has_any_vehicle(user_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM vehicles WHERE user_id=?", (user_id,))
    cnt = cur.fetchone()[0]
    conn.close()
    return cnt > 0

# ============================================================
# Auth token (URL persistence)
# ============================================================
def _token_pepper() -> str:
    p = (os.getenv("APP_TOKEN_PEPPER", "") or "").strip()
    if p:
        return p
    try:
        return (st.secrets.get("APP_TOKEN_PEPPER", "") or "").strip()  # type: ignore
    except Exception:
        return ""

def _hash_token(token: str) -> str:
    return hashlib.sha256((token + _token_pepper()).encode("utf-8")).hexdigest()

def issue_login_token(user_id: int, days_valid: int = 14) -> str:
    token = secrets.token_urlsafe(32)
    th = _hash_token(token)
    now = datetime.now()
    exp = now + timedelta(days=days_valid)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO auth_tokens (user_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (user_id, th, now.isoformat(timespec="seconds"), exp.isoformat(timespec="seconds"))
    )
    conn.commit()
    conn.close()
    return token

def validate_login_token(token: str) -> int | None:
    if not token:
        return None
    th = _hash_token(token)
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id FROM auth_tokens WHERE token_hash=? AND expires_at >= ? LIMIT 1",
        (th, now)
    )
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else None

def revoke_login_token(token: str):
    if not token:
        return
    th = _hash_token(token)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM auth_tokens WHERE token_hash=?", (th,))
    conn.commit()
    conn.close()

# ============================================================
# Vehicles / Trips DB
# ============================================================
def add_vehicle_basic(user_id: int, name: str, fuel_type: str, fuel_eff: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO vehicles (user_id, name, fuel_type, fuel_eff_km_per_l, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, name.strip(), fuel_type, float(fuel_eff), datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()
    vid = int(cur.lastrowid)
    conn.close()
    return vid

def list_vehicles(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, fuel_type, fuel_eff_km_per_l FROM vehicles WHERE user_id=? ORDER BY id DESC",
        (user_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def update_vehicle(user_id: int, vehicle_id: int, name: str, fuel_type: str, fuel_eff: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE vehicles
        SET name=?, fuel_type=?, fuel_eff_km_per_l=?
        WHERE user_id=? AND id=?
    """, (name.strip(), fuel_type, float(fuel_eff), user_id, vehicle_id))
    conn.commit()
    conn.close()

def delete_vehicle_cascade(user_id: int, vehicle_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM trips WHERE user_id=? AND vehicle_id=?", (user_id, vehicle_id))
    cur.execute("DELETE FROM vehicles WHERE user_id=? AND id=?", (user_id, vehicle_id))
    conn.commit()
    conn.close()

def insert_trip(
    user_id: int,
    trip_date: date,
    vehicle_row: dict,
    trip_type: str,
    paid_oneway_km: int,
    empty_oneway_km: int,
    fare_krw: int,
    fuel_price_krw_per_l: int,
    toll_krw: int,
    parking_krw: int,
    other_cost_krw: int,
    origin_text: str,
    dest_text: str,
    route_mode: str,
):
    multiplier = 2.0 if trip_type == "왕복" else 1.0
    paid_total = float(paid_oneway_km) * multiplier
    empty_total = float(empty_oneway_km) * multiplier
    total_distance = paid_total + empty_total

    fuel_used_l = total_distance / float(vehicle_row["fuel_eff_km_per_l"])
    fuel_cost = fuel_used_l * float(fuel_price_krw_per_l)

    total_cost = fuel_cost + float(toll_krw) + float(parking_krw) + float(other_cost_krw)
    profit = float(fare_krw) - total_cost
    margin = (profit / float(fare_krw) * 100.0) if fare_krw > 0 else 0.0

    now = datetime.now().isoformat(timespec="seconds")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trips (
            user_id, trip_date, vehicle_id,
            trip_type, paid_distance_km, empty_distance_km, total_distance_km,
            fare_krw,
            fuel_type, region,
            fuel_price_krw_per_l, fuel_used_l, fuel_cost_krw,
            toll_krw, parking_krw, other_cost_krw,
            total_cost_krw, profit_krw, profit_margin_pct,
            origin_text, dest_text, route_mode,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, trip_date.isoformat(), int(vehicle_row["id"]),
        trip_type, paid_total, empty_total, total_distance,
        float(fare_krw),
        vehicle_row.get("fuel_type") or "휘발유", "전국",
        float(fuel_price_krw_per_l), float(fuel_used_l), float(fuel_cost),
        float(toll_krw), float(parking_krw), float(other_cost_krw),
        float(total_cost), float(profit), float(margin),
        origin_text, dest_text, route_mode,
        now
    ))
    conn.commit()
    conn.close()

    return {
        "total_distance": total_distance,
        "fuel_used_l": fuel_used_l,
        "fuel_cost": fuel_cost,
        "total_cost": total_cost,
        "profit": profit,
        "margin": margin,
    }

def trips_df(user_id: int, vehicle_id: int | None, start: date, end: date) -> pd.DataFrame:
    conn = get_conn()
    params = {"uid": user_id, "start": start.isoformat(), "end": end.isoformat()}
    where = "t.user_id=:uid AND t.trip_date>=:start AND t.trip_date<=:end"
    if vehicle_id is not None:
        where += " AND t.vehicle_id=:vid"
        params["vid"] = vehicle_id

    df = pd.read_sql_query(f"""
        SELECT
            t.id,
            t.trip_date,
            v.name AS vehicle_name,
            t.trip_type,
            t.paid_distance_km,
            t.empty_distance_km,
            t.total_distance_km,
            t.fare_krw,
            t.fuel_price_krw_per_l,
            t.fuel_used_l,
            t.fuel_cost_krw,
            t.toll_krw,
            t.parking_krw,
            t.other_cost_krw,
            t.total_cost_krw,
            t.profit_krw,
            t.profit_margin_pct,
            t.origin_text,
            t.dest_text,
            t.route_mode,
            t.created_at
        FROM trips t
        JOIN vehicles v ON v.id = t.vehicle_id
        WHERE {where}
        ORDER BY t.trip_date DESC, t.id DESC
    """, conn, params=params)
    conn.close()
    return df

# ============================================================
# Admin utilities
# ============================================================
def admin_list_users() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC", conn)
    conn.close()
    return df

def admin_reset_user_password(user_id: int, new_password: str):
    set_user_password(user_id, new_password)

def admin_delete_user(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM trips WHERE user_id=?", (user_id,))
    cur.execute("DELETE FROM vehicles WHERE user_id=?", (user_id,))
    cur.execute("DELETE FROM auth_tokens WHERE user_id=?", (user_id,))
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

def admin_all_trips(start: date, end: date) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT
            t.id,
            t.trip_date,
            u.username,
            v.name AS vehicle_name,
            t.trip_type,
            t.total_distance_km,
            t.fare_krw,
            t.total_cost_krw,
            t.profit_krw,
            t.profit_margin_pct,
            t.origin_text,
            t.dest_text,
            t.route_mode,
            t.created_at
        FROM trips t
        JOIN users u ON u.id = t.user_id
        JOIN vehicles v ON v.id = t.vehicle_id
        WHERE t.trip_date >= ? AND t.trip_date <= ?
        ORDER BY t.trip_date DESC, t.id DESC
    """, conn, params=(start.isoformat(), end.isoformat()))
    conn.close()
    return df

# ============================================================
# Kakao APIs (search + directions)
# ============================================================
KAKAO_LOCAL_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_LOCAL_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_NAVI_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"

@st.cache_data(ttl=120)
def kakao_search_places(query: str, size_address: int = 6, size_keyword: int = 10):
    if not get_kakao_key():
        return [], {"ok": False, "reason": "NO_KEY"}
    q = (query or "").strip()
    if not q:
        return [], {"ok": False, "reason": "EMPTY_QUERY"}

    debug = {"ok": True, "addr_status": None, "kw_status": None, "addr_err": "", "kw_err": ""}
    results = []

    # address search
    try:
        r = requests.get(KAKAO_LOCAL_ADDRESS_URL, headers=_kakao_headers(), params={"query": q, "size": int(size_address)}, timeout=10)
        debug["addr_status"] = r.status_code
        if r.status_code == 200:
            data = r.json()
            docs = data.get("documents", []) or []
            for d in docs:
                x = d.get("x"); y = d.get("y")
                road = ""
                jibun = ""
                if d.get("road_address"):
                    road = d["road_address"].get("address_name") or ""
                if d.get("address"):
                    jibun = d["address"].get("address_name") or ""
                label = road or jibun or q
                results.append({"x": x, "y": y, "place_name": label, "road_address_name": road, "address_name": jibun, "_source": "address"})
        else:
            debug["addr_err"] = (r.text or "")[:200]
    except Exception as e:
        debug["addr_status"] = "EXC"
        debug["addr_err"] = str(e)[:200]

    # keyword search
    try:
        r = requests.get(KAKAO_LOCAL_KEYWORD_URL, headers=_kakao_headers(), params={"query": q, "size": int(size_keyword)}, timeout=10)
        debug["kw_status"] = r.status_code
        if r.status_code == 200:
            data = r.json()
            docs = data.get("documents", []) or []
            for d in docs:
                d["_source"] = "keyword"
                results.append(d)
        else:
            debug["kw_err"] = (r.text or "")[:200]
    except Exception as e:
        debug["kw_status"] = "EXC"
        debug["kw_err"] = str(e)[:200]

    # dedupe
    seen = set()
    merged = []
    for d in results:
        try:
            keyxy = (float(d["x"]), float(d["y"]))
        except Exception:
            continue
        if keyxy in seen:
            continue
        seen.add(keyxy)
        merged.append(d)

    return merged, debug

@st.cache_data(ttl=120)
def kakao_route(origin_lng: float, origin_lat: float, dest_lng: float, dest_lat: float, priority: str, avoid: str | None):
    if not get_kakao_key():
        return None, {"ok": False, "reason": "NO_KEY"}
    params = {"origin": f"{origin_lng},{origin_lat}", "destination": f"{dest_lng},{dest_lat}", "priority": priority}
    if avoid:
        params["avoid"] = avoid
    try:
        r = requests.get(KAKAO_NAVI_DIRECTIONS_URL, headers=_kakao_headers(), params=params, timeout=15)
        if r.status_code != 200:
            return None, {"ok": False, "status": r.status_code, "err": (r.text or "")[:300]}
        j = r.json()
        routes = j.get("routes", [])
        if not routes:
            return None, {"ok": False, "status": 200, "err": "NO_ROUTES"}
        summary = routes[0].get("summary", {}) or {}
        fare = summary.get("fare", {}) or {}
        dist = int(summary.get("distance", 0) or 0)
        dur = int(summary.get("duration", 0) or 0)
        toll = int(fare.get("toll", 0) or 0)
        return {"distance_m": dist, "duration_s": dur, "toll_krw": toll}, {"ok": True}
    except Exception as e:
        return None, {"ok": False, "status": "EXC", "err": str(e)[:300]}

def get_browser_geolocation():
    html = """
    <script>
    const send = (value) => {
      const msg = {isStreamlitMessage: true, type: "streamlit:setComponentValue", value: value};
      window.parent.postMessage(msg, "*");
    };
    function getLoc(){
      if (!navigator.geolocation) { send({error: "Geolocation not supported"}); return; }
      navigator.geolocation.getCurrentPosition(
        (pos)=>send({lat: pos.coords.latitude, lng: pos.coords.longitude}),
        (err)=>send({error: err.message}),
        {enableHighAccuracy: true, timeout: 10000, maximumAge: 0}
      );
    }
    getLoc();
    </script>
    """
    return components.html(html, height=0)

# ============================================================
# App boot
# ============================================================
st.set_page_config(page_title="운행 손익 앱", page_icon="🚗", layout="wide")
inject_css()
init_db()
refresh_fuel_prices_daily_if_needed()

# ------------------------------------------------------------
# session defaults
# ------------------------------------------------------------
for k, v in {
    "user_id": None,
    "username": None,
    "role": None,
    "page": None,
    "selected_vehicle_id": None,

    "_geo": None,
    "_origin_pick": None,
    "_dest_pick": None,

    "origin_query": "",
    "dest_query": "",
    "origin_choice": "(선택 안 함)",
    "dest_choice": "(선택 안 함)",

    "trip_type": "편도",
    "paid_oneway_km_txt": "0",
    "empty_oneway_km_txt": "0",
    "fare_krw_txt": "30,000",

    # 유가/톨비는 '자동 기본값'이 들어오되, 사용자가 수정하면 보호됨
    "fuel_price_txt": "0",
    "fuel_user_edited": False,
    "fuel_price_pending": None,

    "toll_krw_txt": "0",
    "toll_user_edited": False,
    "toll_pending": None,

    "parking_krw_txt": "0",
    "other_cost_krw_txt": "0",

    "nav_km_oneway": 0,
    "nav_toll_oneway": 0,

    "origin_query_pending": None,
    "dest_query_pending": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# apply pending BEFORE widgets
if st.session_state["origin_query_pending"] is not None:
    st.session_state["origin_query"] = st.session_state["origin_query_pending"]
    st.session_state["origin_query_pending"] = None
if st.session_state["dest_query_pending"] is not None:
    st.session_state["dest_query"] = st.session_state["dest_query_pending"]
    st.session_state["dest_query_pending"] = None
if st.session_state["fuel_price_pending"] is not None:
    st.session_state["fuel_price_txt"] = st.session_state["fuel_price_pending"]
    st.session_state["fuel_price_pending"] = None
if st.session_state["toll_pending"] is not None:
    st.session_state["toll_krw_txt"] = st.session_state["toll_pending"]
    st.session_state["toll_pending"] = None

# ------------------------------------------------------------
# auto-login from URL token
# ------------------------------------------------------------
try:
    token_from_url = st.query_params.get("t", "")  # type: ignore
except Exception:
    token_from_url = ""

if (not st.session_state.user_id) and token_from_url:
    uid = validate_login_token(token_from_url)
    if uid:
        uname, role = get_user_info(uid)
        st.session_state.user_id = uid
        st.session_state.username = uname
        st.session_state.role = role
    else:
        try:
            st.query_params.clear()  # type: ignore
        except Exception:
            pass

def do_logout():
    try:
        t = st.query_params.get("t", "")  # type: ignore
    except Exception:
        t = ""
    if t:
        revoke_login_token(t)
    try:
        st.query_params.clear()  # type: ignore
    except Exception:
        pass
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.role = None
    st.rerun()

def on_fuel_edited():
    st.session_state["fuel_user_edited"] = True

def on_toll_edited():
    st.session_state["toll_user_edited"] = True

def login_screen():
    st.markdown('<div class="card"><h2>🔐 로그인</h2><p class="muted">새로고침해도 로그인 유지됩니다.</p></div>', unsafe_allow_html=True)
    tab_login, tab_signup, tab_reset = st.tabs(["로그인", "회원가입", "비밀번호 찾기(재설정)"])

    with tab_login:
        username = st.text_input("아이디", key="login_user")
        password = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인"):
            row = get_user_by_username((username or "").strip())
            if not row:
                st.error("아이디가 없어요.")
                return
            uid, uname, pw_hash, _recovery, role = row
            if _verify_pbkdf2(password, pw_hash):
                st.session_state.user_id = int(uid)
                st.session_state.username = uname
                st.session_state.role = role
                token = issue_login_token(int(uid), days_valid=14)
                try:
                    st.query_params["t"] = token  # type: ignore
                except Exception:
                    pass
                st.rerun()
            else:
                st.error("비밀번호가 틀렸어요.")

    with tab_signup:
        new_user = st.text_input("새 아이디", key="signup_user")
        new_pw = st.text_input("새 비밀번호(6자리 이상)", type="password", key="signup_pw")
        new_pw2 = st.text_input("새 비밀번호 확인", type="password", key="signup_pw2")
        recovery_code = st.text_input("복구코드(숫자 4자리 이상)", type="password", key="signup_rc")
        if st.button("회원가입"):
            if new_pw != new_pw2:
                st.error("비밀번호가 서로 달라요.")
            else:
                ok, msg = create_user(new_user, new_pw, recovery_code)
                (st.success if ok else st.error)(msg)

    with tab_reset:
        u = st.text_input("아이디", key="reset_user")
        rc = st.text_input("복구코드", type="password", key="reset_rc")
        npw = st.text_input("새 비밀번호(6자리 이상)", type="password", key="reset_npw")
        npw2 = st.text_input("새 비밀번호 확인", type="password", key="reset_npw2")
        if st.button("비밀번호 재설정"):
            if npw != npw2:
                st.error("새 비밀번호가 서로 달라요.")
            else:
                ok, msg = reset_password_with_recovery(u, rc, npw)
                (st.success if ok else st.error)(msg)

if not st.session_state.user_id:
    login_screen()
    st.stop()

USER_ID = int(st.session_state.user_id)
USERNAME = st.session_state.username or "user"
ROLE = st.session_state.role or "user"

# default page
if not st.session_state.page:
    st.session_state.page = "운행 입력" if user_has_any_vehicle(USER_ID) else "차량 등록"

# ============================================================
# Sidebar (카카오키 숨김 + 메뉴 고정)
# ============================================================
with st.sidebar:
    st.markdown(f"### 👤 {USERNAME} <span class='pill'>{'관리자' if ROLE=='admin' else '사용자'}</span>", unsafe_allow_html=True)
    if st.button("로그아웃"):
        do_logout()

    vehicles = list_vehicles(USER_ID)
    if vehicles:
        labels = [f"[{int(v[0])}] {v[1]} ({v[2]}, 연비 {iround(v[3])}KM/L)" for v in vehicles]
        chosen = st.selectbox("기본 차량", labels, index=0, key="sidebar_vehicle")
        st.session_state.selected_vehicle_id = int(re.search(r"\[(\d+)\]", chosen).group(1))
    else:
        st.session_state.selected_vehicle_id = None

    st.divider()
    menu = ["차량 등록", "운행 입력", "내역/리포트", "개인정보변경"]
    if ROLE == "admin":
        menu.append("관리자")
    if st.session_state.page not in menu:
        st.session_state.page = "운행 입력"
    st.session_state.page = st.radio("메뉴", menu, index=menu.index(st.session_state.page))

# ============================================================
# PAGE: 차량 등록
# ============================================================
if st.session_state.page == "차량 등록":
    st.markdown('<div class="card"><h2>🚗 차량 등록</h2><p class="muted">차량 종류 / 유종 / 연비만 등록합니다.</p></div>', unsafe_allow_html=True)

    with st.form("vehicle_form", clear_on_submit=True):
        name = st.text_input("차량 종류")
        fuel_type = st.selectbox("유종", ["휘발유", "경유", "LPG"])
        fuel_eff = st.number_input("연비 (KM/L)", min_value=1, max_value=100, value=12, step=1, format="%d")
        if st.form_submit_button("등록"):
            if not (name or "").strip():
                st.error("차량 종류를 입력해줘.")
            else:
                add_vehicle_basic(USER_ID, name, fuel_type, int(fuel_eff))
                st.success("등록 완료!")
                st.session_state.page = "운행 입력"
                st.rerun()

# ============================================================
# PAGE: 운행 입력
# ============================================================
elif st.session_state.page == "운행 입력":
    vehicles = list_vehicles(USER_ID)
    if not vehicles:
        st.warning("차량이 없어요. 먼저 차량 등록을 해줘.")
        st.stop()

    vid = st.session_state.selected_vehicle_id or int(vehicles[0][0])
    vrow = next((v for v in vehicles if int(v[0]) == int(vid)), vehicles[0])
    vehicle_row = {"id": int(vrow[0]), "name": vrow[1], "fuel_type": vrow[2], "fuel_eff_km_per_l": float(vrow[3])}

    # 유가 자동 기본값 (사용자가 수정한 적 없으면 자동으로 갱신)
    auto_price, auto_date, auto_src = get_fuel_price_daily_latest(vehicle_row["fuel_type"])
    auto_price_int = iround(auto_price) if auto_price is not None else 1700
    if not st.session_state["fuel_user_edited"]:
        st.session_state["fuel_price_pending"] = fmt_comma_int(auto_price_int)

    st.markdown(
        f"""
        <div class="card">
          <h2>💰 운행 입력</h2>
          <div class="muted">
            기본 차량: <b>[{vehicle_row['id']}] {vehicle_row['name']}</b> ({vehicle_row['fuel_type']}, 연비 {iround(vehicle_row['fuel_eff_km_per_l'])}KM/L)
            <br/>오늘 전국 평균 유가(자동 기본값): <b>{fmt_won_per_l(auto_price_int)}</b> ({auto_date or '-'} / {auto_src or 'OPINET'})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI preview card (uses current inputs)
    def compute_preview():
        trip_type = st.session_state.get("trip_type", "편도")
        paid_oneway = parse_int_from_text(st.session_state.get("paid_oneway_km_txt"))
        empty_oneway = parse_int_from_text(st.session_state.get("empty_oneway_km_txt"))
        fare = parse_int_from_text(st.session_state.get("fare_krw_txt"))
        fuel_price = parse_int_from_text(st.session_state.get("fuel_price_txt"))
        toll = parse_int_from_text(st.session_state.get("toll_krw_txt"))
        parking = parse_int_from_text(st.session_state.get("parking_krw_txt"))
        other = parse_int_from_text(st.session_state.get("other_cost_krw_txt"))

        mult = 2 if trip_type == "왕복" else 1
        total_km = (paid_oneway + empty_oneway) * mult
        fuel_used = 0
        fuel_cost = 0
        if vehicle_row["fuel_eff_km_per_l"] > 0:
            fuel_used = total_km / float(vehicle_row["fuel_eff_km_per_l"])
            fuel_cost = fuel_used * max(fuel_price, 0)
        total_cost = fuel_cost + toll + parking + other
        profit = fare - total_cost
        return total_km, fuel_used, fuel_cost, total_cost, profit

    km_preview, fuel_used_preview, fuel_cost_preview, total_cost_preview, profit_preview = compute_preview()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("예상 총거리", fmt_km(km_preview))
    c2.metric("예상 연료", fmt_l(fuel_used_preview))
    c3.metric("예상 기름값", fmt_won(fuel_cost_preview))
    c4.metric("예상 총비용", fmt_won(total_cost_preview))
    c5.metric("예상 순이익", fmt_won(profit_preview))

    st.markdown('<div class="card"><h3>📍 네비(선택)</h3><p class="muted">길찾기 결과로 유상거리/톨비를 자동 입력합니다. (유가/톨비는 자동값이 들어오지만 수정 가능)</p></div>', unsafe_allow_html=True)

    if get_kakao_key():
        colA, colB, colC = st.columns([2, 5, 3])
        with colA:
            origin_mode = st.selectbox("출발지 방식", ["현재 위치", "출발지 주소/장소명"], index=0, key="origin_mode")
        with colB:
            st.text_input("출발지 입력", key="origin_query", disabled=(origin_mode == "현재 위치"))
        with colC:
            st.selectbox("경로 옵션", ["추천(추천도로)", "최단시간", "최단거리", "무료도로 우선"], index=0, key="route_mode")

        st.text_input("도착지 입력", key="dest_query")

        def best_text_from_doc(doc: dict) -> str:
            road = (doc.get("road_address_name") or "").strip()
            jibun = (doc.get("address_name") or "").strip()
            place = (doc.get("place_name") or "").strip()
            return road or jibun or place

        # origin list
        if origin_mode == "출발지 주소/장소명" and (st.session_state["origin_query"] or "").strip():
            origin_results, _dbg = kakao_search_places(st.session_state["origin_query"].strip())
            if origin_results:
                olabels = ["(선택 안 함)"]
                odocs = [None]
                for d in origin_results:
                    place = (d.get("place_name") or "(이름없음)").strip()
                    road = (d.get("road_address_name") or "").strip()
                    jibun = (d.get("address_name") or "").strip()
                    olabels.append(f"{place} | 도로명: {road if road else '-'} | 지번: {jibun if jibun else '-'}")
                    odocs.append(d)
                choice = st.selectbox("출발지 검색 결과", olabels, index=0, key="origin_choice")
                if choice != "(선택 안 함)":
                    picked = odocs[olabels.index(choice)]
                    st.session_state["_origin_pick"] = picked
                    new_text = best_text_from_doc(picked)
                    if new_text and new_text != (st.session_state.get("origin_query") or ""):
                        st.session_state["origin_query_pending"] = new_text
                        st.rerun()
                else:
                    st.session_state["_origin_pick"] = None

        # dest list
        if (st.session_state["dest_query"] or "").strip():
            dest_results, _dbg = kakao_search_places(st.session_state["dest_query"].strip())
            if dest_results:
                dlabels = ["(선택 안 함)"]
                ddocs = [None]
                for d in dest_results:
                    place = (d.get("place_name") or "(이름없음)").strip()
                    road = (d.get("road_address_name") or "").strip()
                    jibun = (d.get("address_name") or "").strip()
                    dlabels.append(f"{place} | 도로명: {road if road else '-'} | 지번: {jibun if jibun else '-'}")
                    ddocs.append(d)
                choice = st.selectbox("도착지 검색 결과", dlabels, index=0, key="dest_choice")
                if choice != "(선택 안 함)":
                    picked = ddocs[dlabels.index(choice)]
                    st.session_state["_dest_pick"] = picked
                    new_text = best_text_from_doc(picked)
                    if new_text and new_text != (st.session_state.get("dest_query") or ""):
                        st.session_state["dest_query_pending"] = new_text
                        st.rerun()
                else:
                    st.session_state["_dest_pick"] = None

        if origin_mode == "현재 위치":
            if st.button("현재 위치 가져오기"):
                st.session_state["_geo"] = get_browser_geolocation()

        route_mode = st.session_state.get("route_mode") or "추천(추천도로)"
        if str(route_mode).startswith("추천"):
            priority, avoid = "RECOMMEND", None
        elif route_mode == "최단시간":
            priority, avoid = "TIME", None
        elif route_mode == "최단거리":
            priority, avoid = "DISTANCE", None
        else:
            priority, avoid = "RECOMMEND", "toll"

        if st.button("거리/톨비 계산"):
            origin_lng = origin_lat = None
            if origin_mode == "현재 위치":
                geo = st.session_state.get("_geo")
                if not (isinstance(geo, dict) and geo.get("lat") and geo.get("lng")):
                    st.error("현재 위치를 먼저 가져와줘.")
                else:
                    origin_lat = float(geo["lat"])
                    origin_lng = float(geo["lng"])
            else:
                od = st.session_state.get("_origin_pick")
                if not od:
                    st.error("출발지 검색 결과에서 하나를 선택해줘.")
                else:
                    origin_lng = float(od["x"])
                    origin_lat = float(od["y"])

            dd = st.session_state.get("_dest_pick")
            if not dd:
                st.error("도착지 검색 결과에서 하나를 선택해줘.")
            elif origin_lng is not None and origin_lat is not None:
                dest_lng = float(dd["x"])
                dest_lat = float(dd["y"])
                res, dbg2 = kakao_route(origin_lng, origin_lat, dest_lng, dest_lat, priority=priority, avoid=avoid)
                if not res:
                    st.error(f"길찾기 실패: {dbg2}")
                else:
                    km_oneway = int(round(res["distance_m"] / 1000.0))
                    toll_oneway = int(round(res["toll_krw"]))
                    minutes_oneway = int(round(res["duration_s"] / 60.0))

                    st.session_state["nav_km_oneway"] = km_oneway
                    st.session_state["nav_toll_oneway"] = toll_oneway

                    # 유상거리 자동 반영(안전: pending+rerun 방식)
                    st.session_state["paid_oneway_km_txt"] = fmt_comma_int(km_oneway)

                    # 톨비 자동 기본값 반영(사용자가 수정한 적 없으면 덮어씀)
                    if not st.session_state["toll_user_edited"]:
                        if st.session_state.get("trip_type") == "왕복":
                            st.session_state["toll_pending"] = fmt_comma_int(toll_oneway * 2)
                        else:
                            st.session_state["toll_pending"] = fmt_comma_int(toll_oneway)

                    st.success(f"거리(편도): {fmt_km(km_oneway)} | 톨비(편도): {fmt_won(toll_oneway)} | 소요(편도): {minutes_oneway:,}분")
                    st.rerun()
    else:
        st.info("네비 기능은 서버에 카카오 키가 설정된 경우에만 동작합니다. (키는 화면에 표시하지 않음)")

    st.markdown('<div class="card"><h3>🧾 운행 정보</h3><p class="muted">유가/톨비는 자동값이 들어오며, 필요하면 직접 수정할 수 있어요.</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        trip_date = st.date_input("운행 날짜", value=date.today())
        st.session_state["trip_type"] = st.selectbox("운행 형태", ["편도", "왕복"], index=0 if st.session_state["trip_type"] == "편도" else 1)

    # 왕복이면 (네비 톨비 자동값이 있는 경우) 사용자 미수정일 때만 자동 2배 반영
    nav_toll = int(st.session_state.get("nav_toll_oneway") or 0)
    if nav_toll > 0 and (not st.session_state["toll_user_edited"]):
        st.session_state["toll_pending"] = fmt_comma_int(nav_toll * 2) if st.session_state["trip_type"] == "왕복" else fmt_comma_int(nav_toll)

    with col2:
        st.text_input("유상거리(편도, KM)", key="paid_oneway_km_txt")
        st.text_input("공차거리(편도, KM)", key="empty_oneway_km_txt")
    with col3:
        st.text_input("운임료(원)", key="fare_krw_txt")

    col4, col5, col6, col7 = st.columns(4)
    with col4:
        st.text_input(f"유가(원/L) 자동 기본값 ({auto_date or '-'} / {auto_src or 'OPINET'})", key="fuel_price_txt", on_change=on_fuel_edited)
        st.caption(f"표시: {fmt_won_per_l(st.session_state['fuel_price_txt'])}")
    with col5:
        st.text_input("톨비(원) 자동 기본값(네비)", key="toll_krw_txt", on_change=on_toll_edited)
        st.caption(f"표시: {fmt_won(st.session_state['toll_krw_txt'])}")
    with col6:
        st.text_input("주차비(원)", key="parking_krw_txt")
        st.caption(f"표시: {fmt_won(st.session_state['parking_krw_txt'])}")
    with col7:
        st.text_input("기타비용(원)", key="other_cost_krw_txt")
        st.caption(f"표시: {fmt_won(st.session_state['other_cost_krw_txt'])}")

    if st.button("계산하고 저장"):
        paid_oneway_km = parse_int_from_text(st.session_state["paid_oneway_km_txt"])
        empty_oneway_km = parse_int_from_text(st.session_state["empty_oneway_km_txt"])
        fare_krw = parse_int_from_text(st.session_state["fare_krw_txt"])
        fuel_price = parse_int_from_text(st.session_state["fuel_price_txt"])
        toll = parse_int_from_text(st.session_state["toll_krw_txt"])
        parking = parse_int_from_text(st.session_state["parking_krw_txt"])
        other = parse_int_from_text(st.session_state["other_cost_krw_txt"])

        if paid_oneway_km <= 0 and empty_oneway_km <= 0:
            st.error("유상거리/공차거리 중 하나는 0보다 커야 해.")
        elif fare_krw <= 0:
            st.error("운임료(원)를 입력해줘.")
        elif fuel_price <= 0:
            st.error("유가(원/L)를 입력해줘.")
        else:
            result = insert_trip(
                user_id=USER_ID,
                trip_date=trip_date,
                vehicle_row=vehicle_row,
                trip_type=st.session_state["trip_type"],
                paid_oneway_km=paid_oneway_km,
                empty_oneway_km=empty_oneway_km,
                fare_krw=fare_krw,
                fuel_price_krw_per_l=fuel_price,
                toll_krw=toll,
                parking_krw=parking,
                other_cost_krw=other,
                origin_text=(st.session_state.get("origin_query") or "").strip(),
                dest_text=(st.session_state.get("dest_query") or "").strip(),
                route_mode=(st.session_state.get("route_mode") or "").strip(),
            )

            st.success("저장 완료!")
            a, b, c, d, e, f = st.columns(6)
            a.metric("총거리", fmt_km(result["total_distance"]))
            b.metric("연료사용", fmt_l(result["fuel_used_l"]))
            c.metric("기름값", fmt_won(result["fuel_cost"]))
            d.metric("총비용", fmt_won(result["total_cost"]))
            e.metric("순이익", fmt_won(result["profit"]))
            f.metric("수익률", fmt_pct(result["margin"]))

# ============================================================
# PAGE: 내역/리포트
# ============================================================
elif st.session_state.page == "내역/리포트":
    st.markdown('<div class="card"><h2>📊 내역/리포트</h2><p class="muted">오늘/이번주/이번달 요약 + 차트를 제공합니다.</p></div>', unsafe_allow_html=True)

    vehicles = list_vehicles(USER_ID)
    if not vehicles:
        st.info("차량을 먼저 등록해줘.")
        st.stop()

    vlabels = ["전체 차량"] + [f"[{int(v[0])}] {v[1]} ({v[2]})" for v in vehicles]
    vsel = st.selectbox("차량 필터", vlabels, index=0)
    vehicle_id = None
    if vsel != "전체 차량":
        vehicle_id = int(re.search(r"\[(\d+)\]", vsel).group(1))

    today = date.today()
    start = st.date_input("시작일", value=today - timedelta(days=30))
    end = st.date_input("종료일", value=today)

    df = trips_df(USER_ID, vehicle_id, start, end)
    if df.empty:
        st.write("해당 기간/조건의 운행 내역이 없어.")
        st.stop()

    # Dashboard summary: today / this week / this month (by trip_date)
    df_dt = df.copy()
    df_dt["trip_date_dt"] = pd.to_datetime(df_dt["trip_date"]).dt.date

    def sum_block(dfsub: pd.DataFrame):
        if dfsub.empty:
            return 0, 0, 0, 0
        return (
            iround(dfsub["total_distance_km"].sum()),
            iround(dfsub["fare_krw"].sum()),
            iround(dfsub["total_cost_krw"].sum()),
            iround(dfsub["profit_krw"].sum()),
        )

    # today
    df_today = df_dt[df_dt["trip_date_dt"] == today]
    # week (Mon..today)
    week_start = today - timedelta(days=today.weekday())
    df_week = df_dt[(df_dt["trip_date_dt"] >= week_start) & (df_dt["trip_date_dt"] <= today)]
    # month
    month_start = date(today.year, today.month, 1)
    df_month = df_dt[(df_dt["trip_date_dt"] >= month_start) & (df_dt["trip_date_dt"] <= today)]

    t_km, t_fare, t_cost, t_profit = sum_block(df_today)
    w_km, w_fare, w_cost, w_profit = sum_block(df_week)
    m_km, m_fare, m_cost, m_profit = sum_block(df_month)

    st.markdown('<div class="card"><h3>📌 요약</h3><p class="muted">모든 값은 정수 + 단위 표시</p></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>오늘</h3></div>', unsafe_allow_html=True)
        st.metric("총거리", fmt_km(t_km))
        st.metric("총운임료", fmt_won(t_fare))
        st.metric("총비용", fmt_won(t_cost))
        st.metric("총순이익", fmt_won(t_profit))
    with c2:
        st.markdown('<div class="card"><h3>이번주</h3></div>', unsafe_allow_html=True)
        st.metric("총거리", fmt_km(w_km))
        st.metric("총운임료", fmt_won(w_fare))
        st.metric("총비용", fmt_won(w_cost))
        st.metric("총순이익", fmt_won(w_profit))
    with c3:
        st.markdown('<div class="card"><h3>이번달</h3></div>', unsafe_allow_html=True)
        st.metric("총거리", fmt_km(m_km))
        st.metric("총운임료", fmt_won(m_fare))
        st.metric("총비용", fmt_won(m_cost))
        st.metric("총순이익", fmt_won(m_profit))

    # Charts (daily aggregation)
    st.markdown('<div class="card"><h3>📈 차트</h3><p class="muted">기간 내 일자별 합계(총운임/총비용/총순이익)</p></div>', unsafe_allow_html=True)
    chart_df = df_dt.groupby("trip_date_dt", as_index=False).agg(
        fare=("fare_krw", "sum"),
        cost=("total_cost_krw", "sum"),
        profit=("profit_krw", "sum"),
        km=("total_distance_km", "sum"),
    ).sort_values("trip_date_dt")
    chart_df = chart_df.set_index("trip_date_dt")[["fare", "cost", "profit"]]
    st.line_chart(chart_df)

    st.markdown('<div class="card"><h3>🗂 내역</h3><p class="muted">순이익이 음수면 빨간색 표시</p></div>', unsafe_allow_html=True)

    view = df.copy()
    view.rename(columns={
        "id": "번호",
        "trip_date": "운행일자",
        "vehicle_name": "차량",
        "trip_type": "형태",
        "paid_distance_km": "유상거리",
        "empty_distance_km": "공차거리",
        "total_distance_km": "총거리",
        "fare_krw": "운임료",
        "fuel_price_krw_per_l": "유가",
        "fuel_used_l": "연료사용",
        "fuel_cost_krw": "기름값",
        "toll_krw": "톨비",
        "parking_krw": "주차비",
        "other_cost_krw": "기타비용",
        "total_cost_krw": "총비용",
        "profit_krw": "순이익",
        "profit_margin_pct": "수익률",
        "origin_text": "출발지",
        "dest_text": "도착지",
        "route_mode": "경로옵션",
        "created_at": "등록시각",
    }, inplace=True)

    ordered = [
        "번호", "운행일자", "차량", "형태",
        "출발지", "도착지", "경로옵션",
        "유상거리", "공차거리", "총거리",
        "운임료", "총비용", "순이익", "수익률",
        "유가", "연료사용", "기름값",
        "톨비", "주차비", "기타비용",
        "등록시각"
    ]
    view = view[ordered]

    def style_neg_profit(row):
        styles = [""] * len(row)
        try:
            v = float(row.get("순이익", 0))
        except Exception:
            v = 0.0
        if v < 0 and "순이익" in row.index:
            idx = list(row.index).index("순이익")
            styles[idx] = "color:#d00;font-weight:800;"
        return styles

    fmt_map = {
        "번호": lambda x: str(iround(x)),
        "유상거리": fmt_km,
        "공차거리": fmt_km,
        "총거리": fmt_km,
        "운임료": fmt_won,
        "총비용": fmt_won,
        "순이익": fmt_won,
        "수익률": fmt_pct,
        "유가": fmt_won_per_l,
        "연료사용": fmt_l,
        "기름값": fmt_won,
        "톨비": fmt_won,
        "주차비": fmt_won,
        "기타비용": fmt_won,
    }

    right_cols = ["번호","유상거리","공차거리","총거리","운임료","총비용","순이익","수익률","유가","연료사용","기름값","톨비","주차비","기타비용"]
    left_cols = ["운행일자","차량","형태","출발지","도착지","경로옵션","등록시각"]

    styler = (
        view.style
        .format(fmt_map, na_rep="")
        .set_properties(subset=right_cols, **{"text-align": "right"})
        .set_properties(subset=left_cols, **{"text-align": "left"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center"), ("font-weight", "800")]},
            {"selector": "td", "props": [("padding", "6px 10px")]},
        ])
        .apply(style_neg_profit, axis=1)
    )

    st.dataframe(styler, width="stretch", hide_index=True)
    csv = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSV 다운로드", data=csv, file_name="trips_report.csv", mime="text/csv")

# ============================================================
# PAGE: 개인정보변경 (차량 수정/삭제 + 비밀번호 변경)
# ============================================================
elif st.session_state.page == "개인정보변경":
    st.markdown('<div class="card"><h2>👤 개인정보변경</h2><p class="muted">비밀번호 변경 / 등록 차량 수정·삭제</p></div>', unsafe_allow_html=True)

    # password change
    with st.expander("🔐 비밀번호 변경", expanded=False):
        old_pw = st.text_input("현재 비밀번호", type="password")
        new_pw = st.text_input("새 비밀번호(6자리 이상)", type="password")
        new_pw2 = st.text_input("새 비밀번호 확인", type="password")
        if st.button("비밀번호 변경"):
            if new_pw != new_pw2:
                st.error("새 비밀번호가 서로 달라요.")
            elif len(new_pw) < 6:
                st.error("새 비밀번호는 6자리 이상.")
            else:
                stored = get_user_pw_hash(USER_ID)
                if not stored or not _verify_pbkdf2(old_pw, stored):
                    st.error("현재 비밀번호가 틀렸어요.")
                else:
                    set_user_password(USER_ID, new_pw)
                    st.success("변경 완료! 다음 로그인부터 적용됩니다.")

    vehicles = list_vehicles(USER_ID)
    if not vehicles:
        st.info("등록된 차량이 없어요. 먼저 차량 등록을 해줘.")
        st.stop()

    labels = [f"[{int(v[0])}] {v[1]} ({v[2]}, 연비 {iround(v[3])}KM/L)" for v in vehicles]
    sel = st.selectbox("수정할 차량 선택", labels, index=0)
    vehicle_id = int(re.search(r"\[(\d+)\]", sel).group(1))
    cur_v = next(v for v in vehicles if int(v[0]) == vehicle_id)

    with st.form("vehicle_edit_form"):
        name = st.text_input("차량 종류", value=cur_v[1])
        fuel_type = st.selectbox("유종", ["휘발유", "경유", "LPG"], index=["휘발유","경유","LPG"].index(cur_v[2]) if cur_v[2] in ["휘발유","경유","LPG"] else 0)
        fuel_eff = st.number_input("연비 (KM/L)", min_value=1, max_value=100, value=iround(cur_v[3]) or 12, step=1, format="%d")
        if st.form_submit_button("차량 수정 저장"):
            update_vehicle(USER_ID, vehicle_id, name, fuel_type, int(fuel_eff))
            st.success("수정 완료!")
            st.rerun()

    st.divider()
    st.markdown('<div class="card"><h3>🗑 차량 삭제</h3><p class="muted">삭제하면 해당 차량의 운행 내역도 함께 삭제됩니다.</p></div>', unsafe_allow_html=True)
    confirm = st.checkbox("삭제에 동의합니다(되돌릴 수 없음)")
    if st.button("차량 삭제"):
        if not confirm:
            st.error("체크박스로 확인 후 진행해줘.")
        else:
            delete_vehicle_cascade(USER_ID, vehicle_id)
            st.success("삭제 완료!")
            st.rerun()

# ============================================================
# PAGE: 관리자
# ============================================================
elif st.session_state.page == "관리자":
    if ROLE != "admin":
        st.error("권한이 없습니다.")
        st.stop()

    st.markdown('<div class="card"><h2>🛠 관리자</h2><p class="muted">관리자만 접근 가능합니다.</p></div>', unsafe_allow_html=True)
    tabA, tabB = st.tabs(["사용자 관리", "전체 운행 내역"])

    with tabA:
        udf = admin_list_users()
        st.dataframe(udf, width="stretch", hide_index=True)

        st.divider()
        st.subheader("비밀번호 초기화")
        uid = st.number_input("대상 user_id", min_value=1, step=1)
        new_pw = st.text_input("새 비밀번호(임시)", type="password")
        if st.button("비밀번호 재설정"):
            if len(new_pw) < 6:
                st.error("비밀번호는 6자리 이상.")
            else:
                admin_reset_user_password(int(uid), new_pw)
                st.success("재설정 완료")

        st.divider()
        st.subheader("사용자 삭제(차량/운행/토큰 포함 전부 삭제)")
        del_uid = st.number_input("삭제할 user_id", min_value=1, step=1, key="del_uid")
        confirm = st.checkbox("정말 삭제할게요(되돌릴 수 없음)")
        if st.button("사용자 삭제"):
            if not confirm:
                st.error("체크박스 확인 후 진행해줘.")
            elif int(del_uid) == USER_ID:
                st.error("현재 로그인한 관리자 본인은 삭제할 수 없게 막았어.")
            else:
                admin_delete_user(int(del_uid))
                st.success("삭제 완료")
                st.rerun()

    with tabB:
        today = date.today()
        start = st.date_input("시작일", value=today - timedelta(days=30), key="admin_start")
        end = st.date_input("종료일", value=today, key="admin_end")
        df = admin_all_trips(start, end)
        if df.empty:
            st.write("해당 기간 데이터 없음")
        else:
            view = df.copy()
            view["total_distance_km"] = view["total_distance_km"].apply(fmt_km)
            view["fare_krw"] = view["fare_krw"].apply(fmt_won)
            view["total_cost_krw"] = view["total_cost_krw"].apply(fmt_won)
            view["profit_krw"] = view["profit_krw"].apply(fmt_won)
            view["profit_margin_pct"] = view["profit_margin_pct"].apply(fmt_pct)

            view.rename(columns={
                "id": "번호",
                "trip_date": "운행일자",
                "username": "사용자",
                "vehicle_name": "차량",
                "trip_type": "형태",
                "total_distance_km": "총거리",
                "fare_krw": "운임료",
                "total_cost_krw": "총비용",
                "profit_krw": "순이익",
                "profit_margin_pct": "수익률",
                "origin_text": "출발지",
                "dest_text": "도착지",
                "route_mode": "경로옵션",
                "created_at": "등록시각",
            }, inplace=True)

            st.dataframe(view, width="stretch", hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("CSV 다운로드(원본)", data=csv, file_name="admin_all_trips.csv", mime="text/csv")
