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
        .block-container {padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1200px;}
        h1,h2,h3 {letter-spacing:-0.2px;}
        section[data-testid="stSidebar"] {background: #fbfbfd;}
        section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
        .card{
          background:#fff;border:1px solid rgba(0,0,0,.06);
          border-radius:16px;padding:14px 16px;margin-bottom:12px;
          box-shadow:0 1px 10px rgba(0,0,0,.04);
        }
        .muted{color:rgba(0,0,0,.55);}
        .pill{
          display:inline-block;padding:4px 10px;border-radius:999px;
          background:rgba(0,0,0,.06);font-size:12px;margin-left:6px;
        }
        .stButton>button{border-radius:12px;padding:.65rem .95rem;font-weight:800;}
        .stTextInput input,.stSelectbox>div>div{border-radius:12px !important;}
        div[data-testid="stDataFrame"]{border-radius:14px; overflow:hidden; border:1px solid rgba(0,0,0,.06);}
        @media (max-width: 640px){
          .block-container{padding-left:.8rem; padding-right:.8rem;}
          .stButton>button{width:100%;}
        }
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

def parse_int(s: str | None) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    s = re.sub(r"[^\d\-]", "", s)
    try:
        return int(s)
    except Exception:
        return 0

def fmt_unit(n: int, unit: str) -> str:
    return f"{n:,}{unit}"

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

def unit_formatter(key: str, unit: str, edited_flag: str | None = None):
    def _cb():
        n = parse_int(st.session_state.get(key, ""))
        st.session_state[key] = fmt_unit(n, unit)
        if edited_flag:
            st.session_state[edited_flag] = True
    return _cb

# ============================================================
# Security (PBKDF2)
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
        algo, iters, salt_hex, _ = stored.split("$", 3)
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

def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cur.fetchall()}

def migrate_trips_table_if_needed(conn: sqlite3.Connection):
    """
    Streamlit Cloud에 예전 스키마(trips)가 남아 있을 때 자동 업그레이드.
    """
    try:
        cols = _table_columns(conn, "trips")
    except Exception:
        return

    required = {
        "paid_oneway_km", "empty_oneway_km", "total_km",
        "fare_krw", "fuel_price_krw_per_l", "toll_krw", "parking_krw", "other_krw",
        "fuel_used_l", "fuel_cost_krw", "total_cost_krw", "profit_krw", "profit_pct",
        "origin_text", "dest_text", "route_mode",
    }
    if required.issubset(cols):
        return  # already new schema

    cur = conn.cursor()

    # create v2
    cur.execute("""
      CREATE TABLE IF NOT EXISTS trips_v2(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        trip_date TEXT NOT NULL,
        vehicle_id INTEGER NOT NULL,
        trip_type TEXT NOT NULL,

        paid_oneway_km REAL NOT NULL,
        empty_oneway_km REAL NOT NULL,
        total_km REAL NOT NULL,

        fare_krw REAL NOT NULL,
        fuel_price_krw_per_l REAL NOT NULL,
        toll_krw REAL NOT NULL,
        parking_krw REAL NOT NULL,
        other_krw REAL NOT NULL,

        fuel_used_l REAL NOT NULL,
        fuel_cost_krw REAL NOT NULL,
        total_cost_krw REAL NOT NULL,
        profit_krw REAL NOT NULL,
        profit_pct REAL NOT NULL,

        origin_text TEXT,
        dest_text TEXT,
        route_mode TEXT,

        created_at TEXT NOT NULL
      )
    """)

    def pick(colname: str, fallback: str) -> str:
        return colname if colname in cols else fallback

    paid_col = pick("paid_oneway_km", pick("paid_distance_km", "0"))
    empty_col = pick("empty_oneway_km", pick("empty_distance_km", "0"))
    total_col = pick("total_km", pick("total_distance_km", "0"))
    other_col = pick("other_krw", pick("other_cost_krw", "0"))
    profitpct_col = pick("profit_pct", pick("profit_margin_pct", "0"))

    origin_col = pick("origin_text", "''")
    dest_col = pick("dest_text", "''")
    route_col = pick("route_mode", "''")

    cur.execute(f"""
      INSERT INTO trips_v2(
        user_id, trip_date, vehicle_id, trip_type,
        paid_oneway_km, empty_oneway_km, total_km,
        fare_krw, fuel_price_krw_per_l, toll_krw, parking_krw, other_krw,
        fuel_used_l, fuel_cost_krw, total_cost_krw, profit_krw, profit_pct,
        origin_text, dest_text, route_mode,
        created_at
      )
      SELECT
        {pick("user_id","1")} as user_id,
        {pick("trip_date", pick("created_at","datetime('now')"))} as trip_date,
        {pick("vehicle_id","1")} as vehicle_id,
        {pick("trip_type","'편도'")} as trip_type,

        {paid_col} as paid_oneway_km,
        {empty_col} as empty_oneway_km,
        {total_col} as total_km,

        {pick("fare_krw","0")} as fare_krw,
        {pick("fuel_price_krw_per_l","0")} as fuel_price_krw_per_l,
        {pick("toll_krw","0")} as toll_krw,
        {pick("parking_krw","0")} as parking_krw,
        {other_col} as other_krw,

        {pick("fuel_used_l","0")} as fuel_used_l,
        {pick("fuel_cost_krw","0")} as fuel_cost_krw,
        {pick("total_cost_krw","0")} as total_cost_krw,
        {pick("profit_krw","0")} as profit_krw,
        {profitpct_col} as profit_pct,

        {origin_col} as origin_text,
        {dest_col} as dest_text,
        {route_col} as route_mode,

        {pick("created_at","datetime('now')")} as created_at
      FROM trips
    """)

    cur.execute("DROP TABLE trips")
    cur.execute("ALTER TABLE trips_v2 RENAME TO trips")
    conn.commit()

# ============================================================
# Fuel daily (OPINET) best-effort
# ============================================================
def refresh_fuel_prices_daily_if_needed():
    today = date.today().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM fuel_prices_daily WHERE price_date=?", (today,))
    cnt = cur.fetchone()[0]
    conn.close()
    if cnt >= 2:
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    prices = {}

    try:
        r = requests.get("https://www.opinet.co.kr/user/dopospdrg/dopOsPdrgAreaView.do", headers=headers, timeout=10)
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
                    prices["휘발유"] = g
                    prices["경유"] = d
    except Exception:
        pass

    try:
        r = requests.get("https://www.opinet.co.kr/user/dopvsavsel/dopVsAvselSelect.do", headers=headers, timeout=10)
        if r.status_code == 200:
            html = r.text
            m = re.search(r"(\d{3,4}\.\d+)", html)
            if m:
                prices["LPG"] = float(m.group(1))
    except Exception:
        pass

    if not prices:
        return

    conn = get_conn()
    cur = conn.cursor()
    for ft, p in prices.items():
        cur.execute(
            """
            INSERT INTO fuel_prices_daily(price_date,fuel_type,price_krw_per_l,source,fetched_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(price_date,fuel_type) DO UPDATE SET
              price_krw_per_l=excluded.price_krw_per_l,
              source=excluded.source,
              fetched_at=excluded.fetched_at
            """,
            (today, ft, float(p), "OPINET", datetime.now().isoformat(timespec="seconds"))
        )
    conn.commit()
    conn.close()

def latest_fuel_price(fuel_type: str):
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

# ============================================================
# DB init
# ============================================================
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
      CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        pw_hash TEXT NOT NULL,
        recovery_hash TEXT,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TEXT NOT NULL
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS auth_tokens(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS vehicles(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        fuel_type TEXT NOT NULL,
        fuel_eff_km_per_l REAL NOT NULL,
        created_at TEXT NOT NULL
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS fuel_prices_daily(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        price_date TEXT NOT NULL,
        fuel_type TEXT NOT NULL,
        price_krw_per_l REAL NOT NULL,
        source TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        UNIQUE(price_date, fuel_type)
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS trips(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        trip_date TEXT NOT NULL,
        vehicle_id INTEGER NOT NULL,
        trip_type TEXT NOT NULL,

        paid_oneway_km REAL NOT NULL,
        empty_oneway_km REAL NOT NULL,
        total_km REAL NOT NULL,

        fare_krw REAL NOT NULL,
        fuel_price_krw_per_l REAL NOT NULL,
        toll_krw REAL NOT NULL,
        parking_krw REAL NOT NULL,
        other_krw REAL NOT NULL,

        fuel_used_l REAL NOT NULL,
        fuel_cost_krw REAL NOT NULL,
        total_cost_krw REAL NOT NULL,
        profit_krw REAL NOT NULL,
        profit_pct REAL NOT NULL,

        origin_text TEXT,
        dest_text TEXT,
        route_mode TEXT,

        created_at TEXT NOT NULL
      )
    """)

    # migrate if old schema exists
    migrate_trips_table_if_needed(conn)

    conn.commit()

    # ensure admin exists
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO users(username,pw_hash,recovery_hash,role,created_at) VALUES(?,?,?,?,?)",
            ("admin", _pbkdf2_hash("admin1234"), _pbkdf2_hash("000000"), "admin", datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
    else:
        cur.execute("UPDATE users SET role='admin' WHERE username='admin'")
        conn.commit()

    conn.close()

# ============================================================
# Auth (users + tokens)
# ============================================================
def get_user(username: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, pw_hash, recovery_hash, role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row

def create_user(username: str, password: str, recovery_code: str):
    username = (username or "").strip()
    if not username:
        return False, "아이디를 입력해줘."
    if len(password) < 6:
        return False, "비밀번호는 6자리 이상."
    rc = _normalize_recovery_code(recovery_code)
    if len(rc) < 4:
        return False, "복구코드는 숫자 4자리 이상."
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users(username,pw_hash,recovery_hash,role,created_at) VALUES(?,?,?,?,?)",
            (username, _pbkdf2_hash(password), _pbkdf2_hash(rc), "user", datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()
        conn.close()
        return True, "회원가입 완료! 이제 로그인해줘."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "이미 존재하는 아이디야."
    except Exception as e:
        conn.close()
        return False, f"회원가입 실패: {e}"

def reset_password(username: str, recovery_code: str, new_password: str):
    username = (username or "").strip()
    if len(new_password) < 6:
        return False, "새 비밀번호는 6자리 이상."
    rc = _normalize_recovery_code(recovery_code)
    row = get_user(username)
    if not row:
        return False, "아이디가 없어요."
    uid, _, _, recovery_hash, _ = row
    if not recovery_hash or not _verify_pbkdf2(rc, recovery_hash):
        return False, "복구코드가 틀렸어요."
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET pw_hash=? WHERE id=?", (_pbkdf2_hash(new_password), uid))
    conn.commit()
    conn.close()
    return True, "비밀번호 재설정 완료!"

def get_user_info(user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, role FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return ("user", "user")
    return row[0], row[1]

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
        "INSERT INTO auth_tokens(user_id, token_hash, created_at, expires_at) VALUES(?,?,?,?)",
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
    cur.execute("SELECT user_id FROM auth_tokens WHERE token_hash=? AND expires_at >= ? LIMIT 1", (th, now))
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
# Vehicles / trips
# ============================================================
def list_vehicles_df(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT id,name,fuel_type,fuel_eff_km_per_l,created_at FROM vehicles WHERE user_id=? ORDER BY id DESC",
        conn,
        params=(user_id,)
    )
    conn.close()
    return df

def add_vehicle(user_id: int, name: str, fuel_type: str, eff: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO vehicles(user_id,name,fuel_type,fuel_eff_km_per_l,created_at) VALUES(?,?,?,?,?)",
        (user_id, name.strip(), fuel_type, float(eff), datetime.now().isoformat(timespec="seconds"))
    )
    conn.commit()
    conn.close()

def update_vehicle(user_id: int, vid: int, name: str, fuel_type: str, eff: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE vehicles SET name=?, fuel_type=?, fuel_eff_km_per_l=? WHERE user_id=? AND id=?",
        (name.strip(), fuel_type, float(eff), user_id, vid)
    )
    conn.commit()
    conn.close()

def delete_vehicle_cascade(user_id: int, vid: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM trips WHERE user_id=? AND vehicle_id=?", (user_id, vid))
    cur.execute("DELETE FROM vehicles WHERE user_id=? AND id=?", (user_id, vid))
    conn.commit()
    conn.close()

def save_trip(user_id: int, vehicle_row: dict, trip_date: date, trip_type: str,
             paid_oneway: int, empty_oneway: int, fare: int, fuel_price: int,
             toll: int, parking: int, other: int, origin_text: str, dest_text: str, route_mode: str):
    mult = 2 if trip_type == "왕복" else 1
    total_km = (paid_oneway + empty_oneway) * mult
    eff = float(vehicle_row["fuel_eff_km_per_l"])
    fuel_used = (total_km / eff) if eff > 0 else 0
    fuel_cost = fuel_used * fuel_price
    total_cost = fuel_cost + toll + parking + other
    profit = fare - total_cost
    pct = (profit / fare * 100) if fare > 0 else 0

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO trips(
        user_id,trip_date,vehicle_id,trip_type,
        paid_oneway_km,empty_oneway_km,total_km,
        fare_krw,fuel_price_krw_per_l,toll_krw,parking_krw,other_krw,
        fuel_used_l,fuel_cost_krw,total_cost_krw,profit_krw,profit_pct,
        origin_text,dest_text,route_mode,
        created_at
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, trip_date.isoformat(), int(vehicle_row["id"]), trip_type,
        float(paid_oneway), float(empty_oneway), float(total_km),
        float(fare), float(fuel_price), float(toll), float(parking), float(other),
        float(fuel_used), float(fuel_cost), float(total_cost), float(profit), float(pct),
        origin_text, dest_text, route_mode,
        datetime.now().isoformat(timespec="seconds")
    ))
    conn.commit()
    conn.close()
    return total_km, fuel_used, fuel_cost, total_cost, profit, pct

def trips_report(user_id: int, start: date, end: date, vehicle_id: int | None):
    conn = get_conn()
    params = {"uid": user_id, "s": start.isoformat(), "e": end.isoformat()}
    where = "t.user_id=:uid AND t.trip_date>=:s AND t.trip_date<=:e"
    if vehicle_id is not None:
        where += " AND t.vehicle_id=:vid"
        params["vid"] = vehicle_id
    df = pd.read_sql_query(f"""
      SELECT
        t.id, t.trip_date, v.name AS vehicle_name, t.trip_type,
        t.paid_oneway_km, t.empty_oneway_km, t.total_km,
        t.fare_krw, t.fuel_price_krw_per_l, t.fuel_cost_krw,
        t.toll_krw, t.parking_krw, t.other_krw,
        t.total_cost_krw, t.profit_krw, t.profit_pct,
        t.origin_text, t.dest_text, t.route_mode,
        t.created_at
      FROM trips t
      JOIN vehicles v ON v.id=t.vehicle_id
      WHERE {where}
      ORDER BY t.trip_date DESC, t.id DESC
    """, conn, params=params)
    conn.close()
    return df

def admin_list_users():
    conn = get_conn()
    df = pd.read_sql_query("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC", conn)
    conn.close()
    return df

# ============================================================
# Kakao
# ============================================================
KAKAO_LOCAL_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_LOCAL_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_NAVI_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"

@st.cache_data(ttl=120)
def kakao_search_places(query: str, size_address: int = 6, size_keyword: int = 10):
    if not get_kakao_key():
        return []
    q = (query or "").strip()
    if not q:
        return []
    results = []
    try:
        r = requests.get(KAKAO_LOCAL_ADDRESS_URL, headers=_kakao_headers(), params={"query": q, "size": int(size_address)}, timeout=10)
        if r.status_code == 200:
            docs = (r.json() or {}).get("documents", []) or []
            for d in docs:
                x = d.get("x"); y = d.get("y")
                road = (d.get("road_address") or {}).get("address_name") if d.get("road_address") else ""
                jibun = (d.get("address") or {}).get("address_name") if d.get("address") else ""
                label = road or jibun or q
                results.append({"x": x, "y": y, "place_name": label, "road_address_name": road or "", "address_name": jibun or ""})
    except Exception:
        pass
    try:
        r = requests.get(KAKAO_LOCAL_KEYWORD_URL, headers=_kakao_headers(), params={"query": q, "size": int(size_keyword)}, timeout=10)
        if r.status_code == 200:
            docs = (r.json() or {}).get("documents", []) or []
            for d in docs:
                results.append(d)
    except Exception:
        pass

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
    return merged

@st.cache_data(ttl=120)
def kakao_route(origin_lng: float, origin_lat: float, dest_lng: float, dest_lat: float, priority: str, avoid: str | None):
    if not get_kakao_key():
        return None
    params = {"origin": f"{origin_lng},{origin_lat}", "destination": f"{dest_lng},{dest_lat}", "priority": priority}
    if avoid:
        params["avoid"] = avoid
    try:
        r = requests.get(KAKAO_NAVI_DIRECTIONS_URL, headers=_kakao_headers(), params=params, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
        routes = j.get("routes", [])
        if not routes:
            return None
        summary = routes[0].get("summary", {}) or {}
        fare = summary.get("fare", {}) or {}
        dist = int(summary.get("distance", 0) or 0)
        dur = int(summary.get("duration", 0) or 0)
        toll = int(fare.get("toll", 0) or 0)
        return {"distance_m": dist, "duration_s": dur, "toll_krw": toll}
    except Exception:
        return None

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
# App start
# ============================================================
st.set_page_config(page_title="운행손익", page_icon="🚗", layout="wide")
inject_css()
init_db()
refresh_fuel_prices_daily_if_needed()

# session basics
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "page" not in st.session_state:
    st.session_state.page = "운행 입력"
if "selected_vehicle_id" not in st.session_state:
    st.session_state.selected_vehicle_id = None

# login persistence via URL token
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

# login/signup/reset UI
def login_screen():
    st.markdown('<div class="card"><h2>🔐 로그인</h2><p class="muted">회원가입 / 비밀번호 찾기 가능</p></div>', unsafe_allow_html=True)
    tab_login, tab_signup, tab_reset = st.tabs(["로그인", "회원가입", "비밀번호 찾기"])

    with tab_login:
        u = st.text_input("아이디", key="login_user")
        p = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인"):
            row = get_user((u or "").strip())
            if not row:
                st.error("아이디가 없어요.")
                return
            uid, uname, pw_hash, _rec, role = row
            if _verify_pbkdf2(p, pw_hash):
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
        nu = st.text_input("새 아이디", key="su_user")
        npw = st.text_input("새 비밀번호(6자리 이상)", type="password", key="su_pw")
        npw2 = st.text_input("새 비밀번호 확인", type="password", key="su_pw2")
        rc = st.text_input("복구코드(숫자 4자리 이상)", type="password", key="su_rc")
        if st.button("회원가입"):
            if npw != npw2:
                st.error("비밀번호가 서로 달라요.")
            else:
                ok, msg = create_user(nu, npw, rc)
                (st.success if ok else st.error)(msg)

    with tab_reset:
        ru = st.text_input("아이디", key="rs_user")
        rrc = st.text_input("복구코드", type="password", key="rs_rc")
        rnp = st.text_input("새 비밀번호(6자리 이상)", type="password", key="rs_np")
        rnp2 = st.text_input("새 비밀번호 확인", type="password", key="rs_np2")
        if st.button("비밀번호 재설정"):
            if rnp != rnp2:
                st.error("새 비밀번호가 서로 달라요.")
            else:
                ok, msg = reset_password(ru, rrc, rnp)
                (st.success if ok else st.error)(msg)

if not st.session_state.user_id:
    login_screen()
    st.stop()

USER_ID = int(st.session_state.user_id)
USERNAME = st.session_state.username or "user"
ROLE = st.session_state.role or "user"

# Sidebar (logout + vehicle)
with st.sidebar:
    st.markdown(f"### 👤 {USERNAME} <span class='pill'>{'관리자' if ROLE=='admin' else '사용자'}</span>", unsafe_allow_html=True)
    if st.button("로그아웃"):
        do_logout()

    vdf = list_vehicles_df(USER_ID)
    if not vdf.empty:
        labels = [f"[{int(r.id)}] {r.name} ({r.fuel_type}, {iround(r.fuel_eff_km_per_l)}KM/L)" for r in vdf.itertuples(index=False)]
        chosen = st.selectbox("기본 차량", labels, index=0, key="veh_sel")
        st.session_state.selected_vehicle_id = int(re.search(r"\[(\d+)\]", chosen).group(1))
    else:
        st.session_state.selected_vehicle_id = None

# =========================
# TOP NAV BAR
# =========================
menu = ["차량 등록", "운행 입력", "내역/리포트", "개인정보변경"]
if ROLE == "admin":
    menu.append("관리자")
st.session_state.page = st.radio("메뉴", menu, horizontal=True, index=menu.index(st.session_state.page) if st.session_state.page in menu else 1)

# =========================
# Pages
# =========================
if st.session_state.page == "차량 등록":
    st.markdown('<div class="card"><h2>🚗 차량 등록</h2></div>', unsafe_allow_html=True)
    with st.form("veh_add", clear_on_submit=True):
        name = st.text_input("차량 종류")
        fuel = st.selectbox("유종", ["휘발유", "경유", "LPG"])
        eff = st.number_input("연비(KM/L)", min_value=1, max_value=100, value=12, step=1, format="%d")
        if st.form_submit_button("등록"):
            if not name.strip():
                st.error("차량 종류를 입력해줘.")
            else:
                add_vehicle(USER_ID, name, fuel, int(eff))
                st.success("등록 완료!")
                st.rerun()

elif st.session_state.page == "운행 입력":
    if vdf.empty:
        st.warning("차량을 먼저 등록해줘.")
        st.stop()

    vid = st.session_state.selected_vehicle_id or int(vdf.iloc[0]["id"])
    vehicle_row = vdf[vdf["id"] == vid].iloc[0].to_dict()

    auto_p, auto_d, auto_s = latest_fuel_price(vehicle_row["fuel_type"])
    auto_int = iround(auto_p) if auto_p is not None else 1700

    # init input states if missing
    for k, v in {
        "origin_mode": "출발지 주소/장소명",
        "origin_query": "",
        "dest_query": "",
        "trip_type": "편도",
        "paid_oneway_km_txt": "0KM",
        "empty_oneway_km_txt": "0KM",
        "fare_krw_txt": "30,000원",
        "fuel_price_txt": f"{auto_int:,}원/L",
        "fuel_user_edited": False,
        "toll_krw_txt": "0원",
        "toll_user_edited": False,
        "parking_krw_txt": "0원",
        "other_cost_krw_txt": "0원",
        "_geo": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state["fuel_user_edited"]:
        st.session_state["fuel_price_txt"] = f"{auto_int:,}원/L"

    st.markdown(
        f"<div class='card'><h2>💰 운행 입력</h2>"
        f"<div class='muted'>차량: <b>{vehicle_row['name']}</b> ({vehicle_row['fuel_type']}, 연비 {iround(vehicle_row['fuel_eff_km_per_l'])}KM/L)"
        f"<br/>유가(전국 평균 자동): <b>{auto_int:,}원/L</b> ({auto_d or '-'} / {auto_s or 'OPINET'})</div></div>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="card"><h3>📍 출발지/도착지</h3><p class="muted">검색 리스트 → 선택 → 거리/톨비 계산</p></div>', unsafe_allow_html=True)

    origin_mode = st.selectbox("출발지 방식", ["현재 위치", "출발지 주소/장소명"], index=1)
    st.text_input("출발지 입력", key="origin_query", disabled=(origin_mode == "현재 위치"))
    st.text_input("도착지 입력", key="dest_query")
    route_mode = st.selectbox("경로 옵션", ["추천", "최단시간", "최단거리", "무료도로 우선"], index=0)

    origin_doc = None
    dest_doc = None

    if get_kakao_key():
        if origin_mode == "출발지 주소/장소명" and st.session_state.origin_query.strip():
            origin_results = kakao_search_places(st.session_state.origin_query.strip())
            olabels = ["(선택 안 함)"]
            for d in origin_results[:12]:
                place = (d.get("place_name") or "").strip()
                road = (d.get("road_address_name") or "").strip()
                jibun = (d.get("address_name") or "").strip()
                olabels.append(f"{place} | 도로명: {road or '-'} | 지번: {jibun or '-'}")
            opick = st.selectbox("출발지 검색 결과", olabels, index=0)
            if opick != "(선택 안 함)":
                origin_doc = origin_results[:12][olabels.index(opick) - 1]

        if st.session_state.dest_query.strip():
            dest_results = kakao_search_places(st.session_state.dest_query.strip())
            dlabels = ["(선택 안 함)"]
            for d in dest_results[:12]:
                place = (d.get("place_name") or "").strip()
                road = (d.get("road_address_name") or "").strip()
                jibun = (d.get("address_name") or "").strip()
                dlabels.append(f"{place} | 도로명: {road or '-'} | 지번: {jibun or '-'}")
            dpick = st.selectbox("도착지 검색 결과", dlabels, index=0)
            if dpick != "(선택 안 함)":
                dest_doc = dest_results[:12][dlabels.index(dpick) - 1]

        colA, colB = st.columns(2)
        with colA:
            if st.button("현재 위치 가져오기"):
                st.session_state._geo = get_browser_geolocation()
        with colB:
            if st.button("거리/톨비 계산"):
                origin_lng = origin_lat = None
                if origin_mode == "현재 위치":
                    geo = st.session_state.get("_geo")
                    if not (isinstance(geo, dict) and geo.get("lat") and geo.get("lng")):
                        st.error("먼저 '현재 위치 가져오기'로 권한 허용해줘.")
                    else:
                        origin_lat = float(geo["lat"]); origin_lng = float(geo["lng"])
                else:
                    if not origin_doc:
                        st.error("출발지 검색 결과에서 하나를 선택해줘.")
                    else:
                        origin_lng = float(origin_doc["x"]); origin_lat = float(origin_doc["y"])

                if not dest_doc:
                    st.error("도착지 검색 결과에서 하나를 선택해줘.")
                elif origin_lng is not None and origin_lat is not None:
                    dest_lng = float(dest_doc["x"]); dest_lat = float(dest_doc["y"])

                    if route_mode == "추천":
                        priority, avoid = "RECOMMEND", None
                    elif route_mode == "최단시간":
                        priority, avoid = "TIME", None
                    elif route_mode == "최단거리":
                        priority, avoid = "DISTANCE", None
                    else:
                        priority, avoid = "RECOMMEND", "toll"

                    res = kakao_route(origin_lng, origin_lat, dest_lng, dest_lat, priority=priority, avoid=avoid)
                    if not res:
                        st.error("길찾기 실패")
                    else:
                        km_oneway = iround(res["distance_m"] / 1000.0)
                        toll_oneway = iround(res["toll_krw"])
                        st.session_state["paid_oneway_km_txt"] = f"{km_oneway:,}KM"
                        if not st.session_state["toll_user_edited"]:
                            st.session_state["toll_krw_txt"] = f"{toll_oneway:,}원"
                        st.success(f"거리(편도): {km_oneway:,}KM / 톨비(편도): {toll_oneway:,}원")

    else:
        st.warning("Streamlit Cloud Secrets에 KAKAO_REST_API_KEY가 없으면 검색/길찾기가 동작하지 않습니다.")

    st.markdown('<div class="card"><h3>🧾 운행 정보 입력</h3><p class="muted">입력칸 안에 단위가 자동으로 붙습니다.</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        trip_date = st.date_input("운행 날짜", value=date.today())
        st.session_state["trip_type"] = st.selectbox("운행 형태", ["편도", "왕복"], index=0 if st.session_state["trip_type"] == "편도" else 1)
    with col2:
        st.text_input("유상거리(편도)", key="paid_oneway_km_txt", on_change=unit_formatter("paid_oneway_km_txt", "KM"))
        st.text_input("공차거리(편도)", key="empty_oneway_km_txt", on_change=unit_formatter("empty_oneway_km_txt", "KM"))
    with col3:
        st.text_input("운임료", key="fare_krw_txt", on_change=unit_formatter("fare_krw_txt", "원"))

    col4, col5, col6, col7 = st.columns(4)
    with col4:
        st.text_input("유가(자동 기본, 수정 가능)", key="fuel_price_txt", on_change=unit_formatter("fuel_price_txt", "원/L", "fuel_user_edited"))
    with col5:
        st.text_input("톨비(자동 기본, 수정 가능)", key="toll_krw_txt", on_change=unit_formatter("toll_krw_txt", "원", "toll_user_edited"))
    with col6:
        st.text_input("주차비", key="parking_krw_txt", on_change=unit_formatter("parking_krw_txt", "원"))
    with col7:
        st.text_input("기타비용", key="other_cost_krw_txt", on_change=unit_formatter("other_cost_krw_txt", "원"))

    paid = parse_int(st.session_state["paid_oneway_km_txt"])
    empty = parse_int(st.session_state["empty_oneway_km_txt"])
    fare = parse_int(st.session_state["fare_krw_txt"])
    fuel_price = parse_int(st.session_state["fuel_price_txt"])
    toll = parse_int(st.session_state["toll_krw_txt"])
    parking = parse_int(st.session_state["parking_krw_txt"])
    other = parse_int(st.session_state["other_cost_krw_txt"])

    mult = 2 if st.session_state["trip_type"] == "왕복" else 1
    total_km = (paid + empty) * mult
    eff = float(vehicle_row["fuel_eff_km_per_l"])
    fuel_used = (total_km / eff) if eff > 0 else 0
    fuel_cost = fuel_used * fuel_price
    total_cost = fuel_cost + toll + parking + other
    profit = fare - total_cost

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("예상 총거리", fmt_km(total_km))
    k2.metric("예상 기름값", fmt_won(fuel_cost))
    k3.metric("예상 총비용", fmt_won(total_cost))
    k4.metric("예상 순이익", fmt_won(profit))

    if st.button("저장"):
        if (paid <= 0 and empty <= 0) or fare <= 0 or fuel_price <= 0:
            st.error("거리/운임료/유가를 확인해줘.")
        else:
            save_trip(
                USER_ID, vehicle_row, trip_date, st.session_state["trip_type"],
                paid, empty, fare, fuel_price, toll, parking, other,
                origin_text=st.session_state.get("origin_query","").strip(),
                dest_text=st.session_state.get("dest_query","").strip(),
                route_mode=route_mode
            )
            st.success("저장 완료!")
            st.rerun()

elif st.session_state.page == "내역/리포트":
    st.markdown('<div class="card"><h2>📊 내역/리포트</h2></div>', unsafe_allow_html=True)

    if vdf.empty:
        st.info("차량을 먼저 등록해줘.")
        st.stop()

    vlabels = ["전체 차량"] + [f"[{int(r.id)}] {r.name} ({r.fuel_type})" for r in vdf.itertuples(index=False)]
    vsel = st.selectbox("차량 필터", vlabels, index=0)
    vehicle_id = None
    if vsel != "전체 차량":
        vehicle_id = int(re.search(r"\[(\d+)\]", vsel).group(1))

    today = date.today()
    start = st.date_input("시작일", value=today - timedelta(days=30))
    end = st.date_input("종료일", value=today)

    df = trips_report(USER_ID, start, end, vehicle_id)
    if df.empty:
        st.info("해당 기간 데이터가 없어.")
        st.stop()

    df2 = df.copy()
    df2["d"] = pd.to_datetime(df2["trip_date"]).dt.date
    chart = df2.groupby("d", as_index=False).agg(fare=("fare_krw","sum"), cost=("total_cost_krw","sum"), profit=("profit_krw","sum")).set_index("d")
    st.line_chart(chart[["fare","cost","profit"]])

    view = df.copy()
    view.rename(columns={
        "id":"번호","trip_date":"운행일자","vehicle_name":"차량","trip_type":"형태",
        "paid_oneway_km":"유상거리(편도)","empty_oneway_km":"공차거리(편도)","total_km":"총거리",
        "fare_krw":"운임료","fuel_price_krw_per_l":"유가","fuel_cost_krw":"기름값",
        "toll_krw":"톨비","parking_krw":"주차비","other_krw":"기타비용",
        "total_cost_krw":"총비용","profit_krw":"순이익","profit_pct":"수익률",
        "origin_text":"출발지","dest_text":"도착지","route_mode":"경로옵션",
        "created_at":"등록시각"
    }, inplace=True)

    for c in ["유상거리(편도)","공차거리(편도)","총거리"]:
        view[c] = view[c].apply(fmt_km)
    for c in ["운임료","총비용","기름값","톨비","주차비","기타비용","순이익"]:
        view[c] = view[c].apply(fmt_won)
    view["유가"] = view["유가"].apply(fmt_won_per_l)
    view["수익률"] = view["수익률"].apply(fmt_pct)

    def highlight_negative(row):
        styles = [""] * len(row)
        v = parse_int(row.get("순이익","0"))
        if v < 0 and "순이익" in row.index:
            idx = list(row.index).index("순이익")
            styles[idx] = "color:#d00;font-weight:800;"
        return styles

    st.dataframe(view.style.apply(highlight_negative, axis=1), width="stretch", hide_index=True)

elif st.session_state.page == "개인정보변경":
    st.markdown('<div class="card"><h2>👤 개인정보변경</h2><p class="muted">차량 수정/삭제 + 비밀번호 변경</p></div>', unsafe_allow_html=True)

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
                stored = get_conn().execute("SELECT pw_hash FROM users WHERE id=?", (USER_ID,)).fetchone()
                if not stored or not _verify_pbkdf2(old_pw, stored[0]):
                    st.error("현재 비밀번호가 틀렸어요.")
                else:
                    conn = get_conn()
                    conn.execute("UPDATE users SET pw_hash=? WHERE id=?", (_pbkdf2_hash(new_pw), USER_ID))
                    conn.commit()
                    conn.close()
                    st.success("변경 완료!")

    if vdf.empty:
        st.info("등록된 차량이 없어요.")
        st.stop()

    labels = [f"[{int(r.id)}] {r.name} ({r.fuel_type}, {iround(r.fuel_eff_km_per_l)}KM/L)" for r in vdf.itertuples(index=False)]
    sel = st.selectbox("차량 선택", labels, index=0)
    vid = int(re.search(r"\[(\d+)\]", sel).group(1))
    row = vdf[vdf["id"] == vid].iloc[0]

    with st.form("veh_edit"):
        name = st.text_input("차량 종류", value=row["name"])
        fuel = st.selectbox("유종", ["휘발유","경유","LPG"], index=["휘발유","경유","LPG"].index(row["fuel_type"]))
        eff = st.number_input("연비(KM/L)", min_value=1, max_value=100, value=iround(row["fuel_eff_km_per_l"]), step=1, format="%d")
        if st.form_submit_button("차량 수정 저장"):
            update_vehicle(USER_ID, vid, name, fuel, int(eff))
            st.success("수정 완료!")
            st.rerun()

    st.divider()
    confirm = st.checkbox("차량 삭제에 동의합니다(되돌릴 수 없음)")
    if st.button("차량 삭제(운행 포함)"):
        if not confirm:
            st.error("체크박스 확인 후 진행해줘.")
        else:
            delete_vehicle_cascade(USER_ID, vid)
            st.success("삭제 완료!")
            st.rerun()

else:
    if ROLE != "admin":
        st.error("권한이 없습니다.")
        st.stop()
    st.markdown('<div class="card"><h2>🛠 관리자</h2></div>', unsafe_allow_html=True)
    st.dataframe(admin_list_users(), width="stretch", hide_index=True)
