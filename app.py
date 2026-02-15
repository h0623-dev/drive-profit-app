# app.py
import streamlit as st
import sqlite3
from datetime import datetime, date, timedelta
import re
import secrets
import hashlib
import hmac
import pandas as pd
import requests
import os

DB_PATH = "drive_profit.db"

# =========================
# Mobile-first CSS
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        .block-container {max-width: 980px; padding-top: 1rem; padding-bottom: 2rem;}
        h1,h2,h3 {letter-spacing:-0.2px;}
        .card{
          background:#fff;border:1px solid rgba(0,0,0,.06);
          border-radius:16px;padding:14px 16px;margin-bottom:12px;
          box-shadow:0 1px 10px rgba(0,0,0,.04);
        }
        .muted{color:rgba(0,0,0,.55);}
        .stButton>button{border-radius:12px;padding:.65rem .95rem;font-weight:800;}
        .stTextInput input,.stSelectbox>div>div{border-radius:12px !important;}
        /* 모바일에서 입력/버튼 크게 */
        @media (max-width: 640px){
          .block-container{padding-left: .8rem; padding-right: .8rem;}
          .stButton>button{width:100%;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Formatting / parsing
# =========================
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

def make_unit_formatter(key: str, unit: str, edited_flag: str | None = None):
    def _cb():
        n = parse_int(st.session_state.get(key, ""))
        st.session_state[key] = fmt_unit(n, unit)
        if edited_flag:
            st.session_state[edited_flag] = True
    return _cb

# =========================
# Password hashing
# =========================
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

# =========================
# DB
# =========================
def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    c = conn()
    cur = c.cursor()

    cur.execute("""
      CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        pw_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TEXT NOT NULL
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

    c.commit()

    # 기본 admin 생성
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO users(username,pw_hash,role,created_at) VALUES(?,?,?,?)",
            ("admin", _pbkdf2_hash("admin1234"), "admin", datetime.now().isoformat(timespec="seconds"))
        )
        c.commit()

    c.close()

def get_user(username: str):
    c = conn()
    cur = c.cursor()
    cur.execute("SELECT id, username, pw_hash, role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    c.close()
    return row

def list_vehicles(user_id: int) -> pd.DataFrame:
    c = conn()
    df = pd.read_sql_query(
        "SELECT id,name,fuel_type,fuel_eff_km_per_l,created_at FROM vehicles WHERE user_id=? ORDER BY id DESC",
        c,
        params=(user_id,)
    )
    c.close()
    return df

def add_vehicle(user_id: int, name: str, fuel_type: str, eff: int):
    c = conn()
    cur = c.cursor()
    cur.execute(
        "INSERT INTO vehicles(user_id,name,fuel_type,fuel_eff_km_per_l,created_at) VALUES(?,?,?,?,?)",
        (user_id, name.strip(), fuel_type, float(eff), datetime.now().isoformat(timespec="seconds"))
    )
    c.commit()
    c.close()

def update_vehicle(user_id: int, vid: int, name: str, fuel_type: str, eff: int):
    c = conn()
    cur = c.cursor()
    cur.execute(
        "UPDATE vehicles SET name=?, fuel_type=?, fuel_eff_km_per_l=? WHERE user_id=? AND id=?",
        (name.strip(), fuel_type, float(eff), user_id, vid)
    )
    c.commit()
    c.close()

def delete_vehicle_cascade(user_id: int, vid: int):
    c = conn()
    cur = c.cursor()
    cur.execute("DELETE FROM trips WHERE user_id=? AND vehicle_id=?", (user_id, vid))
    cur.execute("DELETE FROM vehicles WHERE user_id=? AND id=?", (user_id, vid))
    c.commit()
    c.close()

def save_trip(user_id: int, vehicle_row, trip_date: date, trip_type: str,
             paid_oneway: int, empty_oneway: int, fare: int, fuel_price: int,
             toll: int, parking: int, other: int):
    mult = 2 if trip_type == "왕복" else 1
    total_km = (paid_oneway + empty_oneway) * mult
    eff = float(vehicle_row["fuel_eff_km_per_l"])
    fuel_used = (total_km / eff) if eff > 0 else 0
    fuel_cost = fuel_used * fuel_price
    total_cost = fuel_cost + toll + parking + other
    profit = fare - total_cost
    pct = (profit / fare * 100) if fare > 0 else 0

    c = conn()
    cur = c.cursor()
    cur.execute("""
      INSERT INTO trips(
        user_id,trip_date,vehicle_id,trip_type,
        paid_oneway_km,empty_oneway_km,total_km,
        fare_krw,fuel_price_krw_per_l,toll_krw,parking_krw,other_krw,
        fuel_used_l,fuel_cost_krw,total_cost_krw,profit_krw,profit_pct,
        created_at
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id, trip_date.isoformat(), int(vehicle_row["id"]), trip_type,
        float(paid_oneway), float(empty_oneway), float(total_km),
        float(fare), float(fuel_price), float(toll), float(parking), float(other),
        float(fuel_used), float(fuel_cost), float(total_cost), float(profit), float(pct),
        datetime.now().isoformat(timespec="seconds")
    ))
    c.commit()
    c.close()
    return total_km, fuel_used, fuel_cost, total_cost, profit, pct

def trips_report(user_id: int, start: date, end: date, vehicle_id: int | None):
    c = conn()
    params = {"uid": user_id, "s": start.isoformat(), "e": end.isoformat()}
    where = "t.user_id=:uid AND t.trip_date>=:s AND t.trip_date<=:e"
    if vehicle_id:
        where += " AND t.vehicle_id=:vid"
        params["vid"] = vehicle_id
    df = pd.read_sql_query(f"""
      SELECT
        t.id, t.trip_date, v.name AS vehicle_name, t.trip_type,
        t.total_km, t.fare_krw, t.total_cost_krw, t.profit_krw, t.profit_pct,
        t.fuel_price_krw_per_l, t.fuel_cost_krw, t.toll_krw, t.parking_krw, t.other_krw,
        t.created_at
      FROM trips t
      JOIN vehicles v ON v.id=t.vehicle_id
      WHERE {where}
      ORDER BY t.trip_date DESC, t.id DESC
    """, c, params=params)
    c.close()
    return df

# =========================
# Fuel daily (simple)
# =========================
def refresh_fuel_daily():
    # 최소 동작: 이미 오늘 값 있으면 패스
    today = date.today().isoformat()
    c = conn()
    cur = c.cursor()
    cur.execute("SELECT COUNT(*) FROM fuel_prices_daily WHERE price_date=?", (today,))
    cnt = cur.fetchone()[0]
    c.close()
    if cnt >= 2:
        return

    # (간단) OPINET 페이지 파싱(heuristic)
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

    c = conn()
    cur = c.cursor()
    for ft, p in prices.items():
        cur.execute("""
          INSERT INTO fuel_prices_daily(price_date,fuel_type,price_krw_per_l,source,fetched_at)
          VALUES(?,?,?,?,?)
          ON CONFLICT(price_date,fuel_type) DO UPDATE SET
            price_krw_per_l=excluded.price_krw_per_l,
            source=excluded.source,
            fetched_at=excluded.fetched_at
        """, (today, ft, float(p), "OPINET", datetime.now().isoformat(timespec="seconds")))
    c.commit()
    c.close()

def latest_fuel_price(fuel_type: str):
    c = conn()
    cur = c.cursor()
    cur.execute("""
      SELECT price_krw_per_l, price_date, source
      FROM fuel_prices_daily
      WHERE fuel_type=?
      ORDER BY price_date DESC
      LIMIT 1
    """, (fuel_type,))
    row = cur.fetchone()
    c.close()
    if not row:
        return None, None, None
    return float(row[0]), row[1], row[2]

# =========================
# App start
# =========================
st.set_page_config(page_title="운행손익", page_icon="🚗", layout="centered")
inject_css()
init_db()
refresh_fuel_daily()

# ---------- login ----------
if "uid" not in st.session_state:
    st.session_state.uid = None
if "role" not in st.session_state:
    st.session_state.role = "user"
if "uname" not in st.session_state:
    st.session_state.uname = ""

def logout():
    st.session_state.uid = None
    st.session_state.role = "user"
    st.session_state.uname = ""
    st.rerun()

def login_view():
    st.markdown('<div class="card"><h2>🔐 로그인</h2><p class="muted">모바일에서도 앱처럼 사용 가능</p></div>', unsafe_allow_html=True)
    u = st.text_input("아이디")
    p = st.text_input("비밀번호", type="password")
    if st.button("로그인"):
        row = get_user(u.strip())
        if not row:
            st.error("아이디가 없어요.")
            return
        uid, uname, pw_hash, role = row
        if _verify_pbkdf2(p, pw_hash):
            st.session_state.uid = int(uid)
            st.session_state.uname = uname
            st.session_state.role = role
            st.rerun()
        else:
            st.error("비밀번호가 틀렸어요.")

if not st.session_state.uid:
    login_view()
    st.stop()

UID = int(st.session_state.uid)
ROLE = st.session_state.role
UNAME = st.session_state.uname

# ---------- top nav (mobile friendly) ----------
menu_items = ["차량 등록", "운행 입력", "내역/리포트", "개인정보변경"]
if ROLE == "admin":
    menu_items.append("관리자")

st.markdown(
    f"<div class='card'><b>{UNAME}</b><span class='pill'>{'관리자' if ROLE=='admin' else '사용자'}</span>"
    f"<div class='muted' style='margin-top:6px;'>모바일에서는 상단 메뉴로 이동</div></div>",
    unsafe_allow_html=True
)
if st.button("로그아웃"):
    logout()

page = st.radio("메뉴", menu_items, horizontal=True)

# ---------- common: vehicle select ----------
vdf = list_vehicles(UID)
selected_vehicle = None
if not vdf.empty:
    labels = [f"[{int(r.id)}] {r.name} ({r.fuel_type}, {iround(r.fuel_eff_km_per_l)}KM/L)" for r in vdf.itertuples(index=False)]
    chosen = st.selectbox("기본 차량", labels, index=0)
    chosen_id = int(re.search(r"\[(\d+)\]", chosen).group(1))
    selected_vehicle = vdf[vdf["id"] == chosen_id].iloc[0].to_dict()

# =========================
# Page: 차량 등록
# =========================
if page == "차량 등록":
    st.markdown('<div class="card"><h3>🚗 차량 등록</h3><p class="muted">차량 종류 / 유종 / 연비</p></div>', unsafe_allow_html=True)
    name = st.text_input("차량 종류")
    fuel = st.selectbox("유종", ["휘발유", "경유", "LPG"])
    eff = st.number_input("연비(KM/L)", min_value=1, max_value=100, value=12, step=1, format="%d")
    if st.button("등록"):
        if not name.strip():
            st.error("차량 종류를 입력해줘.")
        else:
            add_vehicle(UID, name, fuel, int(eff))
            st.success("등록 완료!")
            st.rerun()

# =========================
# Page: 운행 입력
# =========================
elif page == "운행 입력":
    if selected_vehicle is None:
        st.warning("차량을 먼저 등록해줘.")
        st.stop()

    # auto fuel default (editable)
    auto_p, auto_d, auto_s = latest_fuel_price(selected_vehicle["fuel_type"])
    auto_int = iround(auto_p) if auto_p is not None else 1700

    # initialize formatted defaults if empty
    if "paid_oneway_km_txt" not in st.session_state:
        st.session_state.paid_oneway_km_txt = "0KM"
    if "empty_oneway_km_txt" not in st.session_state:
        st.session_state.empty_oneway_km_txt = "0KM"
    if "fare_krw_txt" not in st.session_state:
        st.session_state.fare_krw_txt = "30,000원"
    if "fuel_price_txt" not in st.session_state:
        st.session_state.fuel_price_txt = f"{auto_int:,}원/L"
    if "toll_krw_txt" not in st.session_state:
        st.session_state.toll_krw_txt = "0원"
    if "parking_krw_txt" not in st.session_state:
        st.session_state.parking_krw_txt = "0원"
    if "other_cost_krw_txt" not in st.session_state:
        st.session_state.other_cost_krw_txt = "0원"
    if "fuel_user_edited" not in st.session_state:
        st.session_state.fuel_user_edited = False
    if "toll_user_edited" not in st.session_state:
        st.session_state.toll_user_edited = False

    if not st.session_state.fuel_user_edited:
        st.session_state.fuel_price_txt = f"{auto_int:,}원/L"

    st.markdown(
        f"<div class='card'><h3>💰 운행 입력</h3>"
        f"<div class='muted'>차량: <b>{selected_vehicle['name']}</b> ({selected_vehicle['fuel_type']}, 연비 {iround(selected_vehicle['fuel_eff_km_per_l'])}KM/L)"
        f"<br/>오늘 전국 평균 유가(자동 기본): <b>{auto_int:,}원/L</b> ({auto_d or '-'} / {auto_s or 'OPINET'})</div></div>",
        unsafe_allow_html=True
    )

    trip_type = st.selectbox("운행 형태", ["편도", "왕복"])
    trip_date = st.date_input("운행 날짜", value=date.today())

    # formatted inputs with units INSIDE
    st.text_input("유상거리(편도)", key="paid_oneway_km_txt", on_change=make_unit_formatter("paid_oneway_km_txt", "KM"))
    st.text_input("공차거리(편도)", key="empty_oneway_km_txt", on_change=make_unit_formatter("empty_oneway_km_txt", "KM"))
    st.text_input("운임료", key="fare_krw_txt", on_change=make_unit_formatter("fare_krw_txt", "원"))

    st.text_input("유가(수정 가능)", key="fuel_price_txt", on_change=make_unit_formatter("fuel_price_txt", "원/L", "fuel_user_edited"))
    st.text_input("톨비(수정 가능)", key="toll_krw_txt", on_change=make_unit_formatter("toll_krw_txt", "원", "toll_user_edited"))
    st.text_input("주차비", key="parking_krw_txt", on_change=make_unit_formatter("parking_krw_txt", "원"))
    st.text_input("기타비용", key="other_cost_krw_txt", on_change=make_unit_formatter("other_cost_krw_txt", "원"))

    # KPI preview
    paid = parse_int(st.session_state.paid_oneway_km_txt)
    empty = parse_int(st.session_state.empty_oneway_km_txt)
    fare = parse_int(st.session_state.fare_krw_txt)
    fuel_price = parse_int(st.session_state.fuel_price_txt)
    toll = parse_int(st.session_state.toll_krw_txt)
    parking = parse_int(st.session_state.parking_krw_txt)
    other = parse_int(st.session_state.other_cost_krw_txt)
    mult = 2 if trip_type == "왕복" else 1
    total_km = (paid + empty) * mult
    eff = float(selected_vehicle["fuel_eff_km_per_l"])
    fuel_used = (total_km / eff) if eff > 0 else 0
    fuel_cost = fuel_used * fuel_price
    total_cost = fuel_cost + toll + parking + other
    profit = fare - total_cost

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("예상 총거리", fmt_km(total_km))
    c2.metric("예상 기름값", fmt_won(fuel_cost))
    c3.metric("예상 총비용", fmt_won(total_cost))
    c4.metric("예상 순이익", fmt_won(profit))

    if st.button("저장"):
        if (paid <= 0 and empty <= 0) or fare <= 0 or fuel_price <= 0:
            st.error("거리/운임료/유가를 확인해줘.")
        else:
            r = save_trip(
                UID,
                selected_vehicle,
                trip_date,
                trip_type,
                paid, empty, fare, fuel_price,
                toll, parking, other
            )
            st.success("저장 완료!")
            st.rerun()

# =========================
# Page: 내역/리포트
# =========================
elif page == "내역/리포트":
    st.markdown('<div class="card"><h3>📊 내역/리포트</h3><p class="muted">일자별 차트 + 표</p></div>', unsafe_allow_html=True)
    today = date.today()
    start = st.date_input("시작일", value=today - timedelta(days=30))
    end = st.date_input("종료일", value=today)

    vid = int(selected_vehicle["id"]) if selected_vehicle is not None else None
    scope = st.selectbox("차량 범위", ["전체", "선택 차량만"])
    vehicle_id = None if scope == "전체" else vid

    df = trips_report(UID, start, end, vehicle_id)
    if df.empty:
        st.info("데이터가 없어.")
        st.stop()

    # chart
    df2 = df.copy()
    df2["d"] = pd.to_datetime(df2["trip_date"]).dt.date
    chart = df2.groupby("d", as_index=False).agg(fare=("fare_krw","sum"), cost=("total_cost_krw","sum"), profit=("profit_krw","sum")).set_index("d")
    st.line_chart(chart[["fare","cost","profit"]])

    # view table (formatted)
    view = df.copy()
    view.rename(columns={
        "id":"번호","trip_date":"운행일자","vehicle_name":"차량","trip_type":"형태",
        "total_km":"총거리","fare_krw":"운임료","total_cost_krw":"총비용","profit_krw":"순이익","profit_pct":"수익률",
        "fuel_price_krw_per_l":"유가","fuel_cost_krw":"기름값","toll_krw":"톨비","parking_krw":"주차비","other_krw":"기타비용",
        "created_at":"등록시각"
    }, inplace=True)

    view["총거리"] = view["총거리"].apply(fmt_km)
    view["운임료"] = view["운임료"].apply(fmt_won)
    view["총비용"] = view["총비용"].apply(fmt_won)
    view["순이익"] = view["순이익"].apply(fmt_won)
    view["수익률"] = view["수익률"].apply(fmt_pct)
    view["유가"] = view["유가"].apply(fmt_won_per_l)
    view["기름값"] = view["기름값"].apply(fmt_won)
    view["톨비"] = view["톨비"].apply(fmt_won)
    view["주차비"] = view["주차비"].apply(fmt_won)
    view["기타비용"] = view["기타비용"].apply(fmt_won)

    st.dataframe(view, width="stretch", hide_index=True)

# =========================
# Page: 개인정보변경
# =========================
elif page == "개인정보변경":
    st.markdown('<div class="card"><h3>👤 개인정보변경</h3><p class="muted">차량 수정/삭제</p></div>', unsafe_allow_html=True)

    if vdf.empty:
        st.info("차량이 없어.")
    else:
        labels = [f"[{int(r.id)}] {r.name} ({r.fuel_type}, {iround(r.fuel_eff_km_per_l)}KM/L)" for r in vdf.itertuples(index=False)]
        sel = st.selectbox("차량 선택", labels, index=0)
        vid = int(re.search(r"\[(\d+)\]", sel).group(1))
        row = vdf[vdf["id"] == vid].iloc[0]

        name = st.text_input("차량 종류", value=row["name"])
        fuel = st.selectbox("유종", ["휘발유","경유","LPG"], index=["휘발유","경유","LPG"].index(row["fuel_type"]))
        eff = st.number_input("연비(KM/L)", min_value=1, max_value=100, value=iround(row["fuel_eff_km_per_l"]), step=1, format="%d")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("차량 수정 저장"):
                update_vehicle(UID, vid, name, fuel, int(eff))
                st.success("수정 완료")
                st.rerun()
        with c2:
            if st.button("차량 삭제(운행 포함)"):
                delete_vehicle_cascade(UID, vid)
                st.success("삭제 완료")
                st.rerun()

# =========================
# Page: 관리자
# =========================
else:
    if ROLE != "admin":
        st.error("권한 없음")
        st.stop()
    st.markdown('<div class="card"><h3>🛠 관리자</h3><p class="muted">관리자 전용</p></div>', unsafe_allow_html=True)
    c = conn()
    df = pd.read_sql_query("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC", c)
    c.close()
    st.dataframe(df, width="stretch", hide_index=True)
