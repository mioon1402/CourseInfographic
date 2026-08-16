"""올인원 데이터 분석 엔진 (dap_engine)

GitHub Pages 포털에서 고른 분석을 Google Colab 에서 실행하는 실제 분석 코드입니다.
노트북은 이 파일을 내려받아 아래 3개만 호출합니다.

    import dap_engine as dap
    dap.bootstrap("spc_control")   # 준비 (필요한 라이브러리만 설치)
    dap.load_data()                # CSV 업로드 (또는 예제 데이터)
    dap.run_and_report()           # 분석 실행 + 리포트 저장

분석 로직을 고치고 싶으면 이 파일만 수정하시면 됩니다.
새 분석 추가 방법은 README_PORTAL.md 5번 항목을 참고하세요.
"""
import base64
import difflib
import html as _html
import importlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import warnings
import zipfile
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from google.colab import output as _colab_output   # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

__version__ = "2.0"

# ── 분석 목록 ────────────────────────────────────────────────────────────
MODULE_CATALOG = [
    ("eda_summary",     "데이터 자동 요약 리포트",           "데이터가 어떻게 생겼는지 전체를 훑어봅니다"),
    ("data_quality",    "데이터 품질 진단 및 자동 정제",     "빠진 값·중복·오타를 찾아 고쳐 줍니다"),
    ("eda_corr",        "다중 변수 상관관계 분석",           "어떤 항목끼리 함께 움직이는지 봅니다"),
    ("compare_groups",  "그룹 간 차이 비교 (통계 검정)",     "A와 B의 차이가 진짜인지 확인합니다"),
    ("ts_forecast",     "미래 추세 및 계절성 예측 (Prophet)", "앞으로의 값을 예측합니다"),
    ("ts_smooth",       "데이터 평활화 및 트렌드 추출",      "잡음을 걷어내고 큰 흐름을 봅니다"),
    ("anomaly_iforest", "이상치 감지 (Isolation Forest)",    "평소와 다른 데이터를 찾아냅니다"),
    ("spc_control",     "공정 관리도 (SPC)",                 "공정이 안정적인지 판정합니다"),
    ("pareto",          "파레토 분석 (80/20)",               "어디부터 손대야 효과가 큰지 찾습니다"),
    ("regress_rf",      "결과 수치 예측 (Random Forest)",    "여러 조건으로 숫자를 예측합니다"),
    ("classify_xgb",    "정상/불량 상태 분류 (XGBoost)",     "여러 조건으로 상태를 판정합니다"),
    ("cluster_kmeans",  "유사 데이터 자동 그룹핑 (K-Means)", "비슷한 것끼리 묶습니다"),
]
MODULE_NAMES = {m[0]: m[1] for m in MODULE_CATALOG}

# 분석별로 꼭 필요한 컬럼 (드롭다운을 무엇으로 보여줄지 결정합니다)
MODULE_NEEDS = {
    "eda_summary":     {"date": "optional", "target": "optional", "group": "none",     "features": "optional"},
    "data_quality":    {"date": "none",     "target": "none",     "group": "none",     "features": "none"},
    "eda_corr":        {"date": "none",     "target": "optional", "group": "none",     "features": "optional"},
    "compare_groups":  {"date": "none",     "target": "required", "group": "required", "features": "none"},
    "ts_forecast":     {"date": "required", "target": "required", "group": "none",     "features": "none"},
    "ts_smooth":       {"date": "required", "target": "required", "group": "none",     "features": "none"},
    "anomaly_iforest": {"date": "optional", "target": "optional", "group": "none",     "features": "optional"},
    "spc_control":     {"date": "optional", "target": "required", "group": "none",     "features": "none"},
    "pareto":          {"date": "none",     "target": "optional", "group": "required", "features": "none"},
    "regress_rf":      {"date": "none",     "target": "required", "group": "none",     "features": "optional"},
    "classify_xgb":    {"date": "none",     "target": "required", "group": "none",     "features": "optional"},
    "cluster_kmeans":  {"date": "none",     "target": "optional", "group": "none",     "features": "optional"},
}

# 분석별 추가 설치 패키지 — 고른 분석에 필요한 것만 설치해 시작 시간을 줄입니다.
MODULE_DEPS = {
    "ts_forecast":  [("prophet", "prophet", "Prophet(시계열 예측)")],
    "ts_smooth":    [("statsmodels", "statsmodels", "statsmodels")],
    "eda_summary":  [("statsmodels", "statsmodels", "statsmodels")],
    "classify_xgb": [("xgboost", "xgboost", "XGBoost")],
}

# ── 실행 중 상태 ─────────────────────────────────────────────────────────
CONFIG = {}
MODULE_ID = "eda_summary"
DATE_COL = TARGET_COL = GROUP_COL = ""
FEATURE_COLS = []
df = None
DATA_SOURCE_NAME = "예제 데이터"
RAW_ROWS = RAW_COLS = 0
HAS_PROPHET = HAS_XGB = False
RESULT = None
FAILED = False


# ── 설정 읽기 ────────────────────────────────────────────────────────────
def _decode_token(token):
    """포털이 만든 'DAP1.xxxx' 설정 코드를 딕셔너리로 되돌립니다."""
    if not token:
        return None
    token = str(token).strip().strip('"').strip("'")
    if token.startswith("DAP1."):
        token = token[5:]
    token = token.replace("-", "+").replace("_", "/")
    token += "=" * (-len(token) % 4)
    try:
        cfg = json.loads(base64.b64decode(token).decode("utf-8"))
        return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None


def _config_from_url():
    """브라우저 주소창의 설정을 읽어 봅니다. (Colab 보안 정책상 실패할 수 있음)"""
    if not IN_COLAB:
        return None
    js = """
    (function () {
      var urls = [];
      try { if (window.parent && window.parent.location && window.parent.location.href) urls.push(window.parent.location.href); } catch (e) {}
      try { if (window.top && window.top.location && window.top.location.href) urls.push(window.top.location.href); } catch (e) {}
      try { if (document.referrer) urls.push(document.referrer); } catch (e) {}
      try { urls.push(window.location.href); } catch (e) {}
      for (var i = 0; i < urls.length; i++) {
        if (urls[i] && urls[i].indexOf('module_id=') !== -1) return urls[i];
      }
      for (var j = 0; j < urls.length; j++) {
        if (urls[j] && urls[j].indexOf('cfg=') !== -1) return urls[j];
      }
      return "";
    })()
    """
    try:
        from google.colab import output as colab_output
        url = colab_output.eval_js(js, timeout_sec=6)
    except Exception:
        return None
    if not url:
        return None
    try:
        from urllib.parse import urlparse, parse_qs, unquote
        qs = parse_qs(urlparse(unquote(str(url))).query)
        flat = {k: v[0] for k, v in qs.items() if v}
        if flat.get("module_id"):
            return flat
        if flat.get("cfg"):
            return _decode_token(flat["cfg"])
    except Exception:
        return None
    return None


def configure(cfg=None, module_id=None):
    """설정을 적용합니다. (포털 설정 → 전역 변수)"""
    global CONFIG, MODULE_ID, DATE_COL, TARGET_COL, GROUP_COL, FEATURE_COLS
    if cfg:
        CONFIG = dict(cfg)
    if module_id:
        MODULE_ID = module_id
    elif CONFIG.get("module_id") in MODULE_NAMES:
        MODULE_ID = CONFIG["module_id"]
    CONFIG["module_id"] = MODULE_ID
    DATE_COL = P_str("date_col")
    TARGET_COL = P_str("target_col")
    GROUP_COL = P_str("group_col")
    FEATURE_COLS = P_list("feature_cols")
    return CONFIG


def P(key, default=None):
    val = CONFIG.get("p_" + key, CONFIG.get(key, default))
    return default if val in (None, "") else val


def P_int(key, default=0):
    try:
        return int(float(P(key, default)))
    except Exception:
        return int(default)


def P_float(key, default=0.0):
    try:
        return float(P(key, default))
    except Exception:
        return float(default)


def P_bool(key, default=False):
    val = P(key, None)
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on", "예", "사용함")


def P_str(key, default=""):
    val = P(key, default)
    return "" if val is None else str(val).strip()


def P_list(key, default=None):
    raw = P(key, "")
    if isinstance(raw, (list, tuple)):
        return [str(s).strip() for s in raw if str(s).strip()]
    raw = str(raw or "")
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else list(default or [])


# ── 필요한 라이브러리만 설치 ──────────────────────────────────────────────
def _have(pkg):
    """설치 여부만 확인합니다. (import 하지 않아 훨씬 빠릅니다)"""
    try:
        return importlib.util.find_spec(pkg) is not None
    except Exception:
        return False


def _ensure(pkg_import, pip_name=None, label=None):
    if _have(pkg_import):
        return True
    name = pip_name or pkg_import
    print(f"  · {label or name} 설치 중… (처음 한 번만 걸립니다)")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", name],
                       check=True, capture_output=True, timeout=900)
        importlib.invalidate_caches()
        return _have(pkg_import)
    except Exception as exc:
        print(f"  ⚠️  {label or name} 설치 실패: {exc}")
        return False


def bootstrap(module_id="", cfg_token=""):
    """1단계 — 설정을 확보하고, 고른 분석에 필요한 라이브러리만 설치합니다."""
    global HAS_PROPHET, HAS_XGB
    cfg = _decode_token(cfg_token) or _config_from_url() or {}
    configure(cfg, module_id or None)

    missing = [p for p in ("pandas", "numpy", "plotly") if not _have(p)]
    if missing:
        for p in missing:
            _ensure(p)
    for pkg, pip_name, label in MODULE_DEPS.get(MODULE_ID, []):
        ok = _ensure(pkg, pip_name, label)
        if pkg == "prophet":
            HAS_PROPHET = ok
        elif pkg == "xgboost":
            HAS_XGB = ok
    if MODULE_ID == "classify_xgb" and not HAS_XGB:
        HAS_XGB = _have("xgboost")
    if MODULE_ID == "ts_forecast" and not HAS_PROPHET:
        HAS_PROPHET = _have("prophet")

    src = ("포털에서 받은 설정" if cfg else "기본값 (2단계에서 직접 고르실 수 있습니다)")
    print(f"✅ 준비 완료 — {MODULE_NAMES.get(MODULE_ID, MODULE_ID)}")
    print(f"   설정 출처: {src}")
    return CONFIG



# ── 데이터 로딩 헬퍼 ────────────────────────────────────────────────────

import numpy as np
import pandas as pd

import io
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


def _make_demo_data(n_days=900, seed=42):
    """연습용 예제 데이터: 어느 공장의 일별 설비/생산 기록."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 1000 + 0.9 * t
    weekly = 90 * np.sin(2 * np.pi * t / 7)
    yearly = 220 * np.sin(2 * np.pi * t / 365.25 - 0.8)
    production = trend + weekly + yearly + rng.normal(0, 45, n_days)

    temperature = 62 + 8 * np.sin(2 * np.pi * t / 365.25 - 1.2) + rng.normal(0, 2.2, n_days)
    humidity = 48 + 12 * np.sin(2 * np.pi * t / 365.25 + 0.4) + rng.normal(0, 4.5, n_days)
    vibration = 2.4 + 0.018 * (temperature - 62) ** 2 + rng.normal(0, 0.28, n_days)
    pressure = 101.0 + rng.normal(0, 1.1, n_days) + 0.02 * (t / n_days) * 30
    speed = 120 + rng.normal(0, 6, n_days) + 0.004 * t

    yield_pct = (
        99.0
        - 0.42 * np.abs(temperature - 62)
        - 2.9 * np.maximum(vibration - 2.6, 0)
        - 0.10 * np.abs(pressure - 101.5)
        - 0.012 * np.maximum(speed - 125, 0) ** 1.4
        + rng.normal(0, 0.55, n_days)
    ).clip(80, 100)

    # 이상 상황(설비 스파이크) 약 2% 주입
    n_anom = max(6, int(n_days * 0.02))
    idx = rng.choice(n_days, size=n_anom, replace=False)
    vibration[idx] += rng.uniform(2.0, 4.5, n_anom)
    temperature[idx] += rng.uniform(6, 14, n_anom)
    yield_pct[idx] -= rng.uniform(5, 14, n_anom)
    production[idx] -= rng.uniform(150, 420, n_anom)
    yield_pct = yield_pct.clip(60, 100)

    df = pd.DataFrame({
        "날짜": dates,
        "생산량": production.round(1),
        "설비온도": temperature.round(2),
        "습도": humidity.round(2),
        "진동값": vibration.round(3),
        "압력": pressure.round(2),
        "라인속도": speed.round(1),
        "수율": yield_pct.round(2),
    })
    df["불량여부"] = np.where(df["수율"] < 93, "불량", "정상")
    df["생산라인"] = rng.choice(["A라인", "B라인", "C라인"], size=n_days, p=[0.45, 0.35, 0.20])
    # 라인마다 실력 차이를 조금 넣어 둡니다 (그룹 비교·파레토 연습용)
    df.loc[df["생산라인"] == "C라인", "수율"] -= 1.4
    df.loc[df["생산라인"] == "B라인", "수율"] -= 0.5
    df["수율"] = df["수율"].round(2)

    # 실제 현장 데이터처럼 '지저분한 부분'을 일부러 넣습니다 (품질 진단 연습용)
    df["설비코드"] = "EQ-100"                                   # 항상 같은 값
    df["비고"] = rng.choice(["", "", "", " 정기점검 ", "교대근무"], size=n_days)  # 앞뒤 공백
    na_idx = rng.choice(n_days, size=int(n_days * 0.03), replace=False)
    df.loc[na_idx, "습도"] = np.nan                              # 결측치
    na_idx2 = rng.choice(n_days, size=int(n_days * 0.015), replace=False)
    df.loc[na_idx2, "압력"] = np.nan
    df = pd.concat([df, df.iloc[rng.choice(n_days, size=6, replace=False)]],
                   ignore_index=True)                            # 중복 행
    return df


def _coerce_numeric(df):
    """'1,234' · '85%' 처럼 문자로 저장된 숫자를 진짜 숫자로 바꿉니다."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        s = df[col].astype(str).str.strip()
        s = s.str.replace(r"[,\s₩$]", "", regex=True).str.replace("%", "", regex=False)
        conv = pd.to_numeric(s, errors="coerce")
        valid = conv.notna().sum()
        total = df[col].notna().sum()
        if total > 0 and valid / total >= 0.95 and valid > 0:
            df[col] = conv
    return df


def _read_any(name, content):
    """확장자·인코딩·구분자를 알아서 판단해 읽습니다."""
    lower = name.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=None,
                             engine="python", skip_blank_lines=True)
            if df.shape[1] == 1:      # 구분자 추론 실패 → 흔한 구분자들로 재시도
                for sep in (",", ";", "\t", "|"):
                    alt = pd.read_csv(io.BytesIO(content), encoding=enc, sep=sep)
                    if alt.shape[1] > df.shape[1]:
                        df = alt
            print(f"  · 인코딩 {enc} 로 읽었습니다.")
            return df
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"파일을 읽지 못했습니다: {last_err}")


# ═══════════════════════════════════════════════════════════════════════
#  2단계 · 데이터 올리기
# ═══════════════════════════════════════════════════════════════════════
def load_data(path=None, use_demo=False):
    """CSV 를 올려 받습니다. 파일이 없으면 연습용 예제 데이터를 만듭니다."""
    global df, DATA_SOURCE_NAME, RAW_ROWS, RAW_COLS
    import pandas as pd

    data = None
    if path:
        data = _read_any(str(path), open(path, "rb").read())
        DATA_SOURCE_NAME = os.path.basename(str(path))
    elif IN_COLAB and not use_demo:
        try:
            from google.colab import files as colab_files
            print("📂 분석할 CSV 파일을 선택하세요.")
            print("   (파일이 없으면 [취소] → 연습용 예제 데이터로 진행합니다)\n")
            uploaded = colab_files.upload()
        except Exception as exc:
            print(f"업로드 창을 열지 못했습니다({exc}). 예제 데이터로 진행합니다.")
            uploaded = {}
        if uploaded:
            fname = list(uploaded.keys())[0]
            print(f"\n📥 '{fname}' 을(를) 읽는 중…")
            data = _read_any(fname, uploaded[fname])
            DATA_SOURCE_NAME = fname

    if data is None:
        print("\n🧪 연습용 예제 데이터(공장 설비 일별 기록)를 생성합니다.")
        data = _make_demo_data()
        DATA_SOURCE_NAME = "예제 데이터 (공장 설비 일별 기록)"

    data.columns = [str(c).strip().replace("\n", " ") for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    data = data.dropna(axis=1, how="all").dropna(axis=0, how="all")
    data = _coerce_numeric(data).reset_index(drop=True)
    df = data
    RAW_ROWS, RAW_COLS = df.shape
    print(f"\n✅ 불러오기 완료 — {RAW_ROWS:,}행 × {RAW_COLS}열")
    print(f"   컬럼: {', '.join(map(str, df.columns[:15]))}"
          + (" …" if RAW_COLS > 15 else ""))
    return df



# ═══════════════════════════════════════════════════════════════════════
#  그래프 테마 · 리포트 도구
# ═══════════════════════════════════════════════════════════════════════

import difflib
import html as _html
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import HTML, display

# ── 색상 팔레트 ──────────────────────────────────────────────────────────
# 색약(색맹) 검증을 통과한 순서입니다. 순서를 바꾸지 마세요.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#f2f7fe", "#cde2fb", "#9ec5f4", "#6da7ec",
            "#3987e5", "#256abf", "#184f95", "#0d366b"]
# 상관계수·편차처럼 '음수 ↔ 0 ↔ 양수' 를 나타낼 때는 반드시 발산형(중앙 회색)
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"],
             [0.75, "#eb6834"], [1.0, "#b3241a"]]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
GOOD, WARN, CRIT = "#0ca30c", "#e08800", "#d03b3b"
# 폰트 이름은 반드시 작은따옴표로 감쌉니다.
# 큰따옴표를 쓰면 style="..." 속성 안에서 문자열이 잘려 디자인이 통째로 깨집니다.
FONT = "system-ui, -apple-system, 'Segoe UI', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif"

pio.templates["dap"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=13, color=INK),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    colorway=PALETTE,
    title=dict(font=dict(size=17, color=INK), x=0.01, xanchor="left", y=0.96),
    margin=dict(l=64, r=28, t=64, b=52),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7",
               tickfont=dict(color=MUTED, size=11.5), title=dict(font=dict(color=INK_2, size=12.5)),
               showline=True, ticks="outside", tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7",
               tickfont=dict(color=MUTED, size=11.5), title=dict(font=dict(color=INK_2, size=12.5)),
               showline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(font=dict(family=FONT, size=12.5), bordercolor=GRID),
    hovermode="closest",
))
pio.templates.default = "dap"
if IN_COLAB:
    try:
        pio.renderers.default = "colab"
    except Exception:
        pass

# ── 리포트 누적 (마지막에 HTML 파일로 저장됩니다) ─────────────────────────
REPORT_PARTS = []      # 그래프·표·해석 HTML 조각
RESULT_FILES = []      # 함께 저장할 결과 데이터 (파일명, 데이터프레임)
REPORT_TOC = []        # 목차 (단계, 제목, 앵커)


def show(fig, height=None, title=None):
    """그래프를 화면에 보여주고 리포트에도 담아 둡니다."""
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=title)
    fig.update_layout(template="dap")
    REPORT_PARTS.append(fig.to_html(full_html=False, include_plotlyjs=False,
                                    config={"displaylogo": False}))
    fig.show(config={"displaylogo": False,
                     "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def _emit(html_str):
    REPORT_PARTS.append(html_str)
    display(HTML(html_str))


def _anchor(text):
    a = f"sec{len(REPORT_TOC)+1}"
    return a


def h_title(text, sub=""):
    a = _anchor(text)
    REPORT_TOC.append((1, text, a))
    _emit(f"""
    <div id="{a}" style="font-family:{FONT};margin:30px 0 6px;padding:0 0 10px;border-bottom:2px solid {GRID}">
      <div style="font-size:21px;font-weight:700;color:{INK}">{_html.escape(text)}</div>
      {f'<div style="font-size:13.5px;color:{INK_2};margin-top:5px">{sub}</div>' if sub else ''}
    </div>""")


def h_sub(text):
    a = _anchor(text)
    REPORT_TOC.append((2, text, a))
    _emit(f"""<div id="{a}" style="font-family:{FONT};font-size:16px;font-weight:700;color:{INK};
              margin:24px 0 8px">{_html.escape(text)}</div>""")


def h_note(text, kind="info"):
    color = {"info": "#2a78d6", "good": GOOD, "warn": WARN, "crit": CRIT}[kind]
    icon = {"info": "💡", "good": "✅", "warn": "⚠️", "crit": "🚨"}[kind]
    _emit(f"""
    <div style="font-family:{FONT};display:flex;gap:10px;align-items:flex-start;
                background:{color}14;border:1px solid {color}55;border-left:4px solid {color};
                border-radius:9px;padding:11px 14px;margin:10px 0;font-size:13.5px;
                line-height:1.65;color:{INK}">
      <span style="flex:none">{icon}</span><div>{text}</div></div>""")


def h_metrics(items):
    """items = [(제목, 값, 부가설명), ...] — 핵심 숫자 타일"""
    cards = "".join(f"""
      <div style="flex:1 1 150px;background:{SURFACE};border:1px solid rgba(11,11,11,.12);
                  border-radius:12px;padding:13px 15px">
        <div style="font-size:11.5px;color:{MUTED};font-weight:600;letter-spacing:.02em">{_html.escape(str(k))}</div>
        <div style="font-size:23px;font-weight:700;color:{INK};margin:3px 0 1px;line-height:1.2">{v}</div>
        <div style="font-size:11.5px;color:{INK_2}">{s}</div>
      </div>""" for k, v, s in items)
    _emit(f'<div style="font-family:{FONT};display:flex;flex-wrap:wrap;gap:10px;margin:14px 0">{cards}</div>')


def h_insight(bullets, title="이 결과를 이렇게 읽으세요"):
    """전문 용어 없이 결론만 정리해 주는 해석 상자."""
    lis = "".join(f'<li style="margin:6px 0">{b}</li>' for b in bullets if b)
    _emit(f"""
    <div style="font-family:{FONT};background:#f4f8fd;border:1px solid #cde2fb;border-radius:12px;
                padding:15px 18px;margin:16px 0">
      <div style="font-weight:700;font-size:14.5px;color:#184f95;margin-bottom:6px">🧭 {_html.escape(title)}</div>
      <ul style="margin:0;padding-left:20px;font-size:13.5px;line-height:1.75;color:{INK}">{lis}</ul>
    </div>""")


def h_table(frame, caption="", max_rows=25, highlight=None):
    """표를 보기 좋게 출력합니다. highlight = 강조할 컬럼명"""
    view = frame.head(max_rows).copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:,.4g}")
    head = "".join(
        f'<th style="text-align:left;padding:8px 11px;font-size:12px;color:{INK_2};'
        f'border-bottom:1.5px solid {GRID};white-space:nowrap">{_html.escape(str(c))}</th>'
        for c in view.columns)
    rows = ""
    for _, r in view.iterrows():
        tds = "".join(
            f'<td style="padding:7px 11px;font-size:12.5px;color:{INK};'
            f'border-bottom:1px solid #f0efec;white-space:nowrap;'
            f'{"font-weight:700" if highlight and c == highlight else ""}">'
            f'{_html.escape(str(r[c]))}</td>' for c in view.columns)
        rows += f"<tr>{tds}</tr>"
    more = (f'<div style="font-size:12px;color:{MUTED};padding:7px 11px">'
            f'… 전체 {len(frame):,}행 중 {len(view):,}행만 표시</div>') if len(frame) > len(view) else ""
    _emit(f"""
    <div style="font-family:{FONT};margin:12px 0">
      {f'<div style="font-size:13px;font-weight:650;color:{INK};margin-bottom:6px">{_html.escape(caption)}</div>' if caption else ''}
      <div style="overflow-x:auto;border:1px solid rgba(11,11,11,.12);border-radius:10px;background:{SURFACE}">
        <table style="border-collapse:collapse;width:100%"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
        {more}
      </div></div>""")


# ── 컬럼 확인 도우미 ─────────────────────────────────────────────────────
def resolve_col(name, kind=None, required=False, purpose=""):
    """사용자가 적은 컬럼명을 실제 데이터의 컬럼과 맞춰 줍니다.

    정확히 일치 → 대소문자/공백 무시 → 비슷한 이름 추천 → 자동 선택 순서로 찾습니다.
    """
    cols = list(df.columns)
    if name:
        if name in cols:
            return name
        norm = {str(c).strip().lower().replace(" ", ""): c for c in cols}
        key = str(name).strip().lower().replace(" ", "")
        if key in norm:
            print(f"  ℹ️  '{name}' → '{norm[key]}' 로 인식했습니다.")
            return norm[key]
        near = difflib.get_close_matches(str(name), [str(c) for c in cols], n=1, cutoff=0.6)
        if near:
            print(f"  ⚠️  '{name}' 컬럼이 없어 가장 비슷한 '{near[0]}' 을(를) 사용합니다.")
            return near[0]
        print(f"  ⚠️  '{name}' 컬럼을 찾지 못했습니다.")

    auto = None
    if kind == "date":
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                auto = c
                break
        if auto is None:
            for c in cols:
                if any(k in str(c).lower() for k in ("date", "time", "날짜", "일자", "시각", "일시", "월", "년")):
                    if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.7:
                        auto = c
                        break
        if auto is None:
            for c in cols:
                if df[c].dtype == object or pd.api.types.is_numeric_dtype(df[c]):
                    try:
                        if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.9:
                            auto = c
                            break
                    except Exception:
                        pass
    elif kind == "number":
        nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
        auto = nums[0] if nums else None
    elif kind == "category":
        cands = [c for c in cols if 2 <= df[c].nunique() <= 20]
        auto = cands[0] if cands else None

    if auto is not None:
        print(f"  ✅ {purpose or '컬럼'}을(를) '{auto}' 로 자동 선택했습니다.")
        return auto
    if required:
        raise ValueError(
            f"{purpose or '필요한 컬럼'}을 찾을 수 없습니다. "
            f"포털에서 컬럼 이름을 다시 지정해 주세요.\n사용 가능한 컬럼: {', '.join(map(str, cols))}")
    return None


def numeric_columns(exclude=(), min_unique=2):
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) >= min_unique:
            out.append(c)
    return out


def pick_features(exclude=(), max_n=40):
    """분석에 쓸 컬럼 목록을 정합니다 (사용자 지정 우선, 없으면 자동)."""
    if FEATURE_COLS:
        chosen, missing = [], []
        for f in FEATURE_COLS:
            r = resolve_col(f)
            if r is not None and r not in exclude and r not in chosen:
                chosen.append(r)
            elif r is None:
                missing.append(f)
        if missing:
            print(f"  ⚠️  다음 컬럼은 찾지 못해 제외합니다: {', '.join(missing)}")
        if chosen:
            return chosen
        print("  ⚠️  지정한 컬럼을 하나도 찾지 못해 자동 선택으로 전환합니다.")
    auto = numeric_columns(exclude=exclude)
    return auto[:max_n]


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def normalize_freq(freq):
    """pandas 버전에 따라 달라진 주기 표기를 안전한 값으로 통일합니다."""
    return {"H": "h", "T": "min", "M": "ME", "Y": "YE"}.get(str(freq), str(freq))


FREQ_LABEL = {"D": "일", "W": "주", "MS": "개월", "h": "시간", "H": "시간",
              "15min": "×15분", "QS": "분기", "ME": "개월"}



# ═══════════════════════════════════════════════════════════════════════
#  분석 모듈
# ═══════════════════════════════════════════════════════════════════════

def _iqr_outliers(s):
    s = s.dropna()
    if len(s) < 8:
        return 0, None, None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, None, None
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), lo, hi


def run_eda_summary():
    """[탐색적 분석] 데이터 자동 요약 리포트."""
    h_title("📋 데이터 자동 요약 리포트",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    max_cols = P_int("max_plot_cols", 12)
    strategy = P_str("missing_strategy", "report")

    work = df.copy()
    n_dup = int(work.duplicated().sum())
    miss_total = work.isna().sum().sum()
    miss_pct = miss_total / max(work.size, 1) * 100
    nums = numeric_columns()
    cats = [c for c in work.columns if c not in nums and not pd.api.types.is_datetime64_any_dtype(work[c])]

    h_metrics([
        ("전체 행 수", f"{len(work):,}", "데이터 건수"),
        ("전체 열 수", f"{work.shape[1]:,}", f"숫자 {len(nums)} · 그 외 {work.shape[1]-len(nums)}"),
        ("결측치", f"{miss_pct:.1f}%", f"{miss_total:,}칸이 비어 있음"),
        ("중복 행", f"{n_dup:,}", "완전히 같은 행" if n_dup else "중복 없음 👍"),
    ])

    # ── 컬럼별 개요 ──────────────────────────────────────────────────────
    h_sub("① 컬럼별 개요")
    rows = []
    for c in work.columns:
        s = work[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "날짜"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "숫자"
        elif s.nunique(dropna=True) <= max(20, len(s) * 0.05):
            kind = "범주(카테고리)"
        else:
            kind = "텍스트"
        n_out, _, _ = _iqr_outliers(s) if kind == "숫자" else (0, None, None)
        sample = s.dropna()
        rows.append({
            "컬럼명": c, "성격": kind,
            "결측": f"{s.isna().sum():,} ({s.isna().mean()*100:.1f}%)",
            "고유값": f"{s.nunique(dropna=True):,}",
            "대표값": ("" if sample.empty else
                     (f"{sample.mean():,.3g} (평균)" if kind == "숫자"
                      else str(sample.mode().iloc[0])[:24] if not sample.mode().empty else "")),
            "이상값 후보": f"{n_out:,}" if kind == "숫자" else "—",
        })
    overview = pd.DataFrame(rows)
    h_table(overview, "각 컬럼이 어떤 성격인지, 빠진 값은 얼마나 되는지 정리했습니다.",
            max_rows=60, highlight="컬럼명")

    # ── 결측치 ───────────────────────────────────────────────────────────
    miss = work.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        h_sub("② 비어 있는 값(결측치)")
        m = miss.head(25).sort_values()
        fig = go.Figure(go.Bar(
            x=m.values, y=[str(i) for i in m.index], orientation="h",
            marker=dict(color=PALETTE[0], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in m.values], textposition="outside",
            cliponaxis=False, hovertemplate="%{y}<br>결측 %{x:.2f}%<extra></extra>"))
        fig.update_layout(title="컬럼별 결측치 비율", xaxis_title="결측 비율 (%)",
                          yaxis_title=None, height=max(260, 30 * len(m) + 130),
                          xaxis=dict(range=[0, min(100, m.max() * 1.25 + 5)]))
        show(fig)
        worst = miss.index[0]
        h_note(f"결측이 가장 많은 컬럼은 <b>{_html.escape(str(worst))}</b> ({miss.iloc[0]:.1f}%)입니다. "
               f"50%를 넘는 컬럼은 분석에서 제외하는 편이 안전합니다.",
               "warn" if miss.iloc[0] > 30 else "info")
    else:
        h_note("빠진 값이 하나도 없습니다. 아주 깨끗한 데이터입니다.", "good")

    # ── 숫자 컬럼 분포 ───────────────────────────────────────────────────
    if nums:
        h_sub("③ 숫자 컬럼의 분포")
        sel = nums[:max_cols]
        ncol = min(3, len(sel))
        nrow = int(np.ceil(len(sel) / ncol))
        fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[str(c) for c in sel],
                            vertical_spacing=max(0.06, 0.32 / nrow), horizontal_spacing=0.08)
        for i, c in enumerate(sel):
            r, cc = i // ncol + 1, i % ncol + 1
            fig.add_trace(go.Histogram(x=work[c].dropna(), nbinsx=32, name=str(c),
                                       marker=dict(color=PALETTE[0], line=dict(width=0)),
                                       hovertemplate="구간 %{x}<br>%{y}건<extra></extra>"), row=r, col=cc)
        fig.update_layout(showlegend=False, height=210 * nrow + 90,
                          title=f"숫자 컬럼 분포 ({len(sel)}개)", bargap=0.04)
        fig.update_annotations(font=dict(size=12, color=INK_2))
        show(fig)

        desc = work[sel].describe().T
        desc["왜도(치우침)"] = work[sel].skew()
        desc = desc.rename(columns={"count": "개수", "mean": "평균", "std": "표준편차",
                                    "min": "최솟값", "25%": "하위25%", "50%": "중앙값",
                                    "75%": "상위25%", "max": "최댓값"})
        h_table(desc.reset_index().rename(columns={"index": "컬럼"}),
                "숫자 컬럼 기초 통계", max_rows=40, highlight="컬럼")

    # ── 범주형 컬럼 ──────────────────────────────────────────────────────
    cat_small = [c for c in cats if 1 < work[c].nunique(dropna=True) <= 15][:6]
    if cat_small:
        h_sub("④ 범주(카테고리) 컬럼 구성비")
        ncol = min(3, len(cat_small))
        nrow = int(np.ceil(len(cat_small) / ncol))
        fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[str(c) for c in cat_small],
                            vertical_spacing=max(0.08, 0.34 / nrow), horizontal_spacing=0.10)
        for i, c in enumerate(cat_small):
            vc = work[c].value_counts().head(10)
            r, cc = i // ncol + 1, i % ncol + 1
            fig.add_trace(go.Bar(x=[str(x) for x in vc.index], y=vc.values,
                                 marker=dict(color=PALETTE[2], line=dict(width=0)),
                                 hovertemplate="%{x}<br>%{y:,}건<extra></extra>"), row=r, col=cc)
        fig.update_layout(showlegend=False, height=230 * nrow + 90, title="범주별 건수")
        fig.update_annotations(font=dict(size=12, color=INK_2))
        show(fig)

    # ── 날짜 컬럼 커버리지 ───────────────────────────────────────────────
    date_col = resolve_col(DATE_COL, kind="date", purpose="날짜 컬럼")
    if date_col:
        h_sub("⑤ 기간 확인")
        ds = pd.to_datetime(work[date_col], errors="coerce").dropna().sort_values()
        if len(ds) > 2:
            gaps = ds.diff().dropna()
            step = gaps.median()
            n_gap = int((gaps > step * 3).sum())
            h_metrics([
                ("시작", f"{ds.iloc[0]:%Y-%m-%d}", "가장 이른 기록"),
                ("종료", f"{ds.iloc[-1]:%Y-%m-%d}", "가장 늦은 기록"),
                ("기록 간격", str(step).replace("0 days ", ""), "중앙값 기준"),
                ("끊긴 구간", f"{n_gap:,}", "평소 간격의 3배 이상" if n_gap else "빠짐 없음 👍"),
            ])
            if n_gap:
                h_note(f"기록이 <b>{n_gap}곳</b>에서 크게 끊겨 있습니다. 시계열 예측을 하실 계획이면 "
                       f"이 구간을 확인해 주세요.", "warn")

    # ── 상관관계 요약 ────────────────────────────────────────────────────
    if P_bool("include_corr", True) and len(nums) >= 2:
        h_sub("⑥ 숫자 컬럼끼리의 관계 (요약)")
        cm = work[nums[:15]].corr(numeric_only=True)
        fig = go.Figure(go.Heatmap(
            z=cm.values, x=[str(c) for c in cm.columns], y=[str(c) for c in cm.index],
            colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="상관계수", thickness=13, len=0.8),
            hovertemplate="%{y} ↔ %{x}<br>상관계수 %{z:.2f}<extra></extra>"))
        fig.update_layout(title="상관관계 한눈에 보기", height=max(360, 34 * len(cm) + 190),
                          xaxis=dict(tickangle=-40, showgrid=False), yaxis=dict(autorange="reversed", showgrid=False))
        show(fig)
        h_note("더 자세한 관계 분석(상위 조합 순위·산점도)은 포털에서 "
               "<b>[다중 변수 상관관계 분석]</b> 을 선택해 실행하세요.")

    # ── 결측치 처리 옵션 ─────────────────────────────────────────────────
    if strategy == "drop":
        before = len(work)
        work = work.dropna()
        h_note(f"결측이 있는 행을 제거했습니다: {before:,}행 → {len(work):,}행", "warn")
    elif strategy == "fill_median":
        for c in nums:
            work[c] = work[c].fillna(work[c].median())
        h_note("숫자 컬럼의 결측치를 각 컬럼의 중앙값으로 채웠습니다.", "warn")

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    tips.append(f"이 데이터는 <b>{len(work):,}건</b>의 기록과 <b>{work.shape[1]}개</b> 항목으로 이루어져 있습니다.")
    if n_dup:
        tips.append(f"완전히 똑같은 행이 <b>{n_dup:,}건</b> 있습니다. 중복 입력이 아닌지 확인해 보세요.")
    hi_miss = [c for c in work.columns if work[c].isna().mean() > 0.5]
    if hi_miss:
        tips.append(f"절반 이상이 비어 있는 컬럼: <b>{_html.escape(', '.join(map(str, hi_miss[:5])))}</b> "
                    f"— 분석에서 빼는 것을 권합니다.")
    const_cols = [c for c in work.columns if work[c].nunique(dropna=True) <= 1]
    if const_cols:
        tips.append(f"값이 항상 같아 분석에 도움이 되지 않는 컬럼: "
                    f"<b>{_html.escape(', '.join(map(str, const_cols[:5])))}</b>")
    skewed = [c for c in nums if abs(work[c].skew()) > 2]
    if skewed:
        tips.append(f"분포가 한쪽으로 심하게 쏠린 컬럼: <b>{_html.escape(', '.join(map(str, skewed[:5])))}</b> "
                    f"— 평균보다 <b>중앙값</b>으로 보는 편이 정확합니다.")
    out_rank = sorted(((c, _iqr_outliers(work[c])[0]) for c in nums), key=lambda x: -x[1])
    if out_rank and out_rank[0][1] > 0:
        tips.append(f"튀는 값이 가장 많은 컬럼은 <b>{_html.escape(str(out_rank[0][0]))}</b>"
                    f"({out_rank[0][1]:,}건)입니다. 이상 탐지 분석을 이어서 해보시면 좋습니다.")
    tips.append("다음 단계 추천: 관계를 보고 싶다면 <b>상관관계 분석</b>, 시간 흐름이 있다면 "
                "<b>시계열 예측</b>, 튀는 값이 궁금하면 <b>이상 탐지</b>를 선택해 보세요.")
    h_insight(tips)

    RESULT_FILES.append(("컬럼요약.csv", overview))
    return work


def run_eda_corr():
    """[탐색적 분석] 다중 변수 상관관계 분석 (Heatmap)."""
    method = P_str("corr_method", "pearson")
    top_n = P_int("top_n_pairs", 12)
    min_abs = P_float("min_abs_corr", 0.3)
    do_cluster = P_bool("cluster_order", True)
    drop_const = P_bool("drop_constant", True)
    mname = {"pearson": "피어슨(직선 관계)", "spearman": "스피어만(순위 관계)",
             "kendall": "켄달(순위 일치도)"}.get(method, method)

    h_title("🔗 다중 변수 상관관계 분석",
            f"계산 방식: {mname} · 파일: {_html.escape(str(DATA_SOURCE_NAME))}")

    cols = pick_features()
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if drop_const:
        dropped = [c for c in cols if df[c].nunique(dropna=True) <= 1]
        cols = [c for c in cols if c not in dropped]
        if dropped:
            h_note(f"값이 하나뿐이라 관계를 계산할 수 없는 컬럼을 제외했습니다: "
                   f"<b>{_html.escape(', '.join(map(str, dropped)))}</b>")
    if len(cols) < 2:
        h_note("상관관계를 계산하려면 숫자 컬럼이 최소 2개 필요합니다. "
               "포털에서 '분석에 사용할 컬럼들'을 다시 지정해 주세요.", "crit")
        return None

    data = df[cols].apply(pd.to_numeric, errors="coerce")
    cm = data.corr(method=method)

    # 비슷한 변수끼리 묶어 정렬하면 덩어리(그룹)가 눈에 보입니다.
    order = list(cm.columns)
    if do_cluster and len(cols) >= 3:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            dist = 1 - cm.abs().fillna(0).values
            np.fill_diagonal(dist, 0)
            dist = (dist + dist.T) / 2
            link = linkage(squareform(dist, checks=False), method="average")
            order = [cm.columns[i] for i in leaves_list(link)]
            cm = cm.loc[order, order]
        except Exception:
            pass

    h_metrics([
        ("분석 컬럼", f"{len(cols)}개", "숫자형만 사용"),
        ("사용 행 수", f"{len(data.dropna()):,}", f"전체 {len(df):,}행 중"),
        ("가능한 조합", f"{len(cols)*(len(cols)-1)//2:,}쌍", "서로 다른 두 컬럼"),
        ("기준값", f"|r| ≥ {min_abs:g}", "주목할 관계 판정선"),
    ])

    # ── 히트맵 ───────────────────────────────────────────────────────────
    h_sub("① 상관관계 히트맵")
    annotate = len(cm) <= 14
    fig = go.Figure(go.Heatmap(
        z=cm.values, x=[str(c) for c in cm.columns], y=[str(c) for c in cm.index],
        colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
        text=np.round(cm.values, 2) if annotate else None,
        texttemplate="%{text}" if annotate else None,
        textfont=dict(size=10.5),
        colorbar=dict(title=dict(text="상관계수", side="right"), thickness=13, len=0.85,
                      tickvals=[-1, -0.5, 0, 0.5, 1],
                      ticktext=["-1<br>정반대", "-0.5", "0<br>무관", "0.5", "+1<br>같이 움직임"]),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>상관계수 %{z:.3f}<extra></extra>",
        xgap=1, ygap=1))
    fig.update_layout(title=f"상관관계 히트맵 ({mname})",
                      height=max(420, 36 * len(cm) + 200),
                      xaxis=dict(tickangle=-40, showgrid=False, side="bottom"),
                      yaxis=dict(autorange="reversed", showgrid=False))
    show(fig)
    h_note("🔵 <b>파란색</b>은 “한쪽이 오르면 다른 쪽도 오른다”, 🔴 <b>붉은색</b>은 “한쪽이 오르면 다른 쪽은 내린다”, "
           "<b>회색</b>은 “서로 상관없다”는 뜻입니다. 색이 진할수록 관계가 강합니다.")

    # ── 상위 조합 ────────────────────────────────────────────────────────
    h_sub("② 관계가 강한 조합 순위")
    pairs = []
    cl = list(cm.columns)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            r = cm.iloc[i, j]
            if pd.notna(r):
                pairs.append((cl[i], cl[j], float(r)))
    pairs.sort(key=lambda x: -abs(x[2]))
    top = pairs[:top_n]

    def _strength(r):
        a = abs(r)
        if a >= 0.9: return "매우 강함"
        if a >= 0.7: return "강함"
        if a >= 0.5: return "뚜렷함"
        if a >= 0.3: return "약함"
        return "거의 없음"

    if top:
        ptab = pd.DataFrame([{
            "순위": k + 1, "항목 A": a, "항목 B": b, "상관계수": round(r, 3),
            "방향": "같이 오름 ↗" if r > 0 else "반대로 움직임 ↘",
            "강도": _strength(r),
            "설명": f"{a}이(가) 오르면 {b}은(는) {'같이 오르는' if r > 0 else '반대로 내려가는'} 경향",
        } for k, (a, b, r) in enumerate(top)])
        h_table(ptab, f"상관계수 절댓값 기준 상위 {len(top)}쌍", max_rows=top_n)

        rev = top[::-1]
        fig = go.Figure(go.Bar(
            x=[r for _, _, r in rev],
            y=[f"{a} ↔ {b}" for a, b, _ in rev], orientation="h",
            marker=dict(color=[PALETTE[0] if r > 0 else "#b3241a" for _, _, r in rev],
                        line=dict(width=0)),
            text=[f"{r:+.2f}" for _, _, r in rev], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>상관계수 %{x:.3f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
        for xv in (min_abs, -min_abs):
            fig.add_vline(x=xv, line=dict(color=MUTED, width=1, dash="dot"))
        fig.update_layout(title=f"관계 강도 순위 (점선 = 기준값 {min_abs:g})",
                          xaxis_title="상관계수  ← 반대로 움직임 | 같이 움직임 →",
                          xaxis=dict(range=[-1.15, 1.15]),
                          height=max(300, 30 * len(rev) + 150), showlegend=False)
        show(fig)

    # ── 타겟 중심 ────────────────────────────────────────────────────────
    tcol = resolve_col(TARGET_COL) if TARGET_COL else None
    if tcol in cm.columns:
        h_sub(f"③ '{tcol}' 와(과) 가장 관련 깊은 항목")
        rel = cm[tcol].drop(labels=[tcol]).dropna().sort_values()
        fig = go.Figure(go.Bar(
            x=rel.values, y=[str(i) for i in rel.index], orientation="h",
            marker=dict(color=[PALETTE[0] if v > 0 else "#b3241a" for v in rel.values], line=dict(width=0)),
            text=[f"{v:+.2f}" for v in rel.values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>상관계수 %{x:.3f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
        fig.update_layout(title=f"'{tcol}' 기준 상관계수", xaxis_title="상관계수",
                          xaxis=dict(range=[-1.15, 1.15]),
                          height=max(300, 30 * len(rel) + 150), showlegend=False)
        show(fig)

    # ── 산점도 ───────────────────────────────────────────────────────────
    if top:
        a, b, r = top[0]
        h_sub(f"④ 가장 강한 관계 자세히 보기 — {a} ↔ {b}")
        sub = data[[a, b]].dropna()
        if len(sub) > 4000:
            sub = sub.sample(4000, random_state=42)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=sub[a], y=sub[b], mode="markers", name="실제 데이터",
            marker=dict(size=7, color=PALETTE[0], opacity=0.55,
                        line=dict(width=1, color=SURFACE)),
            hovertemplate=f"{a} %{{x:,.4g}}<br>{b} %{{y:,.4g}}<extra></extra>"))
        if len(sub) >= 3:
            k, c0 = np.polyfit(sub[a], sub[b], 1)
            xs = np.linspace(sub[a].min(), sub[a].max(), 60)
            fig.add_trace(go.Scatter(x=xs, y=k * xs + c0, mode="lines", name="추세선",
                                     line=dict(color="#b3241a", width=2),
                                     hovertemplate="추세선<extra></extra>"))
        fig.update_layout(title=f"{a} 와(과) {b} 의 관계 (상관계수 {r:+.2f})",
                          xaxis_title=str(a), yaxis_title=str(b), height=470)
        show(fig)

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    strong = [p for p in pairs if abs(p[2]) >= min_abs]
    if strong:
        a, b, r = strong[0]
        tips.append(f"가장 눈에 띄는 관계는 <b>{_html.escape(str(a))}</b> ↔ <b>{_html.escape(str(b))}</b> "
                    f"(상관계수 <b>{r:+.2f}</b>, {_strength(r)})입니다. "
                    f"한쪽이 오를 때 다른 쪽은 {'같이 오르는' if r > 0 else '내려가는'} 경향을 보입니다.")
        tips.append(f"기준값 |r| ≥ {min_abs:g} 을 넘는 조합은 총 <b>{len(strong)}쌍</b>입니다.")
    else:
        tips.append(f"기준값(|r| ≥ {min_abs:g})을 넘는 뚜렷한 관계가 없습니다. "
                    f"항목들이 서로 비교적 독립적으로 움직이고 있습니다.")
    multi = [p for p in pairs if abs(p[2]) >= 0.9]
    if multi:
        tips.append(f"⚠️ 거의 같은 정보를 담고 있는 조합이 <b>{len(multi)}쌍</b> 있습니다 "
                    f"(예: {_html.escape(str(multi[0][0]))} ↔ {_html.escape(str(multi[0][1]))}). "
                    f"예측 모델을 만들 때는 둘 중 하나만 쓰는 편이 좋습니다.")
    if tcol in cm.columns:
        rel = cm[tcol].drop(labels=[tcol]).dropna()
        if len(rel):
            best = rel.abs().idxmax()
            tips.append(f"타겟 <b>{_html.escape(str(tcol))}</b> 에 가장 큰 영향을 주는 것으로 보이는 항목은 "
                        f"<b>{_html.escape(str(best))}</b> (상관계수 {rel[best]:+.2f})입니다.")
    tips.append("❗ <b>상관관계는 인과관계가 아닙니다.</b> 두 값이 함께 움직인다고 해서 "
                "하나가 다른 하나의 <i>원인</i>이라는 뜻은 아닙니다. 숨은 제3의 요인이 있을 수 있습니다.")
    tips.append("다음 단계 추천: 관계가 확인된 항목들로 <b>수치 예측(Random Forest)</b> 을 돌리면 "
                "실제로 얼마나 예측에 도움이 되는지 확인할 수 있습니다.")
    h_insight(tips)

    RESULT_FILES.append(("상관계수_행렬.csv", cm.reset_index().rename(columns={"index": "컬럼"})))
    if top:
        RESULT_FILES.append(("상관관계_상위조합.csv", ptab))
    return cm


FREQ_DAYS = {"D": 1.0, "W": 7.0, "MS": 30.44, "ME": 30.44, "QS": 91.3,
             "h": 1 / 24, "H": 1 / 24, "15min": 1 / 96, "min": 1 / 1440}
FREQ_SEASON = {"D": 7, "W": 52, "MS": 12, "ME": 12, "QS": 4, "h": 24, "H": 24, "15min": 96}


def _prepare_series(date_col, value_col, resample=None, agg="mean"):
    """날짜 컬럼 + 값 컬럼 → 날짜 오름차순의 깨끗한 시계열로 정리합니다."""
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
    bad = s[date_col].isna().sum()
    s = s.dropna(subset=[date_col]).sort_values(date_col)
    n_missing_val = int(s[value_col].isna().sum())
    s = s.dropna(subset=[value_col])
    if bad:
        print(f"  ⚠️  날짜로 읽을 수 없는 행 {bad:,}건을 제외했습니다.")
    if n_missing_val:
        print(f"  ⚠️  값이 비어 있는 행 {n_missing_val:,}건을 제외했습니다.")

    # 같은 시각에 여러 기록이 있으면 하나로 합칩니다 (예측 모델의 필수 조건).
    dup = int(s[date_col].duplicated().sum())
    if dup:
        s = s.groupby(date_col, as_index=False)[value_col].agg(agg)
        print(f"  ℹ️  같은 시각의 기록 {dup:,}건을 '{agg}' 기준으로 합쳤습니다.")

    if resample and resample != "none":
        s = (s.set_index(date_col)[value_col]
               .resample(normalize_freq(resample)).agg(agg).dropna().reset_index())
    return s.reset_index(drop=True)


def _detect_season_period(series, index):
    """반복 주기(계절 길이)를 자동으로 추정합니다."""
    try:
        inferred = pd.infer_freq(index)
    except Exception:
        inferred = None
    if inferred:
        base = "".join(ch for ch in inferred.split("-")[0] if not ch.isdigit()) or inferred
        for key, val in FREQ_SEASON.items():
            if base.upper().startswith(key.upper()):
                return val
    # 자기상관(ACF)에서 가장 높은 봉우리를 찾습니다.
    y = np.asarray(series, dtype=float)
    y = y - y.mean()
    n = len(y)
    if n < 20 or np.allclose(y, 0):
        return 0
    max_lag = min(n // 3, 400)
    denom = np.dot(y, y)
    acf = [np.dot(y[:-l], y[l:]) / denom for l in range(2, max_lag)]
    if not acf:
        return 0
    best = int(np.argmax(acf)) + 2
    return best if acf[best - 2] > 0.2 else 0


def run_ts_forecast():
    """[시계열 분석] 미래 추세 및 계절성 예측 (Prophet)."""
    h_title("🔮 미래 추세 및 계절성 예측 (Prophet)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    if not HAS_PROPHET:
        h_note("Prophet 라이브러리를 불러오지 못했습니다. 상단 메뉴 <b>[런타임] → [세션 다시 시작 및 모두 실행]</b> "
               "을 눌러 다시 시도해 주세요.", "crit")
        return None

    import logging
    from prophet import Prophet
    # Prophet 내부 엔진이 쏟아내는 기술적인 로그를 감춥니다. (사용자에게는 불필요)
    for lg in ("prophet", "prophet.models", "cmdstanpy", "cmdstanpy.model"):
        _log = logging.getLogger(lg)
        _log.setLevel(logging.CRITICAL)
        _log.handlers.clear()
        _log.propagate = False
        _log.disabled = True

    date_col = resolve_col(DATE_COL, kind="date", required=True, purpose="날짜 컬럼")
    value_col = resolve_col(TARGET_COL, kind="number", required=True, purpose="예측할 컬럼")

    periods = max(1, P_int("periods", 30))
    freq = normalize_freq(P_str("freq", "D"))
    mode = P_str("seasonality_mode", "additive")
    cps = P_float("trend_flex", 0.05)
    use_holidays = P_bool("kr_holidays", True)
    floor_zero = P_bool("floor_zero", True)
    do_cv = P_bool("cv_eval", True)
    unit = FREQ_LABEL.get(freq, "기간")

    s = _prepare_series(date_col, value_col)
    if len(s) < 12:
        h_note(f"예측에 쓸 수 있는 데이터가 {len(s)}건뿐입니다. 최소 12건 이상 필요합니다.", "crit")
        return None

    hist = s.rename(columns={date_col: "ds", value_col: "y"})[["ds", "y"]]
    span_days = max((hist["ds"].max() - hist["ds"].min()).days, 1)

    # 값이 음수가 될 수 없다면 로그 변환으로 안정성과 정확도를 함께 올립니다.
    log_mode = bool(floor_zero and (hist["y"] >= 0).all())
    fit_df = hist.copy()
    if log_mode:
        fit_df["y"] = np.log1p(fit_df["y"])

    m = Prophet(seasonality_mode=mode, changepoint_prior_scale=cps,
                interval_width=0.8,
                yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto")
    if use_holidays:
        try:
            m.add_country_holidays(country_name="KR")
        except Exception as exc:
            print(f"  ⚠️  공휴일 정보를 불러오지 못했습니다({exc}). 공휴일 없이 진행합니다.")
    print("  · 모델을 학습하는 중입니다… (10~40초)")
    m.fit(fit_df)

    future = m.make_future_dataframe(periods=periods, freq=freq)
    fcst = m.predict(future)
    if log_mode:
        for c in ("yhat", "yhat_lower", "yhat_upper", "trend"):
            if c in fcst.columns:
                fcst[c] = np.expm1(fcst[c])
    if floor_zero:
        for c in ("yhat", "yhat_lower", "yhat_upper"):
            fcst[c] = fcst[c].clip(lower=0)

    future_part = fcst[fcst["ds"] > hist["ds"].max()]
    end_pred = float(future_part["yhat"].iloc[-1]) if len(future_part) else float(fcst["yhat"].iloc[-1])
    # 마지막 값 하나는 우연히 높거나 낮을 수 있으므로, 같은 길이의 '최근 평균'과 비교합니다.
    recent_avg = float(hist["y"].tail(min(periods, len(hist))).mean())
    future_avg = float(future_part["yhat"].mean()) if len(future_part) else float("nan")
    change = (future_avg - recent_avg) / abs(recent_avg) * 100 if recent_avg else np.nan

    h_metrics([
        ("학습 데이터", f"{len(hist):,}건", f"{hist['ds'].min():%Y-%m-%d} ~ {hist['ds'].max():%Y-%m-%d}"),
        ("예측 구간", f"{periods}{unit}", f"{future_part['ds'].max():%Y-%m-%d} 까지" if len(future_part) else "—"),
        ("예측 구간 평균", fmt(future_avg),
         f"최근 {min(periods, len(hist))}{unit} 평균({fmt(recent_avg)}) 대비 {change:+.1f}%"
         if pd.notna(change) else str(value_col)),
        ("예측 최종값", fmt(end_pred), f"{future_part['ds'].max():%Y-%m-%d} 시점" if len(future_part) else "—"),
    ])

    # ── 예측 그래프 ──────────────────────────────────────────────────────
    h_sub("① 예측 결과")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fcst["ds"], y=fcst["yhat_upper"], mode="lines", line=dict(width=0),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=fcst["ds"], y=fcst["yhat_lower"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(42,120,214,0.16)",
        name="예측 범위 (80% 신뢰구간)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=fcst["ds"], y=fcst["yhat"], mode="lines", name="예측 값",
        line=dict(color=PALETTE[0], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>예측 %{y:,.4g}<extra></extra>"))
    # 실제 값은 예측선 '위에' 점으로 찍어야 둘을 비교할 수 있습니다.
    fig.add_trace(go.Scattergl(
        x=hist["ds"], y=hist["y"], mode="markers", name="실제 값",
        marker=dict(size=3.5, color=INK, opacity=0.5),
        hovertemplate="%{x|%Y-%m-%d}<br>실제 %{y:,.4g}<extra></extra>"))
    fig.add_vline(x=hist["ds"].max(), line=dict(color=MUTED, width=1.2, dash="dot"))
    fig.add_annotation(x=hist["ds"].max(), y=1.0, yref="paper", text="여기서부터 예측",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(size=11, color=MUTED))
    fig.update_layout(title=f"'{value_col}' 예측 — 앞으로 {periods}{unit}",
                      xaxis_title=str(date_col), yaxis_title=str(value_col), height=500,
                      hovermode="x unified",
                      xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)))
    show(fig)
    h_note("굵은 파란 선이 <b>예측값</b>, 옅은 파란 띠는 <b>예측이 흔들릴 수 있는 범위</b>입니다. "
           "띠가 넓을수록 “확신이 덜하다”는 뜻이니, 계획을 세울 때는 띠의 아래쪽(보수적)도 함께 보세요.")

    # ── 구성 요소 분해 ───────────────────────────────────────────────────
    h_sub("② 예측을 이루는 요소들")
    comps = [("trend", "장기 추세 — 큰 흐름이 오르는지 내리는지")]
    if "weekly" in fcst.columns:
        comps.append(("weekly", "요일 패턴 — 어떤 요일에 높고 낮은지"))
    if "yearly" in fcst.columns:
        comps.append(("yearly", "연간 계절 패턴 — 어떤 시기에 높고 낮은지"))
    if "holidays" in fcst.columns and fcst["holidays"].abs().sum() > 0:
        comps.append(("holidays", "공휴일 효과"))

    fig = make_subplots(rows=len(comps), cols=1, shared_xaxes=False,
                        subplot_titles=[c[1] for c in comps],
                        vertical_spacing=max(0.08, 0.26 / max(len(comps), 1)))
    weekday_ko = ["월", "화", "수", "목", "금", "토", "일"]
    for i, (key, _) in enumerate(comps, start=1):
        color = PALETTE[(i - 1) % len(PALETTE)]
        if key == "weekly":
            tmp = fcst[["ds", "weekly"]].copy()
            tmp["wd"] = tmp["ds"].dt.dayofweek
            g = tmp.groupby("wd")["weekly"].mean().reindex(range(7))
            fig.add_trace(go.Bar(x=weekday_ko, y=g.values, marker=dict(color=color, line=dict(width=0)),
                                 hovertemplate="%{x}요일<br>평균 효과 %{y:+,.3g}<extra></extra>",
                                 showlegend=False), row=i, col=1)
        elif key == "yearly":
            tmp = fcst[["ds", "yearly"]].copy()
            tmp["m"] = tmp["ds"].dt.month
            g = tmp.groupby("m")["yearly"].mean().reindex(range(1, 13))
            fig.add_trace(go.Scatter(x=[f"{k}월" for k in range(1, 13)], y=g.values, mode="lines+markers",
                                     line=dict(color=color, width=2), marker=dict(size=8),
                                     hovertemplate="%{x}<br>평균 효과 %{y:+,.3g}<extra></extra>",
                                     showlegend=False), row=i, col=1)
        elif key == "holidays":
            tmp = fcst[["ds", "holidays"]].copy()
            tmp = tmp[tmp["holidays"].abs() > 1e-9]
            fig.add_trace(go.Bar(x=tmp["ds"], y=tmp["holidays"],
                                 marker=dict(color=color, line=dict(width=0)),
                                 hovertemplate="%{x|%Y-%m-%d}<br>효과 %{y:+,.3g}<extra></extra>",
                                 showlegend=False), row=i, col=1)
        else:
            fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst[key], mode="lines",
                                     line=dict(color=color, width=2),
                                     hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.4g}<extra></extra>",
                                     showlegend=False), row=i, col=1)
    fig.update_layout(height=225 * len(comps) + 90, title="패턴 분해")
    fig.update_annotations(font=dict(size=12.5, color=INK_2))
    show(fig)

    # ── 정확도 검증 ──────────────────────────────────────────────────────
    mape = None
    if do_cv:
        h_sub("③ 예측 정확도 검증 (과거 데이터로 시험)")
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            step = FREQ_DAYS.get(freq, 1.0)
            horizon_days = max(1, int(round(periods * step)))
            horizon_days = min(horizon_days, max(1, span_days // 5))
            if span_days < horizon_days * 4:
                raise ValueError("검증에 필요한 과거 데이터가 부족합니다.")
            initial_days = max(horizon_days * 3, span_days - horizon_days * 4)
            print("  · 과거 시점으로 돌아가 예측 성적을 매기는 중입니다… (30초~2분)")
            cv = cross_validation(m, initial=f"{initial_days} days",
                                  period=f"{horizon_days} days",
                                  horizon=f"{horizon_days} days", disable_tqdm=True)
            if log_mode:
                for c in ("yhat", "y", "yhat_lower", "yhat_upper"):
                    if c in cv.columns:
                        cv[c] = np.expm1(cv[c])
            perf = performance_metrics(cv, rolling_window=1)
            mape = float(perf["mape"].iloc[0]) * 100 if "mape" in perf.columns else None
            rmse = float(perf["rmse"].iloc[0]) if "rmse" in perf.columns else None
            mae = float(perf["mae"].iloc[0]) if "mae" in perf.columns else None
            cover = float(perf["coverage"].iloc[0]) * 100 if "coverage" in perf.columns else None
            grade = ("매우 우수" if mape is not None and mape < 10 else
                     "양호" if mape is not None and mape < 20 else
                     "보통" if mape is not None and mape < 50 else "주의 필요")
            h_metrics([
                ("평균 오차율 (MAPE)", f"{mape:.1f}%" if mape is not None else "—", f"판정: {grade}"),
                ("평균 절대오차 (MAE)", fmt(mae), "실제와 예측의 평균 차이"),
                ("RMSE", fmt(rmse), "큰 오차에 민감한 지표"),
                ("범위 적중률", f"{cover:.0f}%" if cover is not None else "—", "실제값이 예측 범위 안에 든 비율"),
            ])
            cvp = cv.copy()
            cvp["오차"] = cvp["yhat"] - cvp["y"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cvp["ds"], y=cvp["y"], mode="markers", name="실제 값",
                                     marker=dict(size=6, color=INK_2, opacity=0.55,
                                                 line=dict(width=1, color=SURFACE)),
                                     hovertemplate="%{x|%Y-%m-%d}<br>실제 %{y:,.4g}<extra></extra>"))
            fig.add_trace(go.Scatter(x=cvp["ds"], y=cvp["yhat"], mode="markers", name="그때 예측했다면",
                                     marker=dict(size=6, color=PALETTE[0], opacity=0.75,
                                                 line=dict(width=1, color=SURFACE)),
                                     hovertemplate="%{x|%Y-%m-%d}<br>예측 %{y:,.4g}<extra></extra>"))
            fig.update_layout(title="과거 시점 재현 검증 — 예측이 실제를 얼마나 따라갔는가",
                              xaxis_title=str(date_col), yaxis_title=str(value_col), height=430)
            show(fig)
        except Exception as exc:
            h_note(f"정확도 검증을 건너뜁니다: {_html.escape(str(exc))}<br>"
                   f"보통 <b>데이터 기간이 짧을 때</b> 발생합니다. 예측 결과 자체는 그대로 사용하실 수 있습니다.", "warn")

    # ── 예측표 ───────────────────────────────────────────────────────────
    h_sub("④ 예측값 표")
    out = future_part[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out.columns = ["날짜", "예측값", "예측 최소", "예측 최대"]
    out["날짜"] = out["날짜"].dt.strftime("%Y-%m-%d %H:%M" if FREQ_DAYS.get(freq, 1) < 1 else "%Y-%m-%d")
    for c in ("예측값", "예측 최소", "예측 최대"):
        out[c] = out[c].round(3)
    h_table(out, f"앞으로 {periods}{unit} 예측 (전체는 CSV로 내려받습니다)", max_rows=20, highlight="날짜")

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    trend_first = float(fcst["trend"].iloc[0])
    trend_last = float(fcst["trend"].iloc[-1])
    direction = "상승" if trend_last > trend_first * 1.01 else "하락" if trend_last < trend_first * 0.99 else "횡보"
    tips.append(f"전체 <b>큰 흐름은 {direction}세</b>입니다. "
                f"({fcst['ds'].iloc[0]:%Y-%m-%d} {fmt(trend_first)} → {fcst['ds'].iloc[-1]:%Y-%m-%d} {fmt(trend_last)})")
    if len(future_part):
        tips.append(f"앞으로 {periods}{unit} 동안 <b>{value_col}</b> 은(는) 평균 <b>{fmt(future_avg)}</b> 수준으로, "
                    f"최근 같은 기간 평균({fmt(recent_avg)}) 대비 <b>{change:+.1f}%</b> 로 예상됩니다.")
        tips.append(f"구간 마지막 시점({future_part['ds'].max():%Y-%m-%d})의 예상값은 <b>{fmt(end_pred)}</b> "
                    f"이며, 가능 범위는 {fmt(float(future_part['yhat_lower'].iloc[-1]))} ~ "
                    f"{fmt(float(future_part['yhat_upper'].iloc[-1]))} 입니다. "
                    f"계획을 세울 때는 범위의 아래쪽도 함께 고려하세요.")
    if "weekly" in fcst.columns:
        tmp = fcst[["ds", "weekly"]].copy()
        tmp["wd"] = tmp["ds"].dt.dayofweek
        g = tmp.groupby("wd")["weekly"].mean()
        if len(g):
            tips.append(f"요일별로 보면 <b>{weekday_ko[int(g.idxmax())]}요일</b>이 가장 높고 "
                        f"<b>{weekday_ko[int(g.idxmin())]}요일</b>이 가장 낮습니다. "
                        f"인력·재고 배치에 참고하세요.")
    if "yearly" in fcst.columns:
        tmp = fcst[["ds", "yearly"]].copy()
        tmp["m"] = tmp["ds"].dt.month
        g = tmp.groupby("m")["yearly"].mean()
        if len(g):
            tips.append(f"연중 <b>{int(g.idxmax())}월</b>에 가장 높고 <b>{int(g.idxmin())}월</b>에 가장 낮은 "
                        f"계절 패턴이 있습니다.")
    if mape is not None:
        judge = ("이 정도면 실무에서 믿고 쓸 수 있는 수준입니다." if mape < 10 else
                 "참고 지표로 쓰기에 무리 없는 수준입니다." if mape < 20 else
                 "방향성 참고용으로만 쓰시고, 중요한 의사결정에는 다른 근거를 함께 보세요.")
        tips.append(f"과거 데이터로 검증한 <b>평균 오차율은 {mape:.1f}%</b> 입니다. {judge}")
    tips.append("❗ 예측은 <b>과거의 패턴이 계속된다는 가정</b> 위에 있습니다. "
                "신제품 출시·설비 교체·정책 변화처럼 과거에 없던 사건은 반영되지 않습니다.")
    h_insight([t for t in tips if t])

    RESULT_FILES.append(("예측결과.csv", out))
    return fcst


def run_ts_smooth():
    """[시계열 분석] 데이터 평활화 및 트렌드 추출."""
    h_title("〰️ 데이터 평활화 및 트렌드 추출",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    date_col = resolve_col(DATE_COL, kind="date", required=True, purpose="날짜 컬럼")
    value_col = resolve_col(TARGET_COL, kind="number", required=True, purpose="분석할 컬럼")
    method = P_str("method", "all")
    window = max(2, P_int("window", 7))
    stl_period = P_int("stl_period", 0)
    resample = P_str("resample", "none")
    agg = P_str("agg", "mean")

    s = _prepare_series(date_col, value_col, resample=resample, agg=agg)
    if len(s) < window * 2:
        h_note(f"데이터가 {len(s)}건뿐이라 평활 구간({window})에 비해 너무 짧습니다. "
               f"평활 구간을 줄이거나 더 긴 기간의 데이터를 사용하세요.", "warn")
    if len(s) < 5:
        h_note("분석할 수 있는 데이터가 너무 적습니다.", "crit")
        return None

    ser = s.set_index(date_col)[value_col].astype(float)
    ma = ser.rolling(window=window, min_periods=max(2, window // 2)).mean()
    ewm = ser.ewm(span=window, adjust=False).mean()

    period = stl_period if stl_period >= 2 else _detect_season_period(ser.values, ser.index)
    h_metrics([
        ("데이터 수", f"{len(ser):,}건", f"{ser.index.min():%Y-%m-%d} ~ {ser.index.max():%Y-%m-%d}"),
        ("평활 구간", f"{window}", "몇 개 값을 묶어 평균내는지"),
        ("추정 계절 주기", f"{period}" if period >= 2 else "없음", "반복되는 패턴의 길이"),
        ("원본 변동성", fmt(float(ser.std())), "표준편차"),
    ])

    # ── 평활 비교 ────────────────────────────────────────────────────────
    if method in ("all", "ma", "ewm"):
        h_sub("① 잡음 걷어내기 — 원본과 평활선 비교")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ser.index, y=ser.values, mode="lines", name="원본 (잡음 포함)",
                                 line=dict(color="#c3c2b7", width=1),
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.4g}<extra></extra>"))
        if method in ("all", "ma"):
            fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines",
                                     name=f"이동평균 ({window}구간)",
                                     line=dict(color=PALETTE[0], width=2.2),
                                     hovertemplate="%{x|%Y-%m-%d}<br>이동평균 %{y:,.4g}<extra></extra>"))
        if method in ("all", "ewm"):
            fig.add_trace(go.Scatter(x=ewm.index, y=ewm.values, mode="lines",
                                     name=f"지수가중 (span={window})",
                                     line=dict(color=PALETTE[1], width=2.2),
                                     hovertemplate="%{x|%Y-%m-%d}<br>지수가중 %{y:,.4g}<extra></extra>"))
        fig.update_layout(title=f"'{value_col}' 평활 결과", xaxis_title=str(date_col),
                          yaxis_title=str(value_col), height=480, hovermode="x unified",
                          xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)))
        show(fig)
        h_note("<b>이동평균</b>은 최근 N개의 단순 평균이라 이해하기 쉽고, "
               "<b>지수가중</b>은 최근 값에 더 큰 비중을 둬서 변화에 빠르게 반응합니다. "
               "선이 너무 울퉁불퉁하면 평활 구간을 늘리고, 변화를 놓치는 느낌이면 줄이세요.")

    # ── STL 분해 ─────────────────────────────────────────────────────────
    seasonal_strength = None
    if method in ("all", "stl") and period >= 2 and len(ser) >= period * 2 + 1:
        h_sub("② 추세 · 계절 · 잔차로 분리하기 (STL)")
        try:
            from statsmodels.tsa.seasonal import STL
            work = ser.astype(float).interpolate(limit_direction="both")
            res = STL(work.values, period=int(period), robust=True).fit()
            parts = [("원본", work.values, "#c3c2b7"),
                     ("추세 (큰 흐름)", res.trend, PALETTE[0]),
                     ("계절 (반복 패턴)", res.seasonal, PALETTE[2]),
                     ("잔차 (설명 안 되는 부분)", res.resid, PALETTE[7])]
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=[p[0] for p in parts], vertical_spacing=0.055)
            for i, (name, vals, color) in enumerate(parts, start=1):
                fig.add_trace(go.Scatter(x=work.index, y=vals, mode="lines", name=name,
                                         line=dict(color=color, width=1.6 if i > 1 else 1),
                                         showlegend=False,
                                         hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.4g}<extra></extra>"),
                              row=i, col=1)
            fig.update_layout(height=760, title=f"STL 분해 (계절 주기 {int(period)})")
            fig.update_annotations(font=dict(size=12.5, color=INK_2))
            show(fig)

            var_resid = float(np.nanvar(res.resid))
            var_sr = float(np.nanvar(res.resid + res.seasonal))
            var_tr = float(np.nanvar(res.resid + res.trend))
            seasonal_strength = max(0.0, 1 - var_resid / var_sr) if var_sr > 0 else 0.0
            trend_strength = max(0.0, 1 - var_resid / var_tr) if var_tr > 0 else 0.0
            h_metrics([
                ("계절성 강도", f"{seasonal_strength*100:.0f}%",
                 "높을수록 반복 패턴이 뚜렷" ),
                ("추세 강도", f"{trend_strength*100:.0f}%", "높을수록 방향성이 뚜렷"),
                ("잔차 표준편차", fmt(float(np.nanstd(res.resid))), "설명되지 않는 흔들림"),
                ("계절 주기", f"{int(period)}", "자동 추정" if stl_period < 2 else "직접 지정"),
            ])
            resid = pd.Series(res.resid, index=work.index)
            thr = 3 * float(np.nanstd(res.resid))
            odd = resid[resid.abs() > thr]
            if len(odd):
                h_note(f"패턴으로 설명되지 않는 <b>특이한 시점이 {len(odd)}건</b> 발견되었습니다. "
                       f"가장 큰 시점: <b>{odd.abs().idxmax():%Y-%m-%d}</b>. "
                       f"이 날 무슨 일이 있었는지 확인해 보세요.", "warn")
        except Exception as exc:
            h_note(f"STL 분해를 건너뜁니다: {_html.escape(str(exc))}", "warn")
    elif method in ("all", "stl"):
        h_note("반복되는 계절 주기를 찾지 못해 STL 분해를 건너뜁니다. "
               "주기를 아신다면 포털에서 <b>계절 주기</b> 값을 직접 지정해 보세요. (일별 주간 패턴 = 7, 월별 연간 = 12)",
               "warn")

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    half = max(2, len(ser) // 2)
    first_avg, last_avg = float(ser.iloc[:half].mean()), float(ser.iloc[-half:].mean())
    diff_pct = (last_avg - first_avg) / abs(first_avg) * 100 if first_avg else np.nan
    if pd.notna(diff_pct):
        word = "올라가는" if diff_pct > 2 else "내려가는" if diff_pct < -2 else "큰 변화 없이 유지되는"
        tips.append(f"전반부 평균 <b>{fmt(first_avg)}</b> → 후반부 평균 <b>{fmt(last_avg)}</b> "
                    f"(<b>{diff_pct:+.1f}%</b>) 로, 전체적으로 <b>{word}</b> 흐름입니다.")
    noise_ratio = float(ser.std() / abs(ser.mean())) if ser.mean() else np.nan
    if pd.notna(noise_ratio):
        tips.append(f"값의 흔들림 정도(변동계수)는 <b>{noise_ratio*100:.0f}%</b> 입니다. "
                    + ("들쭉날쭉이 심한 편이라 평활선으로 보는 것이 훨씬 읽기 좋습니다."
                       if noise_ratio > 0.3 else "비교적 안정적인 데이터입니다."))
    if seasonal_strength is not None:
        tips.append(f"반복 패턴(계절성)의 강도는 <b>{seasonal_strength*100:.0f}%</b> 입니다. "
                    + ("패턴이 뚜렷하므로 <b>시계열 예측(Prophet)</b> 을 돌리면 좋은 결과를 기대할 수 있습니다."
                       if seasonal_strength > 0.4 else
                       "반복 패턴이 약해 예측 정확도는 제한적일 수 있습니다."))
    tips.append("평활선은 <b>지나간 값을 정리한 것</b>이지 미래를 알려주지 않습니다. "
                "미래 값이 필요하면 포털에서 <b>미래 추세 및 계절성 예측(Prophet)</b> 을 선택하세요.")
    h_insight(tips)

    out = pd.DataFrame({str(date_col): ser.index, str(value_col): ser.values,
                        f"이동평균_{window}": ma.values, f"지수가중_{window}": ewm.values})
    RESULT_FILES.append(("평활화_결과.csv", out))
    return out


def _impute_median(frame):
    """모델은 빈칸을 못 다루므로 중앙값으로 채웁니다."""
    out = frame.copy()
    filled = {}
    for c in out.columns:
        n_na = int(out[c].isna().sum())
        if n_na:
            med = out[c].median()
            out[c] = out[c].fillna(0 if pd.isna(med) else med)
            filled[c] = n_na
    return out, filled


def run_anomaly_iforest():
    """[이상 탐지] 장비 불량 및 스파이크 이상치 감지 (Isolation Forest)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    h_title("🚨 이상치 감지 (Isolation Forest)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    contamination = min(max(P_float("contamination", 2.0) / 100.0, 0.001), 0.45)
    n_est = max(50, P_int("n_estimators", 200))
    do_scale = P_bool("scale", True)
    top_n = max(5, P_int("top_n_report", 20))
    do_explain = P_bool("explain", True)

    date_col = resolve_col(DATE_COL, kind="date", purpose="날짜 컬럼") if DATE_COL else None
    target_col = resolve_col(TARGET_COL) if TARGET_COL else None

    feats = pick_features(exclude=[c for c in [date_col] if c])
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    if target_col and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]) \
            and target_col not in feats:
        feats = [target_col] + feats
    if not feats:
        h_note("이상 탐지에 쓸 숫자 컬럼이 없습니다. 포털에서 '분석에 사용할 컬럼들'을 지정해 주세요.", "crit")
        return None

    X_raw = df[feats].apply(pd.to_numeric, errors="coerce")
    X_raw, filled = _impute_median(X_raw)
    if filled:
        h_note("빈칸이 있어 각 컬럼의 중앙값으로 채운 뒤 분석했습니다: "
               + ", ".join(f"{_html.escape(str(k))} {v:,}건" for k, v in list(filled.items())[:6]), "warn")

    X = StandardScaler().fit_transform(X_raw.values) if do_scale else X_raw.values

    model = IsolationForest(n_estimators=n_est, contamination=contamination,
                            random_state=42, n_jobs=-1)
    model.fit(X)
    raw_score = model.score_samples(X)          # 값이 작을수록 이상
    is_anom = model.predict(X) == -1
    # 0~100 점으로 환산 (높을수록 더 이상함) — 비전문가도 바로 이해할 수 있게
    lo, hi = float(raw_score.min()), float(raw_score.max())
    score100 = (hi - raw_score) / (hi - lo) * 100 if hi > lo else np.zeros_like(raw_score)

    res = df.copy()
    res["이상점수"] = np.round(score100, 2)
    res["판정"] = np.where(is_anom, "이상", "정상")
    n_anom = int(is_anom.sum())

    order_idx = np.argsort(-score100)
    threshold_score = float(score100[is_anom].min()) if n_anom else float("nan")

    h_metrics([
        ("전체 데이터", f"{len(res):,}건", f"검사 항목 {len(feats)}개"),
        ("이상 판정", f"{n_anom:,}건", f"전체의 {n_anom/max(len(res),1)*100:.2f}%"),
        ("판정 기준점", f"{threshold_score:.1f}점" if n_anom else "—", "이상점수 100점 만점"),
        ("최고 이상점수", f"{score100.max():.1f}점", "가장 특이한 데이터"),
    ])

    # ── 시간/순서에 따른 이상치 ──────────────────────────────────────────
    h_sub("① 언제 이상이 있었나")
    primary = target_col if (target_col in feats) else feats[0]
    if date_col:
        xs = pd.to_datetime(df[date_col], errors="coerce")
        x_title, x_hover = str(date_col), "%{x|%Y-%m-%d %H:%M}"
    else:
        xs = pd.Series(np.arange(len(df)), index=df.index)
        x_title, x_hover = "데이터 순번", "%{x}번째"

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=xs, y=X_raw[primary], mode="lines+markers", name="전체 흐름",
        line=dict(color="#c3c2b7", width=1),
        marker=dict(size=4, color="#c3c2b7"),
        hovertemplate=f"{x_hover}<br>{primary} %{{y:,.4g}}<extra></extra>"))
    fig.add_trace(go.Scattergl(
        x=xs[is_anom], y=X_raw[primary][is_anom], mode="markers", name="이상 판정",
        marker=dict(size=11, color="#d03b3b", symbol="x-thin",
                    line=dict(width=2.2, color="#d03b3b")),
        hovertemplate=f"{x_hover}<br>{primary} %{{y:,.4g}}<br><b>이상</b><extra></extra>"))
    fig.update_layout(title=f"'{primary}' 흐름 위에 표시한 이상 지점 ({n_anom:,}건)",
                      xaxis_title=x_title, yaxis_title=str(primary), height=470,
                      xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)))
    show(fig)

    # 항목별로 나눠 보기
    multi = [c for c in feats if c != primary][:3]
    if multi:
        fig = make_subplots(rows=len(multi), cols=1, shared_xaxes=True,
                            subplot_titles=[str(c) for c in multi], vertical_spacing=0.07)
        for i, c in enumerate(multi, start=1):
            fig.add_trace(go.Scattergl(x=xs, y=X_raw[c], mode="lines", showlegend=False,
                                       line=dict(color="#c3c2b7", width=1),
                                       hovertemplate=f"{x_hover}<br>%{{y:,.4g}}<extra></extra>"), row=i, col=1)
            fig.add_trace(go.Scattergl(x=xs[is_anom], y=X_raw[c][is_anom], mode="markers",
                                       showlegend=False,
                                       marker=dict(size=8, color="#d03b3b", symbol="x-thin",
                                                   line=dict(width=2, color="#d03b3b")),
                                       hovertemplate=f"{x_hover}<br>%{{y:,.4g}}<br><b>이상</b><extra></extra>"),
                          row=i, col=1)
        fig.update_layout(height=190 * len(multi) + 110, title="다른 항목에서도 같은 시점이 튀었는지 확인")
        fig.update_annotations(font=dict(size=12.5, color=INK_2))
        show(fig)

    # ── 이상점수 분포 ────────────────────────────────────────────────────
    h_sub("② 이상점수 분포")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=score100[~is_anom], nbinsx=60, name="정상",
                               marker=dict(color=PALETTE[0], line=dict(width=0)),
                               hovertemplate="점수 %{x}<br>%{y:,}건<extra></extra>"))
    fig.add_trace(go.Histogram(x=score100[is_anom], nbinsx=60, name="이상 판정",
                               marker=dict(color="#d03b3b", line=dict(width=0)),
                               hovertemplate="점수 %{x}<br>%{y:,}건<extra></extra>"))
    if n_anom:
        fig.add_vline(x=threshold_score, line=dict(color=MUTED, width=1.4, dash="dot"))
        fig.add_annotation(x=threshold_score, y=1.0, yref="paper", text="판정 기준선",
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=11, color=MUTED))
    fig.update_layout(barmode="overlay", title="이상점수가 높을수록 평소와 다른 데이터입니다",
                      xaxis_title="이상점수 (100점 만점)", yaxis_title="건수", height=380, bargap=0.02)
    fig.update_traces(opacity=0.78)
    show(fig)

    # ── 왜 이상한가 ──────────────────────────────────────────────────────
    reason_col = None
    if do_explain and n_anom:
        h_sub("③ 왜 이상으로 판정됐나 — 벗어난 항목 분석")
        normal = X_raw[~is_anom]
        mu, sd = normal.mean(), normal.std(ddof=0).replace(0, np.nan)
        z = (X_raw - mu) / sd
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
        z_anom = z[is_anom]
        contrib = z_anom.abs().mean().sort_values(ascending=False)

        fig = go.Figure(go.Bar(
            x=contrib.values[::-1], y=[str(i) for i in contrib.index[::-1]], orientation="h",
            marker=dict(color=PALETTE[0], line=dict(width=0)),
            text=[f"{v:.1f}σ" for v in contrib.values[::-1]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>평균 %{x:.2f} 표준편차만큼 벗어남<extra></extra>"))
        fig.update_layout(title="이상 데이터가 정상 범위에서 얼마나 벗어났는가 (항목별 평균)",
                          xaxis_title="정상 평균으로부터의 거리 (표준편차 배수)",
                          height=max(280, 30 * len(contrib) + 150), showlegend=False)
        show(fig)

        reason_col = str(contrib.index[0])
        h_note(f"이상 데이터들은 특히 <b>{_html.escape(reason_col)}</b> 항목에서 "
               f"정상 범위를 가장 크게 벗어났습니다 (평균 {contrib.iloc[0]:.1f}배 표준편차). "
               f"보통 2를 넘으면 “확실히 이상하다”고 봅니다.")

        # 개별 이상치마다 가장 크게 벗어난 항목을 지목
        top_reason = z_anom.abs().idxmax(axis=1)
        res.loc[is_anom, "주요 원인 항목"] = top_reason
        cause_counts = top_reason.value_counts()
        if len(cause_counts) > 1:
            fig = go.Figure(go.Bar(
                x=[str(i) for i in cause_counts.index], y=cause_counts.values,
                marker=dict(color=PALETTE[1], line=dict(width=0)),
                text=cause_counts.values, textposition="outside", cliponaxis=False,
                hovertemplate="%{x}<br>%{y:,}건이 이 항목 때문<extra></extra>"))
            fig.update_layout(title="이상 건수를 가장 많이 유발한 항목", xaxis_title=None,
                              yaxis_title="이상 건수", height=340, showlegend=False)
            show(fig)

    # ── 2차원 지도 ───────────────────────────────────────────────────────
    if len(feats) >= 2:
        h_sub("④ 데이터 전체 지도에서 본 이상치")
        try:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=42).fit_transform(
                StandardScaler().fit_transform(X_raw.values))
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=xy[~is_anom, 0], y=xy[~is_anom, 1], mode="markers", name="정상",
                marker=dict(size=6, color=PALETTE[0], opacity=0.45, line=dict(width=1, color=SURFACE)),
                hovertemplate="정상<extra></extra>"))
            fig.add_trace(go.Scattergl(
                x=xy[is_anom, 0], y=xy[is_anom, 1], mode="markers", name="이상 판정",
                marker=dict(size=11, color="#d03b3b", symbol="x-thin", line=dict(width=2.2, color="#d03b3b")),
                hovertemplate="이상<extra></extra>"))
            fig.update_layout(
                title="여러 항목을 2차원으로 압축한 지도 — 중심에서 멀수록 특이한 데이터",
                xaxis_title="주성분 1", yaxis_title="주성분 2", height=470)
            show(fig)
        except Exception as exc:
            print(f"  · 2차원 지도 생략: {exc}")

    # ── 상위 이상치 목록 ─────────────────────────────────────────────────
    h_sub(f"⑤ 가장 이상한 데이터 {min(top_n, n_anom) if n_anom else 0}건")
    if n_anom:
        cols_show = ([date_col] if date_col else []) + feats[:6] + ["이상점수"]
        if "주요 원인 항목" in res.columns:
            cols_show.append("주요 원인 항목")
        cols_show = [c for c in dict.fromkeys(cols_show) if c in res.columns]
        # 이상으로 판정된 행만, 점수가 높은 순서대로 보여줍니다.
        anom_order = [i for i in order_idx if is_anom[i]][:top_n]
        top_tbl = res.iloc[anom_order][cols_show].copy()
        top_tbl.insert(0, "순위", range(1, len(top_tbl) + 1))
        if date_col and date_col in top_tbl.columns:
            _d = pd.to_datetime(top_tbl[date_col], errors="coerce")
            _has_time = bool((_d.dt.hour.ne(0) | _d.dt.minute.ne(0)).any())
            top_tbl[date_col] = _d.dt.strftime("%Y-%m-%d %H:%M" if _has_time else "%Y-%m-%d")
        h_table(top_tbl, "이상점수가 높은 순서 — 이 행들을 먼저 확인하세요",
                max_rows=top_n, highlight="이상점수")
    else:
        h_note("설정한 기준에서는 이상치가 발견되지 않았습니다. "
               "포털에서 <b>예상 이상치 비율</b>을 조금 올려 다시 실행해 보세요.", "good")

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    tips.append(f"전체 <b>{len(res):,}건</b> 중 <b>{n_anom:,}건({n_anom/max(len(res),1)*100:.2f}%)</b>을 "
                f"평소와 다른 데이터로 판정했습니다.")
    if reason_col:
        tips.append(f"이상 판정의 가장 큰 이유는 <b>{_html.escape(reason_col)}</b> 항목입니다. "
                    f"이 항목의 설비·공정을 우선 점검해 보세요.")
    if n_anom and date_col:
        adates = pd.to_datetime(df.loc[is_anom, date_col], errors="coerce").dropna()
        if len(adates):
            by_month = adates.dt.to_period("M").value_counts().sort_values(ascending=False)
            if len(by_month):
                tips.append(f"이상이 가장 잦았던 시기는 <b>{by_month.index[0]}</b> "
                            f"({int(by_month.iloc[0]):,}건)입니다.")
            recent = adates.max()
            tips.append(f"가장 최근 이상 발생 시점은 <b>{recent:%Y-%m-%d}</b> 입니다.")
    tips.append("이상점수는 <b>“평소 패턴에서 얼마나 떨어져 있나”</b>를 나타냅니다. "
                "높은 점수가 곧 불량을 뜻하지는 않으므로, 상위 목록을 실제 기록과 대조해 확인하세요.")
    tips.append("판정이 너무 많거나 적으면 포털의 <b>예상 이상치 비율(%)</b> 을 조정해 다시 실행하면 됩니다.")
    h_insight(tips)

    RESULT_FILES.append(("이상탐지_전체결과.csv", res))
    if n_anom:
        RESULT_FILES.append(("이상탐지_이상데이터만.csv", res[is_anom].sort_values("이상점수", ascending=False)))
    return res


def _build_design_matrix(target_col, exclude=()):
    """모델에 넣을 입력표(X)를 만듭니다. 글자 컬럼은 자동으로 숫자로 바꿉니다."""
    exclude = set([c for c in exclude if c]) | {target_col}
    chosen = pick_features(exclude=exclude, max_n=60)

    # 사용자가 글자(범주) 컬럼을 직접 지정했다면 그것도 살립니다.
    if FEATURE_COLS:
        for f in FEATURE_COLS:
            c = resolve_col(f)
            if c and c not in exclude and c not in chosen:
                chosen.append(c)
    if not chosen:
        chosen = [c for c in df.columns if c not in exclude][:30]

    num_cols, cat_cols, dropped = [], [], []
    for c in chosen:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dropped.append((c, "날짜형"))
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        elif s.nunique(dropna=True) <= 15:
            cat_cols.append(c)
        else:
            dropped.append((c, "종류가 너무 많은 글자"))

    parts = []
    if num_cols:
        parts.append(df[num_cols].apply(pd.to_numeric, errors="coerce"))
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols].astype(str), prefix_sep=" = ", dtype=float))
    if not parts:
        raise ValueError("모델에 넣을 수 있는 컬럼이 없습니다. 포털에서 '분석에 사용할 컬럼들'을 지정해 주세요.")

    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.nunique(dropna=True) > 1]           # 값이 하나뿐인 컬럼 제거
    X, _ = _impute_median(X)
    if dropped:
        h_note("다음 컬럼은 모델에 넣기 어려워 제외했습니다: "
               + ", ".join(f"{_html.escape(str(c))}({why})" for c, why in dropped[:8]))
    return X, num_cols, cat_cols


def _importance_fig(names, values, title, subtitle=None, top=20):
    imp = pd.Series(values, index=names).sort_values(ascending=False).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=imp.values, y=[str(i) for i in imp.index], orientation="h",
        marker=dict(color=PALETTE[0], line=dict(width=0)),
        text=[f"{v:.3g}" for v in imp.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>중요도 %{x:.4g}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=subtitle or "중요도 (높을수록 결과에 큰 영향)",
                      height=max(300, 27 * len(imp) + 160), showlegend=False)
    return fig, imp


def run_regress_rf():
    """[수치 예측] 다중 변수 기반 결과 수치 예측 (Random Forest 회귀)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, KFold
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    h_title("🎯 결과 수치 예측 (Random Forest 회귀)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    date_col = resolve_col(DATE_COL, kind="date") if DATE_COL else None
    target_col = resolve_col(TARGET_COL, kind="number", required=True, purpose="예측할 컬럼")

    test_size = min(max(P_int("test_size", 20) / 100.0, 0.05), 0.5)
    n_est = max(50, P_int("n_estimators", 300))
    max_depth = P_int("max_depth", 0) or None
    cv_folds = P_int("cv_folds", 5)
    do_perm = P_bool("importance", True)
    log_target = P_bool("log_target", False)

    y_all = pd.to_numeric(df[target_col], errors="coerce")
    if y_all.notna().sum() < 30:
        h_note(f"'{target_col}' 을(를) 숫자로 읽을 수 있는 행이 {int(y_all.notna().sum())}건뿐입니다. "
               f"수치 예측에는 최소 30건 이상이 필요합니다.", "crit")
        return None
    if y_all.nunique() <= 10:
        h_note(f"'{target_col}' 의 값 종류가 {y_all.nunique()}가지뿐입니다. "
               f"이런 경우엔 <b>상태 분류(XGBoost)</b> 가 더 적합할 수 있습니다.", "warn")

    X_all, num_cols, cat_cols = _build_design_matrix(target_col, exclude=[date_col])
    mask = y_all.notna()
    X_all, y_all = X_all[mask], y_all[mask]

    y_fit = np.log1p(y_all) if (log_target and (y_all >= 0).all()) else y_all
    used_log = log_target and (y_all >= 0).all()
    if log_target and not used_log:
        h_note("음수 값이 있어 로그 변환을 적용하지 않았습니다.", "warn")

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_fit, test_size=test_size, random_state=42)
    model = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth,
                                  random_state=42, n_jobs=-1, min_samples_leaf=1)
    print("  · 모델을 학습하는 중입니다…")
    model.fit(X_tr, y_tr)

    pred_te, pred_tr = model.predict(X_te), model.predict(X_tr)
    if used_log:
        pred_te, pred_tr = np.expm1(pred_te), np.expm1(pred_tr)
        y_te_o, y_tr_o = np.expm1(y_te), np.expm1(y_tr)
    else:
        y_te_o, y_tr_o = y_te, y_tr

    r2 = r2_score(y_te_o, pred_te)
    r2_tr = r2_score(y_tr_o, pred_tr)
    mae = mean_absolute_error(y_te_o, pred_te)
    rmse = float(np.sqrt(mean_squared_error(y_te_o, pred_te)))
    nz = np.abs(y_te_o) > 1e-9
    mape = float(np.mean(np.abs((np.asarray(y_te_o)[nz] - pred_te[nz]) / np.asarray(y_te_o)[nz])) * 100) \
        if nz.sum() else float("nan")
    grade = ("매우 우수" if r2 >= 0.9 else "우수" if r2 >= 0.75 else
             "쓸 만함" if r2 >= 0.5 else "약함" if r2 >= 0.25 else "예측력 거의 없음")

    h_metrics([
        ("설명력 (R²)", f"{r2*100:.1f}%", f"판정: {grade}"),
        ("평균 오차 (MAE)", fmt(mae), f"{target_col} 단위"),
        ("평균 오차율 (MAPE)", f"{mape:.1f}%" if pd.notna(mape) else "—", "실제 대비 몇 % 틀리는가"),
        ("학습/검증", f"{len(X_tr):,} / {len(X_te):,}건", f"입력 항목 {X_all.shape[1]}개"),
    ])

    # ── 실제 vs 예측 ─────────────────────────────────────────────────────
    h_sub("① 실제값과 예측값 비교")
    lo = float(min(np.min(y_te_o), np.min(pred_te)))
    hi = float(max(np.max(y_te_o), np.max(pred_te)))
    pad = (hi - lo) * 0.05 or 1
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=y_te_o, y=pred_te, mode="markers", name="검증 데이터",
        marker=dict(size=8, color=PALETTE[0], opacity=0.55, line=dict(width=1, color=SURFACE)),
        hovertemplate="실제 %{x:,.4g}<br>예측 %{y:,.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
                             name="완벽한 예측선", line=dict(color="#b3241a", width=1.6, dash="dash"),
                             hoverinfo="skip"))
    fig.update_layout(title="점이 붉은 선에 가까울수록 정확한 예측입니다",
                      xaxis_title=f"실제 {target_col}", yaxis_title=f"예측 {target_col}", height=480)
    show(fig)

    # ── 오차 살펴보기 ────────────────────────────────────────────────────
    h_sub("② 오차 살펴보기")
    resid = np.asarray(y_te_o) - pred_te
    fig = make_subplots(rows=1, cols=2, subplot_titles=("예측값별 오차 (0 주변에 고르게 퍼져야 좋음)",
                                                        "오차 분포"), horizontal_spacing=0.11)
    fig.add_trace(go.Scattergl(x=pred_te, y=resid, mode="markers", showlegend=False,
                               marker=dict(size=7, color=PALETTE[0], opacity=0.5,
                                           line=dict(width=1, color=SURFACE)),
                               hovertemplate="예측 %{x:,.4g}<br>오차 %{y:,.4g}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#b3241a", width=1.3), row=1, col=1)
    fig.add_trace(go.Histogram(x=resid, nbinsx=40, showlegend=False,
                               marker=dict(color=PALETTE[0], line=dict(width=0)),
                               hovertemplate="오차 %{x:,.4g}<br>%{y}건<extra></extra>"), row=1, col=2)
    fig.update_layout(height=400, bargap=0.03)
    fig.update_xaxes(title_text=f"예측 {target_col}", row=1, col=1)
    fig.update_yaxes(title_text="오차 (실제 − 예측)", row=1, col=1)
    fig.update_xaxes(title_text="오차", row=1, col=2)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    show(fig)

    # ── 중요도 ───────────────────────────────────────────────────────────
    h_sub("③ 어떤 항목이 결과를 좌우하는가")
    fig, imp = _importance_fig(X_all.columns, model.feature_importances_,
                               f"'{target_col}' 예측에 중요한 항목 (모델 기준)")
    show(fig)

    perm_imp = None
    if do_perm:
        try:
            from sklearn.inspection import permutation_importance
            print("  · 항목 중요도를 정밀 계산하는 중입니다…")
            pi = permutation_importance(model, X_te, y_te, n_repeats=8, random_state=42, n_jobs=-1)
            fig, perm_imp = _importance_fig(
                X_all.columns, pi.importances_mean,
                "값을 뒤섞었을 때 성능이 얼마나 떨어지는가 (순열 중요도)",
                subtitle="클수록 그 항목 없이는 예측이 어려움")
            show(fig)
            h_note("아래 그래프가 <b>더 믿을 만한 중요도</b>입니다. 특정 항목의 값을 무작위로 섞었을 때 "
                   "예측이 얼마나 나빠지는지를 직접 측정한 결과이기 때문입니다.")
        except Exception as exc:
            print(f"  · 순열 중요도 생략: {exc}")

    # ── 교차검증 ─────────────────────────────────────────────────────────
    cv_mean = None
    if cv_folds and cv_folds >= 2 and len(X_all) >= cv_folds * 10:
        h_sub("④ 반복 검증 (교차검증)")
        try:
            scores = cross_val_score(model, X_all, y_fit,
                                     cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                     scoring="r2", n_jobs=-1)
            cv_mean, cv_std = float(scores.mean()), float(scores.std())
            fig = go.Figure(go.Bar(
                x=[f"{i+1}회차" for i in range(len(scores))], y=scores * 100,
                marker=dict(color=PALETTE[0], line=dict(width=0)),
                text=[f"{s*100:.1f}%" for s in scores], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}<br>설명력 %{y:.1f}%<extra></extra>"))
            fig.add_hline(y=cv_mean * 100, line=dict(color=MUTED, width=1.2, dash="dot"))
            fig.update_layout(title=f"데이터를 {cv_folds}번 다르게 나눠 검증한 결과 "
                                    f"(평균 {cv_mean*100:.1f}%, 편차 ±{cv_std*100:.1f}%p)",
                              yaxis_title="설명력 R² (%)", height=350, showlegend=False)
            show(fig)
        except Exception as exc:
            print(f"  · 교차검증 생략: {exc}")

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    tips.append(f"이 모델은 <b>{target_col}</b> 변동의 <b>약 {max(r2,0)*100:.0f}%</b> 를 설명합니다. "
                f"(판정: <b>{grade}</b>) 나머지는 데이터에 없는 다른 요인이나 우연입니다.")
    if pd.notna(mape):
        tips.append(f"실제값과 비교하면 평균적으로 <b>{mape:.1f}%</b> 정도 차이가 납니다. "
                    f"예측값 하나를 볼 때 <b>±{fmt(mae)}</b> 정도의 오차를 감안하세요.")
    best_imp = (perm_imp if perm_imp is not None else imp)
    if best_imp is not None and len(best_imp):
        top3 = list(best_imp.iloc[::-1].index[:3])
        tips.append(f"결과에 가장 큰 영향을 주는 항목은 <b>{_html.escape(', '.join(map(str, top3)))}</b> 입니다. "
                    f"개선 활동은 이 항목부터 손대는 것이 효율적입니다.")
    if r2_tr - r2 > 0.25:
        tips.append(f"⚠️ 학습 데이터 설명력({r2_tr*100:.0f}%)이 검증 데이터({r2*100:.0f}%)보다 훨씬 높습니다. "
                    f"모델이 <b>답을 외운(과적합)</b> 상태일 수 있으니, 포털에서 <b>나무 최대 깊이</b>를 "
                    f"5~10으로 제한하거나 데이터를 더 모아 보세요.")
    if cv_mean is not None and abs(cv_mean - r2) > 0.15:
        tips.append(f"데이터를 나누는 방식에 따라 성능 편차가 큽니다(교차검증 평균 {cv_mean*100:.0f}%). "
                    f"데이터가 더 필요할 수 있습니다.")
    if r2 < 0.3:
        tips.append("현재 입력 항목만으로는 예측이 어렵습니다. 결과에 영향을 줄 만한 <b>다른 항목을 추가</b>하거나, "
                    "먼저 <b>상관관계 분석</b>으로 관련 있는 항목을 찾아보시길 권합니다.")
    tips.append("❗ 중요도가 높다고 해서 <b>원인</b>이라는 뜻은 아닙니다. 실제 개선 전에는 소규모 시험으로 확인하세요.")
    h_insight(tips)

    out = pd.DataFrame({f"실제_{target_col}": np.asarray(y_te_o),
                        f"예측_{target_col}": pred_te,
                        "오차": resid})
    RESULT_FILES.append(("수치예측_검증결과.csv", out))
    RESULT_FILES.append(("수치예측_항목중요도.csv",
                         (best_imp if best_imp is not None else imp).iloc[::-1]
                         .rename("중요도").reset_index().rename(columns={"index": "항목"})))
    return model


def run_classify_xgb():
    """[상태 분류] 정상/불량 상태 패턴 분류 (XGBoost)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 confusion_matrix, classification_report, roc_auc_score,
                                 roc_curve, precision_recall_curve, average_precision_score)

    h_title("🏷️ 정상 / 불량 상태 분류 (XGBoost)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    if not HAS_XGB:
        h_note("XGBoost 라이브러리를 불러오지 못했습니다. <b>[런타임] → [세션 다시 시작 및 모두 실행]</b> 을 "
               "눌러 다시 시도해 주세요.", "crit")
        return None
    from xgboost import XGBClassifier

    date_col = resolve_col(DATE_COL, kind="date") if DATE_COL else None
    target_col = resolve_col(TARGET_COL, kind="category", required=True, purpose="분류할 상태 컬럼")

    test_size = min(max(P_int("test_size", 20) / 100.0, 0.05), 0.5)
    n_est = max(50, P_int("n_estimators", 400))
    lr = min(max(P_float("learning_rate", 0.08), 0.005), 0.5)
    max_depth = max(2, P_int("max_depth", 5))
    balance = P_bool("balance", True)
    thr_opt = P_bool("threshold_opt", True)

    y_raw = df[target_col]
    valid = y_raw.notna()
    n_class = int(y_raw[valid].nunique())
    if n_class < 2:
        h_note(f"'{target_col}' 의 값이 한 종류뿐이라 분류할 수 없습니다.", "crit")
        return None
    if n_class > 20:
        h_note(f"'{target_col}' 의 값 종류가 {n_class}가지로 너무 많습니다. "
               f"숫자를 맞히는 문제라면 <b>수치 예측(Random Forest)</b> 을 선택하세요.", "crit")
        return None

    X_all, num_cols, cat_cols = _build_design_matrix(target_col, exclude=[date_col])
    X_all = X_all[valid.values]
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw[valid].astype(str))
    labels = [str(c) for c in le.classes_]
    binary = n_class == 2

    counts = pd.Series(y_enc).value_counts().sort_index()
    minority = int(counts.idxmin())
    imbalance = counts.min() / counts.sum()

    if counts.min() < 5:
        h_note(f"가장 적은 상태('{labels[minority]}')가 {int(counts.min())}건뿐입니다. "
               f"신뢰할 만한 학습이 어렵습니다.", "warn")

    strat = y_enc if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_enc, test_size=test_size,
                                              random_state=42, stratify=strat)

    kwargs = dict(n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                  subsample=0.9, colsample_bytree=0.9, random_state=42,
                  n_jobs=-1, tree_method="hist", eval_metric="logloss")
    sample_weight = None
    if balance:
        if binary:
            pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
            if pos > 0:
                kwargs["scale_pos_weight"] = neg / pos
        else:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight("balanced", y_tr)

    model = XGBClassifier(**kwargs)
    print("  · 모델을 학습하는 중입니다…")
    model.fit(X_tr, y_tr, sample_weight=sample_weight)

    proba = model.predict_proba(X_te)
    threshold = 0.5
    if binary and thr_opt:
        p1 = proba[:, 1]
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            f1t = f1_score(y_te, (p1 >= t).astype(int), zero_division=0)
            if f1t > best_f1:
                best_f1, best_t = f1t, float(t)
        threshold = best_t
    y_pred = (proba[:, 1] >= threshold).astype(int) if binary else proba.argmax(axis=1)

    acc = accuracy_score(y_te, y_pred)
    avg = "binary" if binary else "macro"
    prec = precision_score(y_te, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_te, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_te, y_pred, average=avg, zero_division=0)
    try:
        auc = roc_auc_score(y_te, proba[:, 1]) if binary else \
            roc_auc_score(y_te, proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    pos_label = labels[1] if binary else "각 상태"
    h_metrics([
        ("정확도", f"{acc*100:.1f}%", "전체 중 맞힌 비율"),
        (f"검출률(재현율)", f"{rec*100:.1f}%", f"실제 '{pos_label}' 중 잡아낸 비율"),
        ("정밀도", f"{prec*100:.1f}%", f"'{pos_label}' 이라 한 것 중 진짜 비율"),
        ("종합 점수 (AUC)", f"{auc:.3f}" if pd.notna(auc) else "—",
         "1에 가까울수록 좋음 (0.5 = 찍기 수준)"),
    ])
    if binary and thr_opt and abs(threshold - 0.5) > 0.01:
        h_note(f"판정 기준을 자동으로 <b>{threshold:.2f}</b> 로 조정했습니다 "
               f"(기본 0.50). 확률이 {threshold:.0%} 이상이면 '{labels[1]}' 으로 판정합니다.")

    # ── 혼동행렬 ─────────────────────────────────────────────────────────
    h_sub("① 무엇을 맞히고 무엇을 틀렸나")
    cm = confusion_matrix(y_te, y_pred)
    total = cm.sum()
    text = [[f"{cm[i][j]:,}건<br>({cm[i][j]/total*100:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"{l} 로 예측" for l in labels], y=[f"실제 {l}" for l in labels],
        colorscale=SEQ_BLUE, text=text, texttemplate="%{text}", textfont=dict(size=13),
        colorbar=dict(title="건수", thickness=13),
        hovertemplate="%{y} → %{x}<br>%{z:,}건<extra></extra>", xgap=2, ygap=2))
    fig.update_layout(title="대각선(왼쪽 위 → 오른쪽 아래)이 맞힌 것입니다",
                      height=max(340, 90 * len(labels) + 200),
                      xaxis=dict(showgrid=False), yaxis=dict(autorange="reversed", showgrid=False))
    show(fig)

    if binary:
        tn, fp, fn, tp = cm.ravel()
        h_note(f"실제 <b>{labels[1]}</b> {tp+fn:,}건 중 <b>{tp:,}건을 찾아냈고 {fn:,}건을 놓쳤습니다.</b> "
               f"또 실제 <b>{labels[0]}</b> 인데 {labels[1]} 이라고 잘못 알린 경우가 <b>{fp:,}건</b> 있습니다.",
               "warn" if fn > tp else "good")

    rep = classification_report(y_te, y_pred, target_names=labels, output_dict=True, zero_division=0)
    rep_tbl = pd.DataFrame([{
        "상태": k,
        "정밀도": round(v["precision"], 3),
        "검출률": round(v["recall"], 3),
        "F1": round(v["f1-score"], 3),
        "실제 건수": int(v["support"]),
    } for k, v in rep.items() if k in labels])
    h_table(rep_tbl, "상태별 성적표", highlight="상태")

    # ── ROC / PR ─────────────────────────────────────────────────────────
    if binary:
        h_sub("② 판정 성능 곡선")
        fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
        p, r, _ = precision_recall_curve(y_te, proba[:, 1])
        ap = average_precision_score(y_te, proba[:, 1])
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                            subplot_titles=(f"ROC 곡선 (AUC {auc:.3f})",
                                            f"정밀도-검출률 곡선 (AP {ap:.3f})"))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="모델 성능",
                                 line=dict(color=PALETTE[0], width=2.4),
                                 hovertemplate="오탐률 %{x:.2f}<br>검출률 %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="무작위(찍기)",
                                 line=dict(color=MUTED, width=1.3, dash="dash"),
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name="정밀도-검출률",
                                 line=dict(color=PALETTE[1], width=2.4), showlegend=True,
                                 hovertemplate="검출률 %{x:.2f}<br>정밀도 %{y:.2f}<extra></extra>"), row=1, col=2)
        fig.update_xaxes(title_text="잘못 알린 비율 (오탐률)", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="잡아낸 비율 (검출률)", row=1, col=1, range=[0, 1.02])
        fig.update_xaxes(title_text="검출률", row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text="정밀도", row=1, col=2, range=[0, 1.02])
        fig.update_layout(height=420)
        fig.update_annotations(font=dict(size=12.5, color=INK_2))
        show(fig)
        h_note("곡선이 <b>왼쪽 위로 붙을수록</b> 좋은 모델입니다. 회색 점선(찍기)보다 위에 있어야 의미가 있습니다.")

    # ── 중요도 ───────────────────────────────────────────────────────────
    h_sub("③ 판정에 중요한 항목")
    fig, imp = _importance_fig(X_all.columns, model.feature_importances_,
                               f"'{target_col}' 판정에 중요한 항목")
    show(fig)

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    tips.append(f"검증 데이터 <b>{len(y_te):,}건</b> 중 <b>{acc*100:.1f}%</b> 를 올바르게 판정했습니다.")
    if binary:
        tn, fp, fn, tp = cm.ravel()
        tips.append(f"실무 언어로 옮기면 — 실제 <b>{labels[1]}</b> 100건이 있을 때 약 <b>{rec*100:.0f}건</b>을 "
                    f"잡아내고 <b>{(1-rec)*100:.0f}건</b>을 놓칩니다. 또 <b>{labels[1]}</b> 이라고 알린 것 중 "
                    f"<b>{prec*100:.0f}%</b> 만 진짜입니다.")
        if fn > tp * 0.5:
            tips.append("놓치는 건수가 많습니다. 놓치는 비용이 크다면 포털에서 <b>불균형 데이터 보정</b>을 켜고, "
                        "판정 기준값을 낮춰 더 적극적으로 잡아내도록 조정하세요.")
        if fp > tp * 2:
            tips.append("헛알람이 많습니다. 현장 피로도를 줄이려면 판정 기준값을 높이는 편이 좋습니다.")
    if imbalance < 0.1:
        tips.append(f"⚠️ 가장 적은 상태가 전체의 <b>{imbalance*100:.1f}%</b> 뿐인 불균형 데이터입니다. "
                    f"이런 경우 <b>정확도만 보면 안 됩니다</b> — 위의 <b>검출률</b>을 기준으로 판단하세요.")
    if len(imp):
        top3 = list(imp.iloc[::-1].index[:3])
        tips.append(f"판정에 가장 크게 쓰인 항목은 <b>{_html.escape(', '.join(map(str, top3)))}</b> 입니다. "
                    f"현장 점검 항목의 우선순위로 삼을 수 있습니다.")
    if pd.notna(auc):
        judge = ("실무 적용이 가능한 수준입니다." if auc >= 0.85 else
                 "보조 지표로 쓸 만한 수준입니다." if auc >= 0.7 else
                 "아직 신뢰하기 어렵습니다. 더 관련 있는 항목을 추가해 보세요.")
        tips.append(f"종합 판별력(AUC)은 <b>{auc:.3f}</b> 입니다. {judge}")
    tips.append("❗ 검증 성적은 <b>과거 데이터 기준</b>입니다. 공정이나 설비가 바뀌면 성능이 달라질 수 있으니 "
                "주기적으로 다시 학습시켜 주세요.")
    h_insight(tips)

    out = pd.DataFrame({f"실제_{target_col}": le.inverse_transform(y_te),
                        f"예측_{target_col}": le.inverse_transform(y_pred)})
    if binary:
        out[f"'{labels[1]}' 확률"] = np.round(proba[:, 1], 4)
    RESULT_FILES.append(("상태분류_검증결과.csv", out))
    RESULT_FILES.append(("상태분류_항목중요도.csv",
                         imp.iloc[::-1].rename("중요도").reset_index().rename(columns={"index": "항목"})))
    return model


MAX_CLUSTERS = 8   # 사람이 눈으로 구분할 수 있고 색으로도 안전하게 표현 가능한 상한


def run_cluster_kmeans():
    """[군집화] 유사 특성 데이터 자동 그룹핑 (K-Means)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    h_title("🧩 유사 특성 데이터 자동 그룹핑 (K-Means)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    date_col = resolve_col(DATE_COL, kind="date") if DATE_COL else None
    k_req = P_int("k", 0)
    k_max = min(max(P_int("k_max", 10), 3), MAX_CLUSTERS)
    do_scale = P_bool("scale", True)
    do_pca = P_bool("pca_view", True)
    do_profile = P_bool("profile", True)

    feats = pick_features(exclude=[c for c in [date_col] if c])
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    if len(feats) < 2:
        h_note("군집화에는 숫자 컬럼이 최소 2개 필요합니다. "
               "포털에서 '분석에 사용할 컬럼들'을 지정해 주세요.", "crit")
        return None

    X_raw = df[feats].apply(pd.to_numeric, errors="coerce")
    X_raw, filled = _impute_median(X_raw)
    if filled:
        h_note("빈칸은 각 컬럼의 중앙값으로 채웠습니다: "
               + ", ".join(f"{_html.escape(str(k))} {v:,}건" for k, v in list(filled.items())[:6]), "warn")
    X = StandardScaler().fit_transform(X_raw.values) if do_scale else X_raw.values

    if k_req > MAX_CLUSTERS:
        h_note(f"그룹 수를 {k_req}개로 지정하셨지만, 사람이 해석하기 좋은 상한인 "
               f"<b>{MAX_CLUSTERS}개</b>로 조정했습니다.", "warn")
        k_req = MAX_CLUSTERS

    # ── 최적 그룹 수 찾기 ────────────────────────────────────────────────
    sil_scores, inertias, ks = {}, {}, list(range(2, min(k_max, max(2, len(X) - 1)) + 1))
    if (k_req < 2 or do_profile) and len(X) > 10 and ks:
        h_sub("① 몇 개 그룹으로 나누는 것이 좋은가")
        sample = X if len(X) <= 5000 else X[np.random.default_rng(42).choice(len(X), 5000, replace=False)]
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            inertias[k] = float(km.inertia_)
            try:
                lab = km.predict(sample)
                sil_scores[k] = float(silhouette_score(sample, lab)) if len(set(lab)) > 1 else 0.0
            except Exception:
                sil_scores[k] = 0.0

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                            subplot_titles=("그룹 응집도 (실루엣) — 높을수록 좋음",
                                            "그룹 내 흩어짐 (엘보우) — 꺾이는 지점이 적당"))
        fig.add_trace(go.Scatter(x=list(sil_scores), y=list(sil_scores.values()), mode="lines+markers",
                                 line=dict(color=PALETTE[0], width=2.2), marker=dict(size=9),
                                 showlegend=False,
                                 hovertemplate="%{x}개 그룹<br>실루엣 %{y:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(inertias), y=list(inertias.values()), mode="lines+markers",
                                 line=dict(color=PALETTE[1], width=2.2), marker=dict(size=9),
                                 showlegend=False,
                                 hovertemplate="%{x}개 그룹<br>흩어짐 %{y:,.4g}<extra></extra>"), row=1, col=2)
        fig.update_xaxes(title_text="그룹 수", row=1, col=1, dtick=1)
        fig.update_xaxes(title_text="그룹 수", row=1, col=2, dtick=1)
        fig.update_yaxes(title_text="실루엣 점수", row=1, col=1)
        fig.update_yaxes(title_text="그룹 내 거리 합", row=1, col=2)
        fig.update_layout(height=380)
        fig.update_annotations(font=dict(size=12.5, color=INK_2))
        show(fig)

    if k_req >= 2:
        k = k_req
        pick_reason = "직접 지정하신 값"
    elif sil_scores:
        k = int(max(sil_scores, key=sil_scores.get))
        pick_reason = f"실루엣 점수가 가장 높은 값 ({sil_scores[k]:.3f})"
    else:
        k = 3
        pick_reason = "기본값"

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    res = df.copy()
    res["그룹"] = [f"그룹 {i+1}" for i in labels]
    sizes = pd.Series(labels).value_counts().sort_index()

    try:
        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")
    except Exception:
        sil = float("nan")
    sil_grade = ("뚜렷하게 나뉨" if sil >= 0.5 else "어느 정도 나뉨" if sil >= 0.25 else
                 "경계가 흐릿함" if pd.notna(sil) else "—")

    h_metrics([
        ("그룹 수", f"{k}개", pick_reason),
        ("응집도 (실루엣)", f"{sil:.3f}" if pd.notna(sil) else "—", sil_grade),
        ("가장 큰 그룹", f"{int(sizes.max()):,}건", f"그룹 {int(sizes.idxmax())+1}"),
        ("가장 작은 그룹", f"{int(sizes.min()):,}건", f"그룹 {int(sizes.idxmin())+1}"),
    ])

    # ── 그룹 크기 ────────────────────────────────────────────────────────
    h_sub("② 그룹별 데이터 수")
    fig = go.Figure(go.Bar(
        x=[f"그룹 {i+1}" for i in sizes.index], y=sizes.values,
        marker=dict(color=[PALETTE[i % len(PALETTE)] for i in sizes.index], line=dict(width=0)),
        text=[f"{v:,}건<br>({v/sizes.sum()*100:.1f}%)" for v in sizes.values],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:,}건<extra></extra>"))
    fig.update_layout(title="각 그룹에 몇 건씩 묶였는지", yaxis_title="건수", height=360, showlegend=False)
    show(fig)

    # ── 2차원 지도 ───────────────────────────────────────────────────────
    if do_pca and len(feats) >= 2:
        h_sub("③ 그룹 지도 (2차원 압축)")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            xy = pca.fit_transform(StandardScaler().fit_transform(X_raw.values))
            var = pca.explained_variance_ratio_
            fig = go.Figure()
            for i in range(k):
                m = labels == i
                fig.add_trace(go.Scattergl(
                    x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"그룹 {i+1}",
                    marker=dict(size=7, color=PALETTE[i % len(PALETTE)], opacity=0.6,
                                line=dict(width=1, color=SURFACE)),
                    hovertemplate=f"그룹 {i+1}<extra></extra>"))
                # 각 그룹의 중심에 이름표를 직접 붙입니다 (색만으로 구분하지 않도록)
                fig.add_annotation(x=float(xy[m, 0].mean()), y=float(xy[m, 1].mean()),
                                   text=f"<b>그룹 {i+1}</b>", showarrow=False,
                                   font=dict(size=13, color=INK),
                                   bgcolor="rgba(252,252,251,0.82)", borderpad=3)
            fig.update_layout(
                title=f"비슷한 데이터끼리 모인 지도 (전체 정보의 {var.sum()*100:.0f}% 를 2차원으로 표현)",
                xaxis_title=f"주성분 1 ({var[0]*100:.0f}%)",
                yaxis_title=f"주성분 2 ({var[1]*100:.0f}%)", height=520)
            show(fig)
            h_note("여러 항목을 2차원으로 눌러 담은 그림입니다. <b>가까이 있을수록 성격이 비슷</b>합니다. "
                   "축 자체에는 특별한 의미가 없습니다.")
        except Exception as exc:
            print(f"  · 2차원 지도 생략: {exc}")

    # ── 그룹 성격 ────────────────────────────────────────────────────────
    profile_desc = {}
    if do_profile:
        h_sub("④ 각 그룹은 어떤 성격인가")
        means = X_raw.groupby(labels).mean()
        overall_mu, overall_sd = X_raw.mean(), X_raw.std(ddof=0).replace(0, np.nan)
        z = ((means - overall_mu) / overall_sd).fillna(0)
        zmax = float(np.abs(z.values).max()) or 1.0

        fig = go.Figure(go.Heatmap(
            z=z.values, x=[str(c) for c in z.columns], y=[f"그룹 {i+1}" for i in z.index],
            colorscale=DIVERGING, zmid=0, zmin=-zmax, zmax=zmax,
            text=np.round(z.values, 1), texttemplate="%{text}", textfont=dict(size=10.5),
            colorbar=dict(title=dict(text="전체 평균 대비", side="right"), thickness=13,
                          tickvals=[-zmax, 0, zmax], ticktext=["낮음", "평균", "높음"]),
            hovertemplate="%{y}<br>%{x}<br>평균보다 %{z:+.2f} 표준편차<extra></extra>",
            xgap=1.5, ygap=1.5))
        fig.update_layout(title="붉을수록 전체 평균보다 높고, 푸를수록 낮습니다",
                          height=max(300, 52 * len(z) + 210),
                          xaxis=dict(tickangle=-35, showgrid=False),
                          yaxis=dict(autorange="reversed", showgrid=False))
        show(fig)

        prof = means.copy()
        prof.insert(0, "건수", sizes.values)
        prof.index = [f"그룹 {i+1}" for i in prof.index]
        h_table(prof.reset_index().rename(columns={"index": "그룹"}),
                "그룹별 평균값", max_rows=MAX_CLUSTERS, highlight="그룹")

        for i in z.index:
            row = z.loc[i].sort_values()
            highs = [c for c in row.index[::-1][:2] if row[c] > 0.35]
            lows = [c for c in row.index[:2] if row[c] < -0.35]
            bits = []
            if highs:
                bits.append(f"<b>{_html.escape(', '.join(map(str, highs)))}</b> 이(가) 높고")
            if lows:
                bits.append(f"<b>{_html.escape(', '.join(map(str, lows)))}</b> 이(가) 낮은")
            desc = (" ".join(bits) + " 유형") if bits else "전반적으로 평균에 가까운 유형"
            profile_desc[i] = desc

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    tips.append(f"데이터를 <b>{k}개 그룹</b>으로 나눴습니다. ({pick_reason})")
    for i in range(k):
        n = int((labels == i).sum())
        d = profile_desc.get(i, "")
        tips.append(f"<b>그룹 {i+1}</b> — {n:,}건 ({n/len(labels)*100:.1f}%)" + (f" · {d}" if d else ""))
    if pd.notna(sil):
        tips.append(f"그룹 구분의 선명도(실루엣)는 <b>{sil:.3f}</b> 로 <b>{sil_grade}</b> 상태입니다. "
                    + ("그룹별로 다른 전략을 적용할 만합니다." if sil >= 0.25 else
                       "그룹 간 경계가 뚜렷하지 않으니, 사용 항목을 바꾸거나 그룹 수를 조정해 보세요."))
    small = [i for i in range(k) if (labels == i).sum() < max(5, len(labels) * 0.02)]
    if small:
        tips.append(f"⚠️ 그룹 {', '.join(str(i+1) for i in small)} 은(는) 데이터가 매우 적습니다. "
                    f"특이 사례이거나 이상치일 수 있으니 <b>이상 탐지</b> 로 따로 확인해 보세요.")
    tips.append("결과 CSV에는 각 행이 어느 그룹인지 <b>'그룹'</b> 컬럼으로 붙어 있습니다. "
                "엑셀에서 그룹별로 필터를 걸어 살펴보세요.")
    h_insight(tips)

    RESULT_FILES.append(("군집화_결과.csv", res))
    return res


def _p_words(p, alpha=0.05):
    """p-값을 비전문가가 이해할 수 있는 문장으로 바꿉니다."""
    if pd.isna(p):
        return "판단 불가", "—"
    if p < 0.001:
        return "차이가 거의 확실합니다", "우연일 확률 0.1% 미만"
    if p < 0.01:
        return "차이가 있다고 볼 수 있습니다", f"우연일 확률 {p*100:.1f}%"
    if p < alpha:
        return "차이가 있다고 볼 수 있습니다", f"우연일 확률 {p*100:.1f}%"
    if p < 0.1:
        return "차이가 있다고 보기 어렵습니다", f"우연일 확률 {p*100:.0f}% — 애매한 경계"
    return "차이가 있다고 볼 수 없습니다", f"우연일 확률 {p*100:.0f}%"


def _effect_words(d):
    a = abs(d)
    if a >= 0.8:
        return "큼 — 현장에서 체감될 정도"
    if a >= 0.5:
        return "중간 — 의미 있는 수준"
    if a >= 0.2:
        return "작음 — 통계적으로만 보이는 정도"
    return "매우 작음 — 실질적 의미는 거의 없음"


def run_compare_groups():
    """[비교·검정] 그룹 간 차이가 진짜인지 통계로 확인."""
    from scipy import stats

    h_title("⚖️ 그룹 간 차이 비교 (통계 검정)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    group_col = resolve_col(GROUP_COL, kind="category", required=True, purpose="비교 기준(그룹) 컬럼")
    value_col = resolve_col(TARGET_COL, kind="number", required=True, purpose="비교할 숫자 컬럼")
    alpha = min(max(P_float("alpha", 0.05), 0.001), 0.2)
    method = P_str("test_method", "auto")
    do_pairwise = P_bool("pairwise", True)
    min_n = max(2, P_int("min_group_n", 3))

    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    work[group_col] = work[group_col].astype(str)

    counts = work[group_col].value_counts()
    keep = counts[counts >= min_n]
    if len(keep) > 10:
        h_note(f"그룹이 {len(keep)}개나 되어, 데이터가 많은 상위 10개만 비교합니다.", "warn")
        keep = keep.head(10)
    dropped = [g for g in counts.index if g not in keep.index]
    if dropped:
        h_note(f"데이터가 {min_n}건 미만이라 제외한 그룹: "
               f"<b>{_html.escape(', '.join(map(str, dropped[:8])))}</b>", "warn")
    work = work[work[group_col].isin(keep.index)]
    names = list(keep.index)
    samples = [work.loc[work[group_col] == g, value_col].values for g in names]

    if len(names) < 2:
        h_note("비교하려면 그룹이 최소 2개 필요합니다. 다른 컬럼을 그룹 기준으로 선택해 주세요.", "crit")
        return None

    # ── 어떤 검정을 쓸지 자동 판단 ────────────────────────────────────────
    normal_ok, var_ok = True, True
    for s in samples:
        if len(s) < 3:
            normal_ok = False
            continue
        try:
            pv = (stats.shapiro(s[:5000])[1] if len(s) <= 5000 else stats.normaltest(s)[1])
            if pv < 0.05:
                normal_ok = False
        except Exception:
            normal_ok = False
    try:
        var_ok = stats.levene(*samples)[1] >= 0.05
    except Exception:
        var_ok = False

    if method == "auto":
        parametric = normal_ok
    else:
        parametric = (method == "parametric")

    stat_desc = {
        (True, 2): "웰치 t-검정 (평균 비교)",
        (False, 2): "맨-휘트니 U 검정 (순위 비교)",
        (True, 3): "일원배치 분산분석 ANOVA (평균 비교)",
        (False, 3): "크루스칼-왈리스 검정 (순위 비교)",
    }[(parametric, 2 if len(names) == 2 else 3)]

    if len(names) == 2:
        if parametric:
            stat, p = stats.ttest_ind(samples[0], samples[1], equal_var=False)
        else:
            stat, p = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
    else:
        if parametric:
            stat, p = stats.f_oneway(*samples)
        else:
            stat, p = stats.kruskal(*samples)

    # ── 효과 크기 (차이가 '얼마나' 큰가) ─────────────────────────────────
    if len(names) == 2:
        a, b = samples[0], samples[1]
        pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                         / max(len(a) + len(b) - 2, 1))
        effect = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
        effect_name = "코헨의 d"
    else:
        grand = np.concatenate(samples)
        ss_between = sum(len(s) * (np.mean(s) - grand.mean()) ** 2 for s in samples)
        ss_total = ((grand - grand.mean()) ** 2).sum()
        eta = ss_between / ss_total if ss_total > 0 else 0.0
        effect = np.sqrt(eta / (1 - eta)) if eta < 1 else 0.0   # 코헨 f 로 환산해 해석 통일
        effect_name = "에타제곱(η²)"

    verdict, chance = _p_words(p, alpha)
    sig = p < alpha

    h_metrics([
        ("비교 그룹", f"{len(names)}개", " vs ".join(map(str, names[:3])) + (" …" if len(names) > 3 else "")),
        ("사용한 검정", stat_desc.split("(")[0].strip(), stat_desc.split("(")[1].rstrip(")") if "(" in stat_desc else ""),
        ("판정", "차이 있음" if sig else "차이 없음", chance),
        ("차이의 크기", f"{eta*100:.1f}%" if len(names) > 2 else f"{abs(effect):.2f}",
         _effect_words(effect) if len(names) == 2 else f"{effect_name} — 그룹으로 설명되는 비율"),
    ])
    h_note(f"<b>{verdict}.</b> ({stat_desc}, p = {p:.4g}, 기준 {alpha})<br>"
           f"‘우연일 확률’이 {alpha*100:.0f}%보다 작으면 “차이가 진짜”라고 판단합니다.",
           "good" if sig else "warn")

    # ── 그룹별 분포 ──────────────────────────────────────────────────────
    h_sub("① 그룹별 분포 한눈에 보기")
    fig = go.Figure()
    for i, (g, s) in enumerate(zip(names, samples)):
        fig.add_trace(go.Box(
            y=s, name=str(g), boxpoints="outliers", boxmean=True,
            marker=dict(color=PALETTE[i % len(PALETTE)], size=5,
                        outliercolor="#d03b3b", line=dict(width=1, color=SURFACE)),
            line=dict(width=1.6), fillcolor=SURFACE,
            hovertemplate=f"{g}<br>%{{y:,.4g}}<extra></extra>"))
    fig.update_layout(title=f"'{group_col}' 별 '{value_col}' 분포 (상자 = 가운데 50%, 가로선 = 중앙값, 점선 = 평균)",
                      yaxis_title=str(value_col), xaxis_title=str(group_col),
                      height=470, showlegend=False)
    show(fig)

    # ── 평균과 신뢰구간 ──────────────────────────────────────────────────
    h_sub("② 그룹별 평균과 오차 범위")
    stats_tbl = []
    for g, s in zip(names, samples):
        se = np.std(s, ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
        ci = 1.96 * se
        stats_tbl.append({"그룹": g, "건수": len(s), "평균": np.mean(s), "표준편차": np.std(s, ddof=1) if len(s) > 1 else 0.0,
                          "중앙값": np.median(s), "최솟값": np.min(s), "최댓값": np.max(s), "_ci": ci})
    st = pd.DataFrame(stats_tbl)
    fig = go.Figure(go.Scatter(
        x=st["그룹"], y=st["평균"], mode="markers",
        marker=dict(size=13, color=PALETTE[0], line=dict(width=2, color=SURFACE)),
        error_y=dict(type="data", array=st["_ci"], color=PALETTE[0], thickness=1.6, width=8),
        hovertemplate="%{x}<br>평균 %{y:,.4g}<extra></extra>", showlegend=False))
    fig.add_hline(y=float(np.mean(np.concatenate(samples))),
                  line=dict(color=MUTED, width=1.2, dash="dot"))
    fig.add_annotation(x=1.0, xref="paper", y=float(np.mean(np.concatenate(samples))),
                       text="전체 평균", showarrow=False, xanchor="right", yshift=10,
                       font=dict(size=11, color=MUTED))
    fig.update_layout(title="세로 막대(오차 범위)가 서로 겹치지 않으면 차이가 뚜렷하다는 신호입니다",
                      yaxis_title=f"{value_col} 평균 (95% 신뢰구간)", xaxis_title=str(group_col), height=420)
    show(fig)
    h_table(st.drop(columns=["_ci"]), "그룹별 기초 통계", max_rows=12, highlight="그룹")

    # ── 그룹 쌍별 비교 ───────────────────────────────────────────────────
    pair_tbl = None
    if do_pairwise and len(names) > 2:
        h_sub("③ 어느 그룹끼리 차이가 나는가 (쌍별 비교)")
        pairs = []
        n_pairs = len(names) * (len(names) - 1) // 2
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if parametric:
                    _, pp = stats.ttest_ind(samples[i], samples[j], equal_var=False)
                else:
                    _, pp = stats.mannwhitneyu(samples[i], samples[j], alternative="two-sided")
                padj = min(1.0, pp * n_pairs)      # 본페로니 보정 (여러 번 비교하면 우연이 늘어남)
                diff = float(np.mean(samples[i]) - np.mean(samples[j]))
                pairs.append({"그룹 A": names[i], "그룹 B": names[j],
                              "평균 차이": round(diff, 4),
                              "p값(보정)": float(padj),
                              "판정": "차이 있음" if padj < alpha else "차이 없음"})
        pair_tbl = pd.DataFrame(pairs).sort_values("p값(보정)").reset_index(drop=True)
        view = pair_tbl.copy()
        view["p값(보정)"] = view["p값(보정)"].map(lambda x: f"{x:.4g}")
        h_table(view, f"모든 조합을 비교했습니다 (여러 번 비교한 만큼 기준을 엄격하게 보정 · {n_pairs}쌍)",
                max_rows=45, highlight="판정")

        mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
        for r in pairs:
            mat.loc[r["그룹 A"], r["그룹 B"]] = r["p값(보정)"]
            mat.loc[r["그룹 B"], r["그룹 A"]] = r["p값(보정)"]
        cell_text = [["" if pd.isna(v) else f"{v:.3g}" for v in row] for row in mat.values]
        fig = go.Figure(go.Heatmap(
            z=mat.values, x=names, y=names,
            colorscale=[[0.0, "#184f95"], [0.5, "#9ec5f4"], [1.0, "#f0efec"]],
            zmin=0, zmax=1, text=cell_text, texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="p값", thickness=13,
                          tickvals=[0, 0.5, 1], ticktext=["0<br>차이 큼", "0.5", "1<br>차이 없음"]),
            hovertemplate="%{y} ↔ %{x}<br>p = %{z:.4g}<extra></extra>", xgap=2, ygap=2))
        fig.update_layout(title="진할수록 두 그룹의 차이가 뚜렷합니다",
                          height=max(340, 52 * len(names) + 200),
                          xaxis=dict(showgrid=False), yaxis=dict(autorange="reversed", showgrid=False))
        show(fig)

    # ── 자동 해석 ────────────────────────────────────────────────────────
    best = st.loc[st["평균"].idxmax()]
    worst = st.loc[st["평균"].idxmin()]
    gap = float(best["평균"] - worst["평균"])
    gap_pct = gap / abs(worst["평균"]) * 100 if worst["평균"] else np.nan
    tips = [
        f"<b>{_html.escape(str(best['그룹']))}</b> 의 평균이 <b>{fmt(float(best['평균']))}</b> 로 가장 높고, "
        f"<b>{_html.escape(str(worst['그룹']))}</b> 이(가) <b>{fmt(float(worst['평균']))}</b> 로 가장 낮습니다. "
        f"차이는 <b>{fmt(gap)}</b>" + (f" (약 {gap_pct:+.1f}%)" if pd.notna(gap_pct) else "") + " 입니다.",
        f"통계적으로 보면 <b>{verdict}</b>. {chance}이므로, "
        + ("이 차이는 <b>우연으로 보기 어렵습니다.</b>" if sig else
           "지금 데이터만으로는 <b>진짜 차이라고 단정할 수 없습니다.</b> 데이터를 더 모으면 달라질 수 있습니다."),
    ]
    if sig and len(names) == 2:
        tips.append(f"차이의 실질적 크기는 <b>{_effect_words(effect)}</b> 입니다. "
                    f"통계적으로 유의하더라도 크기가 작으면 현장 개선 효과는 제한적일 수 있습니다.")
    if pair_tbl is not None and len(pair_tbl):
        sig_pairs = pair_tbl[pair_tbl["판정"] == "차이 있음"]
        if len(sig_pairs):
            r = sig_pairs.iloc[0]
            tips.append(f"쌍별로 보면 <b>{_html.escape(str(r['그룹 A']))}</b> 와(과) "
                        f"<b>{_html.escape(str(r['그룹 B']))}</b> 의 차이가 가장 뚜렷합니다. "
                        f"({len(sig_pairs)}개 조합에서 차이 확인)")
        else:
            tips.append("전체적으로는 차이가 보여도, 개별 그룹끼리 짝지어 비교하면 "
                        "뚜렷한 차이가 나오는 조합은 없습니다.")
    if not normal_ok:
        tips.append("데이터 분포가 좌우대칭(정규분포)이 아니어서 <b>순위 기반 검정</b>을 사용했습니다. "
                    "이 방식은 이상치의 영향을 덜 받습니다.")
    if not var_ok:
        tips.append("그룹마다 값이 흩어진 정도가 달라, 이를 감안하는 방식으로 계산했습니다.")
    tips.append("❗ ‘차이가 있다’는 것이 <b>원인을 밝힌 것은 아닙니다.</b> 그룹을 나누는 다른 요인이 "
                "함께 달라졌을 수 있으니, 개선 전에는 조건을 맞춘 소규모 시험을 권합니다.")
    h_insight(tips)

    RESULT_FILES.append(("그룹비교_통계.csv", st.drop(columns=["_ci"])))
    if pair_tbl is not None:
        RESULT_FILES.append(("그룹비교_쌍별검정.csv", pair_tbl))
    return st


# 관리도 상수표 (부분군 크기 n = 2~10)
_SPC_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
_SPC_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
_SPC_D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
_SPC_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


def _we_rules(vals, center, sigma):
    """웨스턴일렉트릭 규칙 — 공정이 '평소와 다르게' 움직이는 신호를 찾습니다."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    hits = {1: [], 2: [], 3: [], 4: []}
    if sigma <= 0:
        return hits
    z = (v - center) / sigma
    for i in range(n):
        if abs(z[i]) > 3:
            hits[1].append(i)
    for i in range(2, n):
        w = z[i - 2:i + 1]
        if (w > 2).sum() >= 2 or (w < -2).sum() >= 2:
            hits[2].append(i)
    for i in range(4, n):
        w = z[i - 4:i + 1]
        if (w > 1).sum() >= 4 or (w < -1).sum() >= 4:
            hits[3].append(i)
    for i in range(7, n):
        w = z[i - 7:i + 1]
        if (w > 0).all() or (w < 0).all():
            hits[4].append(i)
    return hits


def run_spc_control_chart():
    """[품질관리] 공정 관리도 (SPC) — 공정이 안정적인지 판정."""
    h_title("📐 공정 관리도 (SPC)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    value_col = resolve_col(TARGET_COL, kind="number", required=True, purpose="관리할 측정값 컬럼")
    date_col = resolve_col(DATE_COL, kind="date") if DATE_COL else None
    subgroup = min(max(P_int("subgroup", 1), 1), 10)
    use_rules = P_bool("we_rules", True)

    def _spec(key):
        raw = P_str(key, "")
        try:
            return float(raw) if raw not in ("", "-", "없음") else None
        except Exception:
            return None
    lsl, usl = _spec("lsl"), _spec("usl")

    work = df[[c for c in [date_col, value_col] if c]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col)
    work = work.dropna(subset=[value_col]).reset_index(drop=True)
    if len(work) < 10:
        h_note(f"관리도를 그리려면 최소 10건이 필요합니다. (현재 {len(work)}건)", "crit")
        return None

    vals = work[value_col].values.astype(float)
    xs = work[date_col] if date_col else pd.Series(np.arange(1, len(vals) + 1))
    x_title = str(date_col) if date_col else "측정 순번"

    # ── 관리선 계산 ──────────────────────────────────────────────────────
    if subgroup >= 2:
        n_grp = len(vals) // subgroup
        if n_grp < 5:
            h_note(f"부분군 크기 {subgroup}로는 그룹이 {n_grp}개뿐입니다. 개별값 관리도(I-MR)로 전환합니다.", "warn")
            subgroup = 1
    if subgroup >= 2:
        n_grp = len(vals) // subgroup
        grouped = vals[:n_grp * subgroup].reshape(n_grp, subgroup)
        plot_vals = grouped.mean(axis=1)
        ranges = grouped.max(axis=1) - grouped.min(axis=1)
        center = float(plot_vals.mean())
        rbar = float(ranges.mean())
        sigma = rbar / _SPC_D2[subgroup]
        ucl = center + _SPC_A2[subgroup] * rbar
        lcl = center - _SPC_A2[subgroup] * rbar
        sec_vals, sec_center = ranges, rbar
        sec_ucl, sec_lcl = _SPC_D4[subgroup] * rbar, _SPC_D3[subgroup] * rbar
        chart_name = f"X̄-R 관리도 (부분군 {subgroup}개씩)"
        sec_name = "R 관리도 — 부분군 안의 산포"
        xs_plot = pd.Series(np.arange(1, n_grp + 1))
        x_title = "부분군 번호"
    else:
        plot_vals = vals
        mr = np.abs(np.diff(vals))
        center = float(plot_vals.mean())
        mrbar = float(mr.mean()) if len(mr) else 0.0
        sigma = mrbar / _SPC_D2[2] if mrbar > 0 else float(np.std(vals, ddof=1))
        ucl, lcl = center + 3 * sigma, center - 3 * sigma
        sec_vals, sec_center = mr, mrbar
        sec_ucl, sec_lcl = _SPC_D4[2] * mrbar, 0.0
        chart_name = "I-MR 관리도 (개별 측정값)"
        sec_name = "MR 관리도 — 연속한 두 값의 차이"
        xs_plot = xs

    hits = _we_rules(plot_vals, center, sigma) if use_rules else {1: [], 2: [], 3: [], 4: []}
    out_idx = sorted(set(hits[1]))
    any_signal = sorted(set(sum(hits.values(), [])))
    in_control = len(any_signal) == 0

    h_metrics([
        ("관리도 종류", chart_name.split("(")[0].strip(), f"측정 {len(vals):,}건"),
        ("중심선 (평균)", fmt(center), f"관리한계 {fmt(lcl)} ~ {fmt(ucl)}"),
        ("관리 이탈", f"{len(out_idx)}건", "관리한계(±3σ)를 벗어난 점"),
        ("공정 상태", "안정" if in_control else "이상 신호",
         "규칙 위반 없음 👍" if in_control else f"이상 신호 {len(any_signal)}건"),
    ])

    # ── 관리도를 써도 되는 데이터인지 먼저 확인 ──────────────────────────
    # 관리도는 '공정이 일정한 수준을 유지한다'는 전제 위에서만 뜻이 있습니다.
    # 값이 계속 오르거나 계절을 타는 데이터에 쓰면 대부분의 점이 이탈로 나옵니다.
    idx_arr = np.arange(len(plot_vals))
    trend_r = abs(float(np.corrcoef(idx_arr, plot_vals)[0, 1])) if len(plot_vals) > 3 else 0.0
    viol_ratio = len(out_idx) / max(len(plot_vals), 1)
    unsuitable = trend_r > 0.5 or viol_ratio > 0.2
    if unsuitable:
        h_note(
            f"<b>이 데이터에는 관리도가 적합하지 않아 보입니다.</b><br>"
            f"값이 시간에 따라 꾸준히 변하고 있고(추세 강도 {trend_r:.2f}), "
            f"전체의 <b>{viol_ratio*100:.0f}%</b> 가 관리한계를 벗어났습니다. "
            f"관리도는 <b>“공정이 일정한 수준을 유지한다”</b>는 전제에서만 의미가 있어서, "
            f"성장·감소 추세나 계절 변동이 있는 값에 쓰면 거의 모든 점이 이탈로 표시됩니다.<br>"
            f"👉 이럴 때는 ① <b>평활화 및 트렌드 추출</b>로 추세를 먼저 파악하거나, "
            f"② <b>최근의 안정된 기간만 잘라서</b> 다시 실행하거나, "
            f"③ 매출·생산량 같은 <b>누적 성격의 값 대신 불량률·수율처럼 일정 수준을 유지해야 하는 값</b>을 "
            f"선택해 보세요.", "crit")

    # ── 관리도 ───────────────────────────────────────────────────────────
    h_sub("① 관리도")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=(f"{chart_name} — 값이 관리한계 안에서 무작위로 흔들리면 정상",
                                        sec_name),
                        row_heights=[0.62, 0.38])
    # 1σ / 2σ 구역 음영
    for k, op in ((1, 0.10), (2, 0.06)):
        fig.add_hrect(y0=center - k * sigma, y1=center + k * sigma, row=1, col=1,
                      fillcolor="#2a78d6", opacity=op, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=xs_plot, y=plot_vals, mode="lines+markers", name="측정값",
                             line=dict(color=PALETTE[0], width=1.5),
                             marker=dict(size=6, color=PALETTE[0], line=dict(width=1, color=SURFACE)),
                             hovertemplate="%{x}<br>%{y:,.4g}<extra></extra>"), row=1, col=1)
    if len(any_signal):
        fig.add_trace(go.Scatter(x=xs_plot.iloc[any_signal] if hasattr(xs_plot, "iloc") else xs_plot[any_signal],
                                 y=np.asarray(plot_vals)[any_signal], mode="markers", name="이상 신호",
                                 marker=dict(size=12, color="#d03b3b", symbol="circle-open",
                                             line=dict(width=2.6, color="#d03b3b")),
                                 hovertemplate="%{x}<br>%{y:,.4g}<br><b>이상 신호</b><extra></extra>"), row=1, col=1)
    for y, lab, color, dash in ((ucl, "UCL 관리상한", "#d03b3b", "dash"),
                               (center, "CL 중심선", "#0ca30c", "solid"),
                               (lcl, "LCL 관리하한", "#d03b3b", "dash")):
        fig.add_hline(y=y, row=1, col=1, line=dict(color=color, width=1.4, dash=dash))
        fig.add_annotation(x=1, xref="paper", y=y, yref="y", text=f"{lab} {fmt(y)}",
                           showarrow=False, xanchor="right", yshift=9,
                           font=dict(size=10.5, color=color), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs_plot[1:] if subgroup < 2 else xs_plot, y=sec_vals,
                             mode="lines+markers", name=sec_name.split("—")[0].strip(),
                             line=dict(color=PALETTE[2], width=1.4),
                             marker=dict(size=5, color=PALETTE[2]),
                             showlegend=False,
                             hovertemplate="%{x}<br>%{y:,.4g}<extra></extra>"), row=2, col=1)
    for y, color in ((sec_ucl, "#d03b3b"), (sec_center, "#0ca30c"), (sec_lcl, "#d03b3b")):
        fig.add_hline(y=y, row=2, col=1, line=dict(color=color, width=1.2,
                                                   dash="dash" if color != "#0ca30c" else "solid"))
    fig.update_xaxes(title_text=x_title, row=2, col=1)
    fig.update_yaxes(title_text=str(value_col), row=1, col=1)
    fig.update_layout(height=680)
    fig.update_annotations(font=dict(size=12.5, color=INK_2))
    show(fig)

    rule_names = {1: "① 관리한계(±3σ) 이탈 — 명백한 이상",
                  2: "② 연속 3점 중 2점이 2σ 밖 — 평균이 이동하는 중",
                  3: "③ 연속 5점 중 4점이 1σ 밖 — 한쪽으로 치우침",
                  4: "④ 연속 8점이 중심선 한쪽 — 공정 수준이 바뀜"}
    if use_rules:
        rule_tbl = pd.DataFrame([{"규칙": rule_names[k], "발생 건수": len(v),
                                  "판정": "확인 필요" if v else "정상"} for k, v in hits.items()])
        h_table(rule_tbl, "이상 신호 규칙별 점검 결과", highlight="판정")

    # ── 공정 능력 ────────────────────────────────────────────────────────
    cpk = None
    if lsl is not None or usl is not None:
        h_sub("② 공정 능력 (규격을 만족하는가)")
        mu, sd = float(np.mean(vals)), (sigma if sigma > 0 else float(np.std(vals, ddof=1)))
        cp = (usl - lsl) / (6 * sd) if (lsl is not None and usl is not None and sd > 0) else None
        cpu = (usl - mu) / (3 * sd) if (usl is not None and sd > 0) else None
        cpl = (mu - lsl) / (3 * sd) if (lsl is not None and sd > 0) else None
        cpk = min([v for v in (cpu, cpl) if v is not None], default=None)
        oos = int(((vals < lsl) if lsl is not None else False).sum()
                  + ((vals > usl) if usl is not None else False).sum())
        grade = ("우수 — 여유 있게 만족" if cpk and cpk >= 1.33 else
                 "보통 — 관리 필요" if cpk and cpk >= 1.0 else
                 "부족 — 불량 위험 높음" if cpk is not None else "—")
        h_metrics([
            ("Cp (잠재 능력)", f"{cp:.2f}" if cp else "—", "규격 폭 대비 산포"),
            ("Cpk (실제 능력)", f"{cpk:.2f}" if cpk else "—", grade),
            ("규격 이탈", f"{oos:,}건", f"전체의 {oos/len(vals)*100:.2f}%"),
            ("규격", f"{fmt(lsl) if lsl is not None else '—'} ~ {fmt(usl) if usl is not None else '—'}",
             f"평균 {fmt(mu)}"),
        ])
        fig = go.Figure(go.Histogram(x=vals, nbinsx=45, name="측정값 분포",
                                     marker=dict(color=PALETTE[0], line=dict(width=0)),
                                     hovertemplate="%{x}<br>%{y:,}건<extra></extra>"))
        for spec, lab, color in ((lsl, "LSL 규격하한", "#d03b3b"), (usl, "USL 규격상한", "#d03b3b")):
            if spec is not None:
                fig.add_vline(x=spec, line=dict(color=color, width=2))
                fig.add_annotation(x=spec, y=1, yref="paper", text=lab, showarrow=False,
                                   xanchor="left", xshift=5, font=dict(size=11, color=color))
        fig.add_vline(x=mu, line=dict(color="#0ca30c", width=1.6, dash="dot"))
        fig.update_layout(title="측정값이 규격 안에 충분히 들어와 있는지 확인하세요",
                          xaxis_title=str(value_col), yaxis_title="건수", height=400, bargap=0.03)
        show(fig)

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = []
    if unsuitable:
        tips.append("⚠️ <b>결과를 그대로 믿지 마세요.</b> 이 데이터는 시간에 따라 계속 변하는 성격이라 "
                    "관리도의 전제와 맞지 않습니다. 위 안내대로 대상 값이나 기간을 바꿔 다시 실행해 보세요.")
    if in_control:
        tips.append("공정은 <b>통계적으로 안정된 상태</b>입니다. 값이 관리한계 안에서 무작위로 흔들리고 있어, "
                    "특별한 원인 없이 늘 있는 정도의 변동만 보입니다.")
    else:
        first_rule = next((k for k in (1, 2, 3, 4) if hits[k]), None)
        tips.append(f"<b>이상 신호가 {len(any_signal)}건</b> 발견되었습니다. "
                    f"특히 <b>{rule_names[first_rule]}</b> 항목을 확인하세요.")
        if hits[1]:
            pos = hits[1][0]
            when = (f"{xs_plot.iloc[pos]:%Y-%m-%d}" if date_col and subgroup < 2
                    else f"{pos+1}번째 측정")
            tips.append(f"관리한계를 벗어난 첫 지점은 <b>{when}</b> 입니다. "
                        f"이 시점의 작업 조건·자재·설비 변경 이력을 확인해 보세요.")
        if hits[4]:
            tips.append("연속으로 한쪽에 치우친 구간이 있습니다. 공정 <b>평균 자체가 이동</b>했을 가능성이 큽니다. "
                        "설비 세팅이나 자재 로트가 바뀌지 않았는지 점검하세요.")
    if cpk is not None:
        tips.append(f"공정 능력 <b>Cpk = {cpk:.2f}</b> — {grade}. "
                    + ("일반적으로 1.33 이상을 목표로 합니다." if cpk < 1.33 else
                       "현재 수준을 유지하는 관리가 중요합니다."))
        if cpk < 1.0:
            tips.append("Cpk가 1.0 미만이면 <b>규격을 벗어나는 제품이 꾸준히 나올 수 있습니다.</b> "
                        "산포를 줄이거나(설비·작업 표준화) 평균을 규격 중앙으로 옮기는 조치가 필요합니다.")
    tips.append("❗ 관리한계(UCL/LCL)는 <b>규격이 아니라 공정 자신의 목소리</b>입니다. "
                "‘이 공정이 평소에 내는 변동 범위’를 뜻하며, 고객 규격(LSL/USL)과는 다른 개념입니다.")
    h_insight(tips)

    out = work.copy()
    if subgroup < 2:
        out["중심선"], out["관리상한"], out["관리하한"] = center, ucl, lcl
        out["이상신호"] = ["예" if i in set(any_signal) else "" for i in range(len(out))]
    RESULT_FILES.append(("관리도_결과.csv", out))
    return out


def run_pareto_analysis():
    """[품질관리] 파레토 분석 — 어디부터 손대야 효과가 큰지 찾습니다."""
    h_title("📊 파레토 분석 (80 / 20 법칙)",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    cat_col = resolve_col(GROUP_COL, kind="category", required=True, purpose="분류(원인·항목) 컬럼")
    top_n = max(3, P_int("top_n", 15))
    target_pct = min(max(P_float("cum_target", 80), 50), 99)
    agg = P_str("agg", "sum")

    value_col = None
    if TARGET_COL:
        c = resolve_col(TARGET_COL)
        if c and pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
        elif c:
            h_note(f"'{c}' 은(는) 숫자가 아니어서 <b>건수(개수)</b> 기준으로 집계합니다.", "warn")

    if value_col:
        s = (df[[cat_col, value_col]].dropna()
             .assign(**{value_col: lambda d: pd.to_numeric(d[value_col], errors="coerce")})
             .dropna().groupby(cat_col)[value_col].agg(agg))
        unit_label = f"{value_col} {'합계' if agg == 'sum' else '평균'}"
    else:
        s = df[cat_col].dropna().astype(str).value_counts()
        unit_label = "건수"

    s = s.sort_values(ascending=False)
    if s.empty or s.sum() <= 0:
        h_note("집계할 값이 없습니다. 분류 컬럼과 값 컬럼을 다시 확인해 주세요.", "crit")
        return None

    total = float(s.sum())
    shown = s.head(top_n)
    others = float(s.iloc[top_n:].sum())
    if others > 0:
        shown = pd.concat([shown, pd.Series({f"기타 ({len(s) - top_n}개)": others})])
    pct = shown / total * 100
    cum = pct.cumsum()
    vital_n = int((cum < target_pct).sum()) + 1
    vital_n = min(vital_n, len(shown))
    vital = list(shown.index[:vital_n])

    h_metrics([
        ("분류 항목 수", f"{len(s):,}개", str(cat_col)),
        ("전체 합계", fmt(total), unit_label),
        (f"핵심 항목 (상위 {target_pct:.0f}%)", f"{vital_n}개",
         f"전체 항목의 {vital_n/len(s)*100:.0f}%"),
        ("핵심이 차지하는 비중", f"{float(cum.iloc[vital_n-1]):.1f}%", "여기부터 개선하면 효과가 큼"),
    ])

    # ── 파레토 차트 ──────────────────────────────────────────────────────
    # 누적선을 같은 그림에 겹치면 두 개의 세로축이 필요해 값이 왜곡돼 보입니다.
    # 그래서 위아래로 나누되 가로축을 공유해, 같은 항목을 나란히 읽도록 했습니다.
    h_sub("① 파레토 차트")
    labels = [str(i) for i in shown.index]
    colors = [PALETTE[0] if i < vital_n else "#c3c2b7" for i in range(len(shown))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.62, 0.38],
                        subplot_titles=(f"항목별 {unit_label} — 파란색이 핵심 소수",
                                        f"누적 비율 — {target_pct:.0f}% 선을 넘는 지점까지가 핵심"))
    fig.add_trace(go.Bar(x=labels, y=shown.values, marker=dict(color=colors, line=dict(width=0)),
                         text=[f"{v:,.4g}" for v in shown.values], textposition="outside",
                         cliponaxis=False, showlegend=False,
                         hovertemplate="%{x}<br>%{y:,.4g}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=cum.values, mode="lines+markers",
                             line=dict(color=PALETTE[1], width=2.2),
                             marker=dict(size=8, color=PALETTE[1], line=dict(width=1, color=SURFACE)),
                             showlegend=False,
                             hovertemplate="%{x}<br>누적 %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_hline(y=target_pct, row=2, col=1, line=dict(color=MUTED, width=1.3, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=target_pct, yref="y2", text=f"{target_pct:.0f}%",
                       showarrow=False, xanchor="right", yshift=9, font=dict(size=11, color=MUTED))
    fig.update_yaxes(title_text=unit_label, row=1, col=1)
    fig.update_yaxes(title_text="누적 비율 (%)", row=2, col=1, range=[0, 105])
    fig.update_xaxes(tickangle=-35, row=2, col=1)
    fig.update_layout(height=680, bargap=0.25)
    fig.update_annotations(font=dict(size=12.5, color=INK_2))
    show(fig)

    tbl = pd.DataFrame({
        "순위": range(1, len(shown) + 1),
        str(cat_col): labels,
        unit_label: shown.values.round(4),
        "비중(%)": pct.values.round(2),
        "누적(%)": cum.values.round(2),
        "구분": ["🎯 핵심" if i < vital_n else "" for i in range(len(shown))],
    })
    h_table(tbl, "항목별 집계와 누적 비율", max_rows=top_n + 1, highlight="구분")

    # ── 자동 해석 ────────────────────────────────────────────────────────
    top1 = shown.index[0]
    tips = [
        f"전체 <b>{len(s):,}개</b> 항목 중 상위 <b>{vital_n}개</b> 가 전체의 "
        f"<b>{float(cum.iloc[vital_n-1]):.1f}%</b> 를 차지합니다. "
        f"이 {vital_n}개만 해결해도 문제의 대부분이 줄어듭니다.",
        f"1위는 <b>{_html.escape(str(top1))}</b> 로 혼자서 전체의 <b>{float(pct.iloc[0]):.1f}%</b> 입니다. "
        f"가장 먼저 손댈 대상입니다.",
        f"핵심 항목: <b>{_html.escape(', '.join(map(str, vital[:6])))}</b>"
        + (" 등" if len(vital) > 6 else ""),
    ]
    ratio = vital_n / len(s) * 100
    if ratio <= 30:
        tips.append(f"항목의 <b>{ratio:.0f}%</b> 만으로 대부분이 설명되는 <b>전형적인 파레토 형태</b>입니다. "
                    f"선택과 집중 전략이 잘 통합니다.")
    else:
        tips.append(f"원인이 여러 항목에 <b>고르게 퍼져 있습니다</b>(상위 {ratio:.0f}% 필요). "
                    f"한두 개를 고쳐서는 효과가 크지 않으니, 공통 원인이 있는지 살펴보세요.")
    tips.append("다음 단계 추천: 핵심 항목을 <b>그룹 간 차이 비교</b>로 검증하거나, "
                "시간에 따라 늘고 있는지 <b>시계열</b> 로 확인해 보세요.")
    h_insight(tips)

    RESULT_FILES.append(("파레토_분석.csv", tbl))
    return tbl


def run_data_quality():
    """[데이터 정제] 품질 점수를 매기고 문제를 자동으로 고쳐 줍니다."""
    h_title("🧹 데이터 품질 진단 및 자동 정제",
            f"파일: {_html.escape(str(DATA_SOURCE_NAME))} · 생성 시각: {datetime.now():%Y-%m-%d %H:%M}")

    fix_dup = P_bool("fix_duplicates", True)
    missing_how = P_str("fix_missing", "keep")
    outlier_how = P_str("fix_outliers", "flag")
    drop_useless = P_bool("drop_useless", True)
    trim_text = P_bool("trim_text", True)

    work = df.copy()
    n0, c0 = work.shape
    issues = []

    def add_issue(kind, target, count, severity, action):
        issues.append({"문제 유형": kind, "대상": target, "건수": count,
                       "심각도": severity, "조치": action})

    nums = numeric_columns()

    # ── 진단 ─────────────────────────────────────────────────────────────
    n_dup = int(work.duplicated().sum())
    if n_dup:
        add_issue("중복된 행", "전체", f"{n_dup:,}행", "높음" if n_dup / n0 > 0.05 else "보통",
                  "제거" if fix_dup else "유지")

    miss = work.isna().mean()
    for c in work.columns[miss > 0]:
        r = miss[c]
        add_issue("빈 값(결측치)", str(c), f"{int(work[c].isna().sum()):,}칸 ({r*100:.1f}%)",
                  "높음" if r > 0.3 else "보통" if r > 0.05 else "낮음",
                  {"keep": "유지", "drop": "행 제거", "median": "중앙값으로 채움",
                   "ffill": "직전 값으로 채움"}[missing_how])

    const_cols = [c for c in work.columns if work[c].nunique(dropna=True) <= 1]
    for c in const_cols:
        add_issue("항상 같은 값", str(c), "전체", "보통", "제거" if drop_useless else "유지")

    id_like = [c for c in work.columns
               if work[c].dtype == object and work[c].nunique(dropna=True) > 0.95 * n0 and n0 > 20]
    for c in id_like:
        add_issue("사실상 고유값 (ID 성격)", str(c), f"{work[c].nunique():,}종", "낮음", "분석에서 제외 권장")

    n_out_total = 0
    for c in nums:
        n_out, lo, hi = _iqr_outliers(work[c])
        if n_out:
            n_out_total += n_out
            add_issue("튀는 값(이상치 후보)", str(c), f"{n_out:,}건",
                      "높음" if n_out / n0 > 0.1 else "보통" if n_out / n0 > 0.02 else "낮음",
                      {"flag": "표시만", "clip": "경계값으로 조정", "remove": "행 제거"}[outlier_how])

    text_cols = [c for c in work.columns if work[c].dtype == object]
    n_space = 0
    for c in text_cols:
        s = work[c].dropna().astype(str)
        bad = int((s != s.str.strip()).sum())
        if bad:
            n_space += bad
            add_issue("앞뒤 공백", str(c), f"{bad:,}칸", "낮음", "제거" if trim_text else "유지")

    mixed = []
    for c in text_cols:
        s = work[c].dropna().astype(str)
        if len(s) < 5:
            continue
        num_like = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce").notna().mean()
        if 0.5 < num_like < 0.95:
            mixed.append(c)
            add_issue("숫자와 글자가 섞임", str(c), f"숫자처럼 보이는 값 {num_like*100:.0f}%",
                      "높음", "원본 확인 필요")

    # ── 품질 점수 ────────────────────────────────────────────────────────
    completeness = (1 - work.isna().mean().mean()) * 100
    uniqueness = (1 - n_dup / max(n0, 1)) * 100
    validity = max(0.0, (1 - n_out_total / max(n0 * max(len(nums), 1), 1)) * 100)
    consistency = (1 - (len(const_cols) + len(mixed)) / max(c0, 1)) * 100
    overall = float(np.mean([completeness, uniqueness, validity, consistency]))
    grade = ("매우 좋음" if overall >= 90 else "양호" if overall >= 75 else
             "보통" if overall >= 60 else "정리 필요")

    h_metrics([
        ("종합 품질 점수", f"{overall:.0f}점", f"100점 만점 · {grade}"),
        ("데이터 크기", f"{n0:,}행 × {c0}열", str(DATA_SOURCE_NAME)[:26]),
        ("발견된 문제", f"{len(issues)}건", f"높음 {sum(1 for i in issues if i['심각도']=='높음')}건"),
        ("바로 쓸 수 있나", "예" if overall >= 75 and not mixed else "정리 후 권장",
         "아래 조치를 적용합니다"),
    ])

    h_sub("① 품질 점수 상세")
    dims = [("완전성", completeness, "빈 값 없이 채워져 있는가"),
            ("유일성", uniqueness, "중복 없이 한 번씩만 있는가"),
            ("유효성", validity, "값이 정상 범위 안에 있는가"),
            ("일관성", consistency, "형식과 의미가 일정한가")]
    fig = go.Figure(go.Bar(
        x=[d[1] for d in dims][::-1], y=[d[0] for d in dims][::-1], orientation="h",
        marker=dict(color=[("#0ca30c" if v >= 90 else "#fab219" if v >= 70 else "#d03b3b")
                           for _, v, _ in dims][::-1], line=dict(width=0)),
        text=[f"{v:.0f}점" for _, v, _ in dims][::-1], textposition="outside", cliponaxis=False,
        customdata=[d[2] for d in dims][::-1],
        hovertemplate="%{y}: %{x:.1f}점<br>%{customdata}<extra></extra>"))
    fig.add_vline(x=90, line=dict(color=MUTED, width=1.2, dash="dot"))
    fig.update_layout(title="항목별 품질 점수 (점선 = 90점 목표선)", xaxis_title="점수",
                      xaxis=dict(range=[0, 108]), height=330, showlegend=False)
    show(fig)
    h_note("초록 = 문제 없음 · 노랑 = 확인 권장 · 빨강 = 정리 필요. "
           "막대에 마우스를 올리면 각 항목의 뜻이 나옵니다.")

    h_sub("② 발견된 문제 목록")
    if issues:
        it = pd.DataFrame(issues)
        order = {"높음": 0, "보통": 1, "낮음": 2}
        it = it.sort_values("심각도", key=lambda s: s.map(order)).reset_index(drop=True)
        h_table(it, "심각도가 높은 순서입니다", max_rows=50, highlight="심각도")
    else:
        h_note("발견된 문제가 없습니다. 아주 깨끗한 데이터입니다.", "good")

    # ── 자동 정제 ────────────────────────────────────────────────────────
    h_sub("③ 자동 정제 실행")
    log = []
    if trim_text:
        for c in text_cols:
            if c in work.columns:
                work[c] = work[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
        if n_space:
            log.append(f"앞뒤 공백 {n_space:,}칸을 정리했습니다.")
    if fix_dup and n_dup:
        work = work.drop_duplicates()
        log.append(f"중복된 행 {n_dup:,}건을 제거했습니다.")
    if drop_useless and const_cols:
        work = work.drop(columns=const_cols)
        log.append(f"값이 항상 같은 컬럼 {len(const_cols)}개를 제거했습니다: "
                   f"{_html.escape(', '.join(map(str, const_cols[:5])))}")
    if missing_how == "drop":
        before = len(work)
        work = work.dropna()
        log.append(f"빈 값이 있는 행 {before - len(work):,}건을 제거했습니다.")
    elif missing_how == "median":
        for c in work.columns:
            if pd.api.types.is_numeric_dtype(work[c]) and work[c].isna().any():
                work[c] = work[c].fillna(work[c].median())
        log.append("숫자 컬럼의 빈 값을 중앙값으로 채웠습니다.")
    elif missing_how == "ffill":
        work = work.ffill().bfill()
        log.append("빈 값을 직전(없으면 직후) 값으로 채웠습니다.")
    if outlier_how in ("clip", "remove"):
        touched = 0
        for c in [c for c in numeric_columns() if c in work.columns]:
            n_out, lo, hi = _iqr_outliers(work[c])
            if not n_out or lo is None:
                continue
            if outlier_how == "clip":
                work[c] = work[c].clip(lo, hi)
            else:
                work = work[(work[c] >= lo) & (work[c] <= hi)]
            touched += n_out
        if touched:
            log.append(f"튀는 값 {touched:,}건을 "
                       + ("정상 범위 경계값으로 조정했습니다." if outlier_how == "clip" else "제거했습니다."))
    if not log:
        log.append("설정한 조치가 없어 원본을 그대로 유지했습니다.")
    for line in log:
        h_note(line, "good")

    n1, c1 = work.shape
    comp_after = (1 - work.isna().mean().mean()) * 100 if c1 else 100
    dup_after = int(work.duplicated().sum())
    fig = go.Figure()
    cats = ["행 수", "열 수", "빈 값 비율(%)", "중복 행"]
    before_v = [n0, c0, round(df.isna().mean().mean() * 100, 2), n_dup]
    after_v = [n1, c1, round(100 - comp_after, 2), dup_after]
    fig.add_trace(go.Bar(x=cats, y=before_v, name="정제 전",
                         marker=dict(color="#c3c2b7", line=dict(width=0)),
                         text=[f"{v:,}" for v in before_v], textposition="outside", cliponaxis=False,
                         hovertemplate="정제 전 %{x}<br>%{y:,}<extra></extra>"))
    fig.add_trace(go.Bar(x=cats, y=after_v, name="정제 후",
                         marker=dict(color=PALETTE[0], line=dict(width=0)),
                         text=[f"{v:,}" for v in after_v], textposition="outside", cliponaxis=False,
                         hovertemplate="정제 후 %{x}<br>%{y:,}<extra></extra>"))
    fig.update_layout(title="정제 전후 비교", barmode="group", height=400,
                      yaxis_title="값", bargap=0.28, bargroupgap=0.08)
    show(fig)

    h_table(work.head(8), "정제된 데이터 미리보기", max_rows=8)

    # ── 자동 해석 ────────────────────────────────────────────────────────
    tips = [f"이 데이터의 종합 품질 점수는 <b>{overall:.0f}점 / 100점</b> 으로 <b>{grade}</b> 수준입니다."]
    worst_dim = min(dims, key=lambda d: d[1])
    if worst_dim[1] < 90:
        tips.append(f"가장 약한 부분은 <b>{worst_dim[0]}</b>({worst_dim[1]:.0f}점) — {worst_dim[2]} 입니다.")
    if mixed:
        tips.append(f"⚠️ <b>{_html.escape(', '.join(map(str, mixed[:4])))}</b> 컬럼은 숫자와 글자가 섞여 있습니다. "
                    f"이런 컬럼은 계산이 안 되므로, 원본 엑셀에서 '측정불가' 같은 글자를 "
                    f"<b>빈칸으로 바꿔</b> 다시 올리시는 것이 가장 좋습니다.")
    if n_dup:
        tips.append(f"중복 행 {n_dup:,}건은 같은 자료를 두 번 입력했을 가능성이 큽니다. "
                    f"집계 수치가 부풀려지므로 제거를 권합니다.")
    if n_out_total:
        tips.append(f"튀는 값 후보가 총 {n_out_total:,}건입니다. <b>실수로 잘못 입력한 값</b>일 수도, "
                    f"<b>진짜 중요한 이상 신호</b>일 수도 있습니다. 지우기 전에 "
                    f"<b>이상 탐지</b> 분석으로 확인해 보시길 권합니다.")
    tips.append("정제된 데이터는 CSV로 저장됩니다. <b>이 파일을 다시 올려</b> 다른 분석을 돌리면 "
                "결과가 더 정확해집니다.")
    h_insight(tips)

    RESULT_FILES.append(("정제된_데이터.csv", work))
    if issues:
        RESULT_FILES.append(("품질진단_문제목록.csv", pd.DataFrame(issues)))
    return work


# ═══════════════════════════════════════════════════════════════════════
#  분석 설정 고르기 — 타이핑 없이 드롭다운으로
# ═══════════════════════════════════════════════════════════════════════
def _col_kinds():
    """각 컬럼이 날짜인지 숫자인지 글자인지 미리 판정해 둡니다."""
    kinds = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            kinds[c] = "date"
        elif pd.api.types.is_numeric_dtype(s):
            kinds[c] = "number"
        else:
            try:
                if pd.to_datetime(s, errors="coerce").notna().mean() > 0.9:
                    kinds[c] = "date"
                    continue
            except Exception:
                pass
            kinds[c] = "text"
    return kinds


def _auto_defaults():
    """포털 설정이 없을 때 쓸 기본 컬럼을 데이터에서 골라 둡니다."""
    kinds = _col_kinds()
    dates = [c for c, k in kinds.items() if k == "date"]
    nums = [c for c, k in kinds.items() if k == "number" and df[c].nunique() > 2]
    cats = [c for c, k in kinds.items() if k != "number" and 2 <= df[c].nunique() <= 20]
    need = MODULE_NEEDS.get(MODULE_ID, {})
    out = {}
    if need.get("date", "none") != "none" and not DATE_COL and dates:
        out["date_col"] = dates[0]
    if need.get("target", "none") != "none" and not TARGET_COL:
        if MODULE_ID == "classify_xgb" and cats:
            out["target_col"] = cats[0]
        elif nums:
            out["target_col"] = nums[0]
    if need.get("group", "none") != "none" and not GROUP_COL and cats:
        out["group_col"] = cats[0]
    return out


def choose_options(show_widgets=True):
    """2단계 — 분석에 쓸 컬럼을 드롭다운으로 고릅니다. (타이핑 불필요)

    포털에서 설정이 넘어왔으면 그 값이 미리 선택돼 있습니다.
    그대로 두고 다음 셀을 실행하면 됩니다.
    """
    if df is None:
        print("⚠️  먼저 데이터를 올려 주세요. (2단계 셀)")
        return None

    auto = _auto_defaults()
    if auto:
        CONFIG.update(auto)
        configure()

    kinds = _col_kinds()
    need = MODULE_NEEDS.get(MODULE_ID, {})
    cols = list(df.columns)

    def label(c):
        return f"{c}  ({'날짜' if kinds[c]=='date' else '숫자' if kinds[c]=='number' else '글자'})"

    if not show_widgets:
        _print_settings()
        return None

    try:
        import ipywidgets as widgets
        from IPython.display import display as _display
    except Exception:
        print("드롭다운을 표시할 수 없어 자동으로 고른 설정으로 진행합니다.")
        _print_settings()
        return None

    rows, controls = [], {}
    style = {"description_width": "170px"}
    layout = widgets.Layout(width="620px")

    mod_dd = widgets.Dropdown(
        options=[(f"{name}", mid) for mid, name, _ in MODULE_CATALOG],
        value=MODULE_ID, description="분석 종류", style=style, layout=layout)
    rows.append(mod_dd)
    controls["module_id"] = mod_dd

    spec = [("date_col", "날짜 / 시간 컬럼", ["date"]),
            ("group_col", "분류(그룹) 컬럼" if MODULE_ID == "pareto" else "그룹(비교 기준) 컬럼", None),
            ("target_col", "분석할 값 컬럼", None)]
    for key, desc, prefer in spec:
        kind_need = need.get(key.split("_")[0], "none")
        if kind_need == "none":
            continue
        opts = [("(자동으로 고르기)", "")]
        pri = [c for c in cols if not prefer or kinds[c] in prefer]
        rest = [c for c in cols if c not in pri]
        opts += [(label(c), c) for c in pri] + [(label(c), c) for c in rest]
        cur = CONFIG.get(key, "")
        dd = widgets.Dropdown(options=opts, value=cur if cur in cols else "",
                              description=desc + (" *" if kind_need == "required" else ""),
                              style=style, layout=layout)
        rows.append(dd)
        controls[key] = dd

    if need.get("features", "none") != "none":
        num_cols = [c for c in cols if kinds[c] == "number"]
        sel = widgets.SelectMultiple(
            options=[(label(c), c) for c in cols],
            value=tuple(c for c in FEATURE_COLS if c in cols),
            description="사용할 컬럼", style=style,
            layout=widgets.Layout(width="620px", height="140px"))
        rows.append(sel)
        rows.append(widgets.HTML(
            "<div style='color:#666;font-size:12px;margin:-6px 0 6px 176px'>"
            "아무것도 고르지 않으면 숫자 컬럼을 자동으로 모두 사용합니다. "
            "(여러 개는 Ctrl 또는 Shift 를 누른 채 클릭)</div>"))
        controls["feature_cols"] = sel

    status = widgets.HTML()

    def _sync(_=None):
        for key, w in controls.items():
            val = w.value
            if key == "feature_cols":
                CONFIG["feature_cols"] = ",".join(val)
            elif val:
                CONFIG[key] = val
            else:
                CONFIG.pop(key, None)
        configure(module_id=controls["module_id"].value)
        ok, msg = _check_ready()
        color = "#0a7a0a" if ok else "#c26a00"
        status.value = (f"<div style='font-size:13px;color:{color};margin-top:6px'>"
                        f"{'✅' if ok else '⚠️'} {msg}</div>")

    for w in controls.values():
        w.observe(_sync, names="value")
    _sync()

    _display(widgets.HTML(
        "<div style='font-family:sans-serif;font-size:14px;font-weight:700;margin-bottom:6px'>"
        "🎛️ 분석 설정 — 바꾸고 싶은 것만 고르세요</div>"
        "<div style='font-family:sans-serif;font-size:12.5px;color:#555;margin-bottom:10px'>"
        "포털에서 넘어온 설정이 이미 선택돼 있습니다. 그대로 두셔도 됩니다.<br>"
        "설정을 바꾼 뒤에는 <b>아래 3단계 셀만 다시 실행</b>하면 됩니다. (처음부터 다시 할 필요 없습니다)</div>"))
    _display(widgets.VBox(rows + [status]))
    return controls


def _check_ready():
    """필수 컬럼이 채워졌는지 확인해 한국어 안내 문구를 돌려줍니다."""
    need = MODULE_NEEDS.get(MODULE_ID, {})
    miss = []
    if need.get("date") == "required" and not P_str("date_col"):
        miss.append("날짜 컬럼")
    if need.get("target") == "required" and not P_str("target_col"):
        miss.append("분석할 값 컬럼")
    if need.get("group") == "required" and not P_str("group_col"):
        miss.append("그룹 컬럼")
    if miss:
        return False, f"{', '.join(miss)}을(를) 골라 주세요. (자동 선택이 어려운 항목입니다)"
    return True, f"{MODULE_NAMES.get(MODULE_ID, MODULE_ID)} — 준비되었습니다. 아래 3단계 셀을 실행하세요."


def _print_settings():
    print("\n현재 분석 설정")
    print(f"  · 분석 종류 : {MODULE_NAMES.get(MODULE_ID, MODULE_ID)}")
    for key, name in (("date_col", "날짜 컬럼"), ("group_col", "그룹 컬럼"),
                      ("target_col", "분석할 값"), ("feature_cols", "사용 컬럼")):
        val = P_str(key) if key != "feature_cols" else ", ".join(P_list(key))
        if val:
            print(f"  · {name} : {val}")
    ok, msg = _check_ready()
    print(f"  {'✅' if ok else '⚠️'} {msg}")


# ═══════════════════════════════════════════════════════════════════════
#  분석 실행 (모듈 라우터)
# ═══════════════════════════════════════════════════════════════════════
def run_module():
    """고른 분석 하나만 실행합니다."""
    global RESULT, FAILED
    import traceback
    RESULT, FAILED = None, False

    print("═" * 66)
    print(f"▶ 실행할 분석: {MODULE_NAMES.get(MODULE_ID, MODULE_ID)}")
    print("═" * 66 + "\n")

    try:
        if MODULE_ID == "eda_summary":            # [탐색적 분석] 데이터 자동 요약
            RESULT = run_eda_summary()
        elif MODULE_ID == "data_quality":         # [탐색적 분석] 품질 진단·자동 정제
            RESULT = run_data_quality()
        elif MODULE_ID == "eda_corr":             # [탐색적 분석] 상관관계 히트맵
            RESULT = run_eda_corr()
        elif MODULE_ID == "compare_groups":       # [비교·검정] 그룹 간 차이
            RESULT = run_compare_groups()
        elif MODULE_ID == "ts_forecast":          # [시계열] Prophet 예측
            RESULT = run_ts_forecast()
        elif MODULE_ID == "ts_smooth":            # [시계열] 평활화·트렌드
            RESULT = run_ts_smooth()
        elif MODULE_ID == "anomaly_iforest":      # [이상 탐지] Isolation Forest
            RESULT = run_anomaly_iforest()
        elif MODULE_ID == "spc_control":          # [품질관리] 공정 관리도
            RESULT = run_spc_control_chart()
        elif MODULE_ID == "pareto":               # [품질관리] 파레토 분석
            RESULT = run_pareto_analysis()
        elif MODULE_ID == "regress_rf":           # [수치 예측] Random Forest
            RESULT = run_regress_rf()
        elif MODULE_ID == "classify_xgb":         # [상태 분류] XGBoost
            RESULT = run_classify_xgb()
        elif MODULE_ID == "cluster_kmeans":       # [군집화] K-Means
            RESULT = run_cluster_kmeans()
        # 새 분석은 여기에 elif 를 한 줄 추가하세요.
        else:
            h_note(f"'{MODULE_ID}' 는 아직 준비되지 않은 분석입니다.", "crit")
    except Exception as exc:
        FAILED = True
        h_note(f"<b>분석 중 문제가 발생했습니다.</b><br>{_html.escape(str(exc))}", "crit")
        h_insight([
            "가장 흔한 원인은 <b>컬럼을 잘못 고른 경우</b>입니다. "
            "위 2단계 셀의 드롭다운에서 다른 컬럼을 골라 <b>이 셀만 다시 실행</b>해 보세요.",
            f"이 파일의 컬럼: <b>{_html.escape(', '.join(map(str, df.columns[:25])))}</b>",
            "숫자여야 할 컬럼에 글자(예: '측정불가', '-')가 섞여 있으면 오류가 납니다. "
            "<b>데이터 품질 진단</b> 분석을 먼저 돌려 확인해 보세요.",
            "그래도 해결되지 않으면 아래 상세 오류 내용을 담당자에게 전달해 주세요.",
        ], title="이럴 때 이렇게 해보세요")
        print("\n──── 상세 오류 내용 (담당자 전달용) ────")
        traceback.print_exc()
    return RESULT


# ═══════════════════════════════════════════════════════════════════════
#  리포트 저장
# ═══════════════════════════════════════════════════════════════════════
def save_report(download=True):
    """그래프·표·해석을 HTML 리포트로 묶고 결과 CSV와 함께 ZIP 으로 저장합니다."""
    if not REPORT_PARTS:
        print("저장할 결과가 없습니다.")
        return None

    # 그래프 엔진을 파일 안에 넣어, 인터넷이 없어도 리포트가 열리게 합니다.
    try:
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    except Exception:
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>'

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = f"분석리포트_{MODULE_ID}_{stamp}"
    report_name = base + ".html"

    toc_html = ""
    if len(REPORT_TOC) > 1:
        items = "".join(
            f'<li style="margin:{"7px 0 3px" if lvl == 1 else "3px 0 3px 16px"}">'
            f'<a href="#{a}" style="color:{"#184f95" if lvl == 1 else INK_2};'
            f'text-decoration:none;font-weight:{700 if lvl == 1 else 500};'
            f'font-size:{13.5 if lvl == 1 else 13}px">{_html.escape(t)}</a></li>'
            for lvl, t, a in REPORT_TOC)
        toc_html = (f'<div class="toc"><div style="font-weight:700;font-size:14px;margin-bottom:8px">'
                    f'📑 목차 <span style="font-weight:500;color:{MUTED};font-size:12.5px">'
                    f'— 눌러서 바로 이동</span></div>'
                    f'<ul style="margin:0;padding-left:18px;list-style:none">{items}</ul></div>')

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(MODULE_NAMES.get(MODULE_ID, MODULE_ID))} 리포트</title>
{plotly_tag}
<style>
  html{{scroll-behavior:smooth}}
  body{{margin:0;background:#f9f9f7;color:{INK};font-family:{FONT};line-height:1.6}}
  .wrap{{max-width:1080px;margin:0 auto;padding:26px 20px 80px}}
  .cover{{background:{SURFACE};border:1px solid rgba(11,11,11,.12);border-radius:16px;
         padding:26px 28px;margin-bottom:14px;box-shadow:0 8px 24px rgba(11,11,11,.06)}}
  .cover h1{{margin:0 0 6px;font-size:26px}}
  .cover p{{margin:0;color:{INK_2};font-size:14px}}
  .howto{{background:#f4f8fd;border:1px solid #cde2fb;border-radius:12px;
         padding:14px 18px;margin-bottom:14px;font-size:13.5px;line-height:1.75}}
  .toc{{background:{SURFACE};border:1px solid rgba(11,11,11,.12);border-radius:12px;
       padding:15px 18px;margin-bottom:10px}}
  .toc a:hover{{text-decoration:underline}}
  .foot{{margin-top:40px;padding-top:16px;border-top:1px solid {GRID};
        color:{MUTED};font-size:12.5px;text-align:center;line-height:1.9}}
</style></head><body><div class="wrap">
<div class="cover">
  <h1>{_html.escape(MODULE_NAMES.get(MODULE_ID, MODULE_ID))}</h1>
  <p>데이터: {_html.escape(str(DATA_SOURCE_NAME))} · 생성: {datetime.now():%Y년 %m월 %d일 %H:%M} ·
     대상: {RAW_ROWS:,}행 × {RAW_COLS}열</p>
</div>
<div class="howto">
  <b>📖 이 리포트 읽는 법</b><br>
  · 그래프에 <b>마우스를 올리면</b> 정확한 값이 나오고, <b>드래그하면</b> 그 구간만 확대됩니다. (더블클릭하면 원래대로)<br>
  · 어려운 숫자는 넘기셔도 됩니다. 각 그래프 아래 <b>🧭 “이 결과를 이렇게 읽으세요”</b> 상자에 결론이 한국어로 정리되어 있습니다.<br>
  · 이 파일은 <b>인터넷 없이도</b> 열립니다. 그대로 동료에게 보내셔도 똑같이 보입니다.
</div>
{toc_html}
{''.join(REPORT_PARTS)}
<div class="foot">
  올인원 데이터 분석 포털에서 자동 생성된 리포트입니다.<br>
  분석 종류를 바꾸시려면 포털로 돌아가 다른 카드를 선택하세요.
</div>
</div></body></html>"""

    with open(report_name, "w", encoding="utf-8") as f:
        f.write(doc)

    saved = [report_name]
    for fname, frame in RESULT_FILES:
        try:
            safe = f"{base}__{fname}"
            frame.to_csv(safe, index=False, encoding="utf-8-sig")   # 엑셀 한글 안 깨짐
            saved.append(safe)
        except Exception as exc:
            print(f"  ⚠️  '{fname}' 저장 실패: {exc}")

    zip_name = base + ".zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for p in saved:
            z.write(p)

    print("═" * 66)
    print("📦 결과가 준비되었습니다")
    print("═" * 66)
    for p in saved:
        print(f"   · {p}")
    print(f"\n   → {zip_name} ({os.path.getsize(zip_name)/1024:,.0f} KB) 를 내려받습니다.")
    print("     (자동으로 시작되지 않으면 왼쪽 📁 폴더 아이콘에서 받으실 수 있습니다.)")

    if IN_COLAB and download:
        try:
            from google.colab import files as colab_files
            colab_files.download(zip_name)
        except Exception as exc:
            print(f"   ⚠️  자동 다운로드 실패({exc}) — 왼쪽 폴더 아이콘에서 받아 주세요.")
    return zip_name


def run_and_report(download=True):
    """3단계 — 분석 실행 + 리포트 저장까지 한 번에."""
    global REPORT_PARTS, RESULT_FILES, REPORT_TOC
    if df is None:
        print("⚠️  먼저 2단계에서 데이터를 올려 주세요.")
        return None

    ok, msg = _check_ready()
    if not ok:
        h_note(f"<b>{msg}</b><br>위 2단계 셀의 드롭다운에서 컬럼을 고른 뒤, "
               f"<b>이 셀만 다시 실행</b>하시면 됩니다.", "warn")
        return None

    # 다시 실행할 때 이전 결과가 섞이지 않도록 비웁니다.
    REPORT_PARTS.clear()
    RESULT_FILES.clear()
    REPORT_TOC.clear()

    # 2단계에서 분석 종류를 바꿨을 수도 있으니 필요한 라이브러리를 다시 확인합니다.
    global HAS_PROPHET, HAS_XGB
    for pkg, pip_name, label in MODULE_DEPS.get(MODULE_ID, []):
        okp = _have(pkg) or _ensure(pkg, pip_name, label)
        if pkg == "prophet":
            HAS_PROPHET = okp
        elif pkg == "xgboost":
            HAS_XGB = okp

    run_module()
    zip_name = save_report(download=download)

    if not FAILED and REPORT_PARTS:
        display(HTML(f"""
        <div style="font-family:{FONT};background:#eaf6ea;border:1px solid #0ca30c66;
                    border-radius:12px;padding:16px 18px;margin-top:14px">
          <div style="font-weight:700;color:#0a7a0a;font-size:15px">✅ 분석이 끝났습니다</div>
          <div style="font-size:13.5px;color:{INK};margin-top:6px;line-height:1.7">
            · 위로 스크롤하면 그래프와 <b>“이 결과를 이렇게 읽으세요”</b> 해석을 볼 수 있습니다.<br>
            · <b>다른 컬럼으로 다시 해보고 싶다면</b> 2단계 드롭다운을 바꾸고 <b>이 셀만</b> 다시 실행하세요.
              (파일을 다시 올릴 필요 없습니다)<br>
            · 내려받은 ZIP 안의 <b>HTML 리포트</b>는 동료에게 그대로 보내도 똑같이 열립니다.
          </div></div>"""))
    return zip_name

