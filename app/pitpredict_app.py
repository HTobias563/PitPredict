#!/usr/bin/env python3
"""PitPredict Streamlit Web Application - F1 Race Outcome Prediction Suite"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import sys
import os
import json
import traceback
from typing import Dict, List, Optional
from datetime import datetime, date
import yaml

app_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_root)
sys.path.append(project_root)

# joblib unpickles classes by looking in __main__; inject them so deserialization works
try:
    from src.pitpredict.models.final_position_predict import (
        FinalPositionPredictionConfig,
        FinalPositionPredictor,
        FinalPositionFeatureEngineer,
    )
    _main = sys.modules["__main__"]
    _main.FinalPositionPredictionConfig = FinalPositionPredictionConfig
    _main.FinalPositionPredictor = FinalPositionPredictor
    _main.FinalPositionFeatureEngineer = FinalPositionFeatureEngineer
except Exception:
    pass

st.set_page_config(
    page_title="PitPredict",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #E10600;
        text-align: center;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .race-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border-left: 5px solid #E10600;
        color: white;
        margin-bottom: 1rem;
    }
    .race-card h2 { color: #E10600; margin: 0 0 0.3rem 0; font-size: 1.6rem; }
    .race-card p  { margin: 0.2rem 0; color: #ccc; }
    .podium-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #1a1a1a; }
    .podium-2 { background: linear-gradient(135deg, #C0C0C0, #A9A9A9); color: #1a1a1a; }
    .podium-3 { background: linear-gradient(135deg, #CD7F32, #8B4513); color: white; }
    .podium-card {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.3rem;
    }
    .podium-card h3 { margin: 0 0 0.3rem 0; font-size: 1rem; }
    .podium-card h2 { margin: 0 0 0.3rem 0; font-size: 1.4rem; }
    .podium-card p  { margin: 0.15rem 0; font-size: 0.85rem; }
    .standings-table th { background: #E10600 !important; color: white !important; }
    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #E10600;
        margin-bottom: 0.7rem;
    }
    .metric-box h4 { margin: 0 0 0.2rem 0; color: #E10600; font-size: 0.85rem; text-transform: uppercase; }
    .metric-box p  { margin: 0; font-size: 1.6rem; font-weight: 700; color: #1a1a1a; }
    .metric-box small { color: #666; font-size: 0.8rem; }
    .rookie-badge {
        background: #fff3cd;
        color: #856404;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

DRIVERS_2025: Dict[str, Dict] = {
    "VER": {"name": "Max Verstappen",      "team": "Red Bull Racing",  "number": 1},
    "LAW": {"name": "Liam Lawson",         "team": "Red Bull Racing",  "number": 30,  "rookie": True},
    "NOR": {"name": "Lando Norris",        "team": "McLaren",          "number": 4},
    "PIA": {"name": "Oscar Piastri",       "team": "McLaren",          "number": 81},
    "HAM": {"name": "Lewis Hamilton",      "team": "Ferrari",          "number": 44},
    "LEC": {"name": "Charles Leclerc",     "team": "Ferrari",          "number": 16},
    "RUS": {"name": "George Russell",      "team": "Mercedes",         "number": 63},
    "ANT": {"name": "Kimi Antonelli",      "team": "Mercedes",         "number": 12,  "rookie": True},
    "ALO": {"name": "Fernando Alonso",     "team": "Aston Martin",     "number": 14},
    "STR": {"name": "Lance Stroll",        "team": "Aston Martin",     "number": 18},
    "GAS": {"name": "Pierre Gasly",        "team": "Alpine",           "number": 10},
    "DOO": {"name": "Jack Doohan",         "team": "Alpine",           "number": 7,   "rookie": True},
    "SAI": {"name": "Carlos Sainz",        "team": "Williams",         "number": 55},
    "ALB": {"name": "Alexander Albon",     "team": "Williams",         "number": 23},
    "TSU": {"name": "Yuki Tsunoda",        "team": "Racing Bulls",     "number": 22},
    "HAD": {"name": "Isack Hadjar",        "team": "Racing Bulls",     "number": 6,   "rookie": True},
    "MAG": {"name": "Kevin Magnussen",     "team": "Haas F1 Team",     "number": 20},
    "BEA": {"name": "Oliver Bearman",      "team": "Haas F1 Team",     "number": 87},
    "HUL": {"name": "Nico Hülkenberg",    "team": "Kick Sauber",      "number": 27},
    "BOR": {"name": "Gabriel Bortoleto",   "team": "Kick Sauber",      "number": 5,   "rookie": True},
}

TEAM_COLORS: Dict[str, str] = {
    "Red Bull Racing": "#3671C6",
    "McLaren":         "#FF8000",
    "Ferrari":         "#E8002D",
    "Mercedes":        "#27F4D2",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Williams":        "#64C4FF",
    "Racing Bulls":    "#6692FF",
    "Haas F1 Team":    "#B6BABD",
    "Kick Sauber":     "#52E252",
}

# Drivers with 2024 training data (rookies in 2025 may be skipped by model)
KNOWN_TO_MODEL = {
    "VER", "NOR", "LEC", "PIA", "RUS", "HAM", "SAI", "PER", "ALO", "STR",
    "GAS", "OCO", "TSU", "LAW", "ALB", "COL", "HUL", "BOT", "ZHO",
    "BEA", "RIC", "MAG", "DOO",
}

CALENDAR_2025 = [
    {"round": 1,  "name": "Australian Grand Prix",     "circuit": "Albert Park",               "country": "Australien",      "date": "2025-03-16", "track_type": "default"},
    {"round": 2,  "name": "Chinese Grand Prix",        "circuit": "Shanghai",                  "country": "China",           "date": "2025-03-23", "track_type": "default"},
    {"round": 3,  "name": "Japanese Grand Prix",       "circuit": "Suzuka",                    "country": "Japan",           "date": "2025-04-06", "track_type": "default"},
    {"round": 4,  "name": "Bahrain Grand Prix",        "circuit": "Bahrain International",     "country": "Bahrain",         "date": "2025-04-13", "track_type": "default"},
    {"round": 5,  "name": "Saudi Arabian Grand Prix",  "circuit": "Jeddah Corniche",           "country": "Saudi-Arabien",   "date": "2025-04-20", "track_type": "default"},
    {"round": 6,  "name": "Miami Grand Prix",          "circuit": "Miami International",       "country": "USA",             "date": "2025-05-04", "track_type": "default"},
    {"round": 7,  "name": "Emilia Romagna Grand Prix", "circuit": "Imola",                     "country": "Italien",         "date": "2025-05-18", "track_type": "default"},
    {"round": 8,  "name": "Monaco Grand Prix",         "circuit": "Circuit de Monaco",         "country": "Monaco",          "date": "2025-05-25", "track_type": "monaco"},
    {"round": 9,  "name": "Spanish Grand Prix",        "circuit": "Barcelona-Catalunya",       "country": "Spanien",         "date": "2025-06-01", "track_type": "default"},
    {"round": 10, "name": "Canadian Grand Prix",       "circuit": "Gilles Villeneuve",         "country": "Kanada",          "date": "2025-06-15", "track_type": "default"},
    {"round": 11, "name": "Austrian Grand Prix",       "circuit": "Red Bull Ring",             "country": "Österreich",     "date": "2025-06-29", "track_type": "default"},
    {"round": 12, "name": "British Grand Prix",        "circuit": "Silverstone",               "country": "Großbritannien", "date": "2025-07-06", "track_type": "silverstone"},
    {"round": 13, "name": "Belgian Grand Prix",        "circuit": "Spa-Francorchamps",         "country": "Belgien",         "date": "2025-07-27", "track_type": "spa"},
    {"round": 14, "name": "Hungarian Grand Prix",      "circuit": "Hungaroring",               "country": "Ungarn",          "date": "2025-08-03", "track_type": "default"},
    {"round": 15, "name": "Dutch Grand Prix",          "circuit": "Zandvoort",                 "country": "Niederlande",     "date": "2025-08-31", "track_type": "netherlands"},
    {"round": 16, "name": "Italian Grand Prix",        "circuit": "Monza",                     "country": "Italien",         "date": "2025-09-07", "track_type": "default"},
    {"round": 17, "name": "Azerbaijan Grand Prix",     "circuit": "Baku City Circuit",         "country": "Aserbaidschan",   "date": "2025-09-21", "track_type": "default"},
    {"round": 18, "name": "Singapore Grand Prix",      "circuit": "Marina Bay",                "country": "Singapur",        "date": "2025-10-05", "track_type": "default"},
    {"round": 19, "name": "United States Grand Prix",  "circuit": "Circuit of the Americas",   "country": "USA",             "date": "2025-10-19", "track_type": "default"},
    {"round": 20, "name": "Mexico City Grand Prix",    "circuit": "Hermanos Rodriguez",        "country": "Mexiko",          "date": "2025-10-26", "track_type": "default"},
    {"round": 21, "name": "São Paulo Grand Prix",     "circuit": "Interlagos",                "country": "Brasilien",       "date": "2025-11-09", "track_type": "default"},
    {"round": 22, "name": "Las Vegas Grand Prix",      "circuit": "Las Vegas Strip Circuit",   "country": "USA",             "date": "2025-11-22", "track_type": "default"},
    {"round": 23, "name": "Qatar Grand Prix",          "circuit": "Lusail",                    "country": "Katar",           "date": "2025-11-30", "track_type": "default"},
    {"round": 24, "name": "Abu Dhabi Grand Prix",      "circuit": "Yas Marina",                "country": "VAE",             "date": "2025-12-07", "track_type": "default"},
]

TRACK_TYPE_LABELS = {
    "netherlands": "Niederlande (Zandvoort) – schwer zu überholen",
    "monaco":      "Monaco – Straßenkurs, fast unmöglich zu überholen",
    "spa":         "Spa-Francorchamps – leicht zu überholen",
    "silverstone": "Silverstone – ausgeglichene Strecke",
    "default":     "Standard-Strecke",
}

# ── Cached API helpers ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_driver_standings(year: int) -> Optional[List[Dict]]:
    try:
        url = f"https://api.jolpi.ca/ergast/f1/{year}/driverStandings.json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        lists = r.json()["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return None
        return lists[0]["DriverStandings"]
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_constructor_standings(year: int) -> Optional[List[Dict]]:
    try:
        url = f"https://api.jolpi.ca/ergast/f1/{year}/constructorStandings.json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        lists = r.json()["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return None
        return lists[0]["ConstructorStandings"]
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_race_calendar(year: int) -> Optional[List[Dict]]:
    try:
        url = f"https://api.jolpi.ca/ergast/f1/{year}.json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return r.json()["MRData"]["RaceTable"]["Races"]
    except Exception:
        return None


def find_next_race(races: List[Dict]) -> Optional[Dict]:
    today = date.today()
    for race in races:
        race_date = datetime.strptime(race["date"], "%Y-%m-%d").date()
        if race_date >= today:
            return race
    return None


# ── App ───────────────────────────────────────────────────────────────────────

class PitPredictApp:
    def __init__(self):
        self.project_root = project_root
        if "config" not in st.session_state:
            st.session_state.config = self._load_config()

    def _load_config(self) -> Dict:
        try:
            with open(os.path.join(self.project_root, "config.yaml")) as f:
                return yaml.safe_load(f)
        except Exception:
            return {"season": 2024}

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        st.markdown('<h1 class="main-header">PitPredict</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">F1 Race Outcome Prediction · Saison 2025</p>', unsafe_allow_html=True)

        page = st.sidebar.radio(
            "Navigation",
            ["Übersicht", "Vorhersage", "Evaluation"],
            label_visibility="collapsed",
        )
        st.sidebar.markdown("---")
        st.sidebar.markdown("**PitPredict** · F1 ML-System  \nModell: Gradient Boosting  \nTrainingsdaten: Saison 2024")

        if page == "Übersicht":
            self.show_overview()
        elif page == "Vorhersage":
            self.show_prediction()
        else:
            self.show_evaluation()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1: ÜBERSICHT
    # ─────────────────────────────────────────────────────────────────────────

    def show_overview(self):
        st.header("Übersicht · Saison 2025")

        # ── Nächstes / letztes Rennen ─────────────────────────────────────
        with st.spinner("Lade Renndaten..."):
            api_races = fetch_race_calendar(2025)

        st.subheader("Rennkalender 2025")
        if api_races:
            next_race = find_next_race(api_races)
            if next_race:
                self._render_next_race_card(next_race)
            else:
                st.info("Die Saison 2025 ist beendet. Alle Rennen sind abgeschlossen.")
                self._render_season_summary_card()
        else:
            # Fallback: hardcoded calendar
            today = date.today()
            next_race_local = None
            for r in CALENDAR_2025:
                if datetime.strptime(r["date"], "%Y-%m-%d").date() >= today:
                    next_race_local = r
                    break
            if next_race_local:
                self._render_next_race_card_local(next_race_local)
            else:
                self._render_season_summary_card()

        st.markdown("---")

        # ── Standings ─────────────────────────────────────────────────────
        col_d, col_c = st.columns(2)

        with col_d:
            st.subheader("Fahrer-WM")
            with st.spinner("Lade Fahrer-Standings..."):
                driver_standings = fetch_driver_standings(2025)
            if driver_standings:
                self._render_driver_standings(driver_standings)
            else:
                st.warning("Fahrer-Standings konnten nicht geladen werden.")

        with col_c:
            st.subheader("Konstrukteurs-WM")
            with st.spinner("Lade Konstrukteurs-Standings..."):
                constructor_standings = fetch_constructor_standings(2025)
            if constructor_standings:
                self._render_constructor_standings(constructor_standings)
            else:
                st.warning("Konstrukteurs-Standings konnten nicht geladen werden.")

    def _render_next_race_card(self, race: Dict):
        race_date = datetime.strptime(race["date"], "%Y-%m-%d")
        days_until = (race_date.date() - date.today()).days
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})

        label = f"in {days_until} Tagen" if days_until > 0 else "Heute!"
        st.markdown(f"""
        <div class="race-card">
            <h2>{race.get('raceName', 'Unbekannt')}</h2>
            <p>📍 {circuit.get('circuitName', '')} · {location.get('locality', '')}, {location.get('country', '')}</p>
            <p>📅 {race_date.strftime('%d. %B %Y')} &nbsp;|&nbsp; {label}</p>
            <p>Runde {race.get('round', '?')} / 24</p>
        </div>
        """, unsafe_allow_html=True)

    def _render_next_race_card_local(self, race: Dict):
        race_date = datetime.strptime(race["date"], "%Y-%m-%d")
        days_until = (race_date.date() - date.today()).days
        label = f"in {days_until} Tagen" if days_until > 0 else "Heute!"
        st.markdown(f"""
        <div class="race-card">
            <h2>{race['name']}</h2>
            <p>📍 {race['circuit']} · {race['country']}</p>
            <p>📅 {race_date.strftime('%d. %B %Y')} &nbsp;|&nbsp; {label}</p>
            <p>Runde {race['round']} / 24</p>
        </div>
        """, unsafe_allow_html=True)

    def _render_season_summary_card(self):
        st.markdown("""
        <div class="race-card">
            <h2>Saison 2025 – Abgeschlossen</h2>
            <p>📅 24 Rennen · März bis Dezember 2025</p>
            <p>Die vollständigen Ergebnisse sind in den Standings unten zu sehen.</p>
        </div>
        """, unsafe_allow_html=True)

    def _render_driver_standings(self, standings: List[Dict]):
        rows = []
        for entry in standings[:20]:
            driver = entry.get("Driver", {})
            constructors = entry.get("Constructors", [{}])
            team = constructors[0].get("name", "") if constructors else ""
            rows.append({
                "Pos": int(entry.get("position", 0)),
                "Fahrer": f"{driver.get('givenName', '')} {driver.get('familyName', '')}",
                "Team": team,
                "Punkte": int(float(entry.get("points", 0))),
                "Siege": int(entry.get("wins", 0)),
            })

        df = pd.DataFrame(rows)

        # Color bars by team
        team_col = df["Team"].map(lambda t: TEAM_COLORS.get(t, "#888"))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Punkte"],
            y=df["Fahrer"],
            orientation="h",
            marker_color=team_col.tolist(),
            text=df["Punkte"],
            textposition="outside",
            hovertemplate="%{y}<br>%{x} Punkte<extra></extra>",
        ))
        fig.update_layout(
            height=500,
            margin=dict(l=0, r=40, t=10, b=10),
            xaxis_title="Punkte",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table below chart
        st.dataframe(
            df.style.background_gradient(subset=["Punkte"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )

    def _render_constructor_standings(self, standings: List[Dict]):
        rows = []
        for entry in standings[:10]:
            constructor = entry.get("Constructor", {})
            name = constructor.get("name", "")
            rows.append({
                "Pos": int(entry.get("position", 0)),
                "Team": name,
                "Punkte": int(float(entry.get("points", 0))),
                "Siege": int(entry.get("wins", 0)),
            })

        df = pd.DataFrame(rows)
        team_col = df["Team"].map(lambda t: TEAM_COLORS.get(t, "#888"))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Punkte"],
            y=df["Team"],
            orientation="h",
            marker_color=team_col.tolist(),
            text=df["Punkte"],
            textposition="outside",
            hovertemplate="%{y}<br>%{x} Punkte<extra></extra>",
        ))
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=40, t=10, b=10),
            xaxis_title="Punkte",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df.style.background_gradient(subset=["Punkte"], cmap="Oranges"),
            use_container_width=True,
            hide_index=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: VORHERSAGE
    # ─────────────────────────────────────────────────────────────────────────

    def show_prediction(self):
        st.header("Vorhersage · 2025 Saison")
        st.markdown("Wähle ein Rennen, passe die Startaufstellung an und generiere eine Vorhersage.")

        # Load predictor
        try:
            from src.pitpredict.models.future_position_predict import FutureRacePredictor
            if "future_predictor" not in st.session_state:
                with st.spinner("Lade Modell..."):
                    st.session_state.future_predictor = FutureRacePredictor()
        except Exception as e:
            st.error(f"Modell konnte nicht geladen werden: {e}")
            return

        # ── Race selector ─────────────────────────────────────────────────
        race_names = [f"R{r['round']:02d} · {r['name']}" for r in CALENDAR_2025]
        selected_idx = st.selectbox(
            "Rennen auswählen",
            range(len(CALENDAR_2025)),
            format_func=lambda i: race_names[i],
        )
        selected_race = CALENDAR_2025[selected_idx]

        race_date = datetime.strptime(selected_race["date"], "%Y-%m-%d")

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Strecke", selected_race["circuit"])
        col_info2.metric("Land", selected_race["country"])
        col_info3.metric("Datum", race_date.strftime("%d. %B %Y"))

        # Track type (auto from calendar, but overrideable)
        track_type = st.selectbox(
            "Streckentyp",
            options=list(TRACK_TYPE_LABELS.keys()),
            index=list(TRACK_TYPE_LABELS.keys()).index(selected_race["track_type"]),
            format_func=lambda k: TRACK_TYPE_LABELS[k],
        )

        st.markdown("---")

        # ── Grid customizer ───────────────────────────────────────────────
        st.subheader("Startaufstellung")

        rookies = {code for code, info in DRIVERS_2025.items() if info.get("rookie")}
        unknown = {code for code in DRIVERS_2025 if code not in KNOWN_TO_MODEL}
        if unknown:
            st.info(
                f"Rookies ohne 2024-Trainingsdaten: **{', '.join(sorted(unknown))}** "
                "– werden im Modell möglicherweise übersprungen."
            )

        # Default grid: VER 1, NOR 2, LEC 3, ...
        default_positions = {code: i + 1 for i, code in enumerate(DRIVERS_2025)}

        grid_mode = st.radio(
            "Startaufstellung",
            ["Standard-Grid verwenden", "Positionen anpassen"],
            horizontal=True,
        )

        if grid_mode == "Positionen anpassen":
            grid_positions = self._grid_customizer(default_positions)
        else:
            grid_positions = default_positions
            self._show_grid_preview(default_positions)

        st.markdown("---")

        # ── Predict button ────────────────────────────────────────────────
        if st.button("Vorhersage generieren", type="primary", use_container_width=True):
            with st.spinner("Berechne Vorhersage..."):
                try:
                    predictions = st.session_state.future_predictor.predict_future_race(
                        race_name=selected_race["name"],
                        grid_positions=grid_positions,
                        track_type=track_type,
                        season=2025,
                    )
                    if predictions is not None and len(predictions) > 0:
                        st.session_state.last_predictions = predictions
                        st.session_state.last_race_name = selected_race["name"]
                        st.success(f"Vorhersage für **{selected_race['name']}** erstellt.")
                    else:
                        st.error("Keine Ergebnisse – überprüfe ob das Modell geladen ist.")
                except Exception as e:
                    st.error(f"Fehler bei der Vorhersage: {e}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

        # ── Results ───────────────────────────────────────────────────────
        if "last_predictions" in st.session_state and st.session_state.last_predictions is not None:
            self._display_predictions(
                st.session_state.last_predictions,
                st.session_state.get("last_race_name", "Rennen"),
            )

    def _grid_customizer(self, default_positions: Dict[str, int]) -> Dict[str, int]:
        st.markdown("Vergib Startplätze (1–20). Doppelte Positionen erzeugen eine Warnung.")
        custom = {}
        cols = st.columns(4)
        for i, (code, info) in enumerate(DRIVERS_2025.items()):
            rookie_tag = " 🆕" if info.get("rookie") else ""
            label = f"#{info['number']} {info['name']}{rookie_tag}"
            with cols[i % 4]:
                custom[code] = st.number_input(
                    label,
                    min_value=1, max_value=20,
                    value=default_positions[code],
                    key=f"grid_{code}",
                )
        used = list(custom.values())
        if len(set(used)) != len(used):
            st.warning("Achtung: Zwei Fahrer haben dieselbe Startposition.")
        return custom

    def _show_grid_preview(self, grid_positions: Dict[str, int]):
        sorted_grid = sorted(grid_positions.items(), key=lambda x: x[1])
        cols = st.columns(4)
        for i, (code, pos) in enumerate(sorted_grid):
            info = DRIVERS_2025.get(code, {})
            team = info.get("team", "")
            color = TEAM_COLORS.get(team, "#888")
            with cols[i % 4]:
                st.markdown(
                    f"P{pos} &nbsp; <span style='color:{color}; font-weight:700'>{code}</span> "
                    f"<span style='color:#888; font-size:0.8rem'>{info.get('name','')}</span>",
                    unsafe_allow_html=True,
                )

    def _display_predictions(self, predictions: pd.DataFrame, race_name: str):
        st.markdown("---")
        st.subheader(f"Ergebnis: {race_name}")

        # Sort by predicted position
        pos_col = "predicted_position_rounded" if "predicted_position_rounded" in predictions.columns else "predicted_final_position"
        pred_sorted = predictions.sort_values(pos_col).reset_index(drop=True)

        # ── Podium ────────────────────────────────────────────────────────
        st.markdown("#### Podium")
        podium_cols = st.columns(3)
        for i in range(min(3, len(pred_sorted))):
            row = pred_sorted.iloc[i]
            driver_code = row.get("driver", "???")
            driver_info = DRIVERS_2025.get(driver_code, {})
            team = row.get("team", driver_info.get("team", ""))
            grid_pos = int(row.get("grid_position", 0))
            dnf_pct = row.get("dnf_risk", 0.0) * 100
            css = f"podium-{i+1}"
            position_label = ["1st", "2nd", "3rd"][i]
            with podium_cols[i]:
                st.markdown(f"""
                <div class="podium-card {css}">
                    <h3>P{i+1} · {position_label}</h3>
                    <h2>{driver_code}</h2>
                    <p>{driver_info.get('name', driver_code)}</p>
                    <p>{team}</p>
                    <p>Start: P{grid_pos} &nbsp;|&nbsp; DNF-Risiko: {dnf_pct:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Full table ────────────────────────────────────────────────────
        display_rows = []
        for rank, (_, row) in enumerate(pred_sorted.iterrows(), 1):
            code = row.get("driver", "???")
            info = DRIVERS_2025.get(code, {})
            pred_pos = int(row.get(pos_col, rank))
            grid = int(row.get("grid_position", 0))
            change = grid - pred_pos
            display_rows.append({
                "Pos": rank,
                "Fahrer": info.get("name", code),
                "Code": code,
                "Team": row.get("team", info.get("team", "")),
                "Start": grid,
                "Vorhergesagt": pred_pos,
                "Δ Positionen": change,
                "DNF-Risiko": f"{row.get('dnf_risk', 0)*100:.0f}%",
            })

        df = pd.DataFrame(display_rows)

        def color_delta(val):
            if val > 0:
                return "color: #16a34a; font-weight: 600"
            elif val < 0:
                return "color: #dc2626; font-weight: 600"
            return ""

        st.dataframe(
            df.style.applymap(color_delta, subset=["Δ Positionen"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── Charts ────────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Start vs. Vorhersage**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Start"], y=df["Vorhergesagt"],
                mode="markers+text",
                text=df["Code"],
                textposition="middle right",
                marker=dict(size=10, color="#E10600"),
                hovertemplate="%{text}<br>Start P%{x} → Vorher. P%{y}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[1, 20], y=[1, 20],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                showlegend=False,
            ))
            fig.update_layout(
                height=400,
                xaxis_title="Startposition",
                yaxis_title="Vorhergesagte Position",
                yaxis=dict(autorange="reversed"),
                xaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=20, b=40),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**Positionsgewinne / -verluste**")
            colors = ["#16a34a" if v > 0 else "#dc2626" if v < 0 else "#888"
                      for v in df["Δ Positionen"]]
            fig2 = go.Figure(go.Bar(
                x=df["Code"],
                y=df["Δ Positionen"],
                marker_color=colors,
                text=df["Δ Positionen"].apply(lambda v: f"+{v}" if v > 0 else str(v)),
                textposition="outside",
                hovertemplate="%{x}: %{y} Positionen<extra></extra>",
            ))
            fig2.update_layout(
                height=400,
                yaxis_title="Δ Positionen (+ = besser)",
                margin=dict(l=0, r=0, t=20, b=40),
                plot_bgcolor="white",
                yaxis=dict(zeroline=True, zerolinecolor="#aaa"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "CSV herunterladen",
            data=csv,
            file_name=f"pitpredict_{race_name.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3: EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    def show_evaluation(self):
        st.header("Evaluation · Modellleistung")
        st.markdown(
            "Bewertung des Gradient-Boosting-Regressors auf Kreuzvalidierungs- und Holdout-Daten "
            "(Saison 2024, Runden 1–24)."
        )

        metrics_dir = os.path.join(self.project_root, "artifacts", "metrics")
        cv_path = os.path.join(metrics_dir, "final_position_cv_report.json")
        holdout_path = os.path.join(metrics_dir, "final_position_holdout_report.json")

        cv_data = self._load_json(cv_path)
        holdout_data = self._load_json(holdout_path)

        # ── Summary metrics ───────────────────────────────────────────────
        st.subheader("Kennzahlen auf einen Blick")
        c1, c2, c3, c4 = st.columns(4)

        if cv_data:
            self._metric_box(c1, "Ø MAE (CV)",         f"{cv_data.get('overall_mae', 0):.2f}",     "Positionen")
            self._metric_box(c2, "R² (CV)",             f"{cv_data.get('overall_r2', 0):.3f}",      "Bestimmtheitsmaß")
            self._metric_box(c3, "Podium-Genauigkeit",  f"{cv_data.get('overall_podium_accuracy', 0)*100:.1f}%", "Top-3-Trefferquote")
            self._metric_box(c4, "Punkte-Genauigkeit",  f"{cv_data.get('overall_points_accuracy', 0)*100:.1f}%", "Top-10-Trefferquote")
        else:
            st.warning("CV-Report nicht gefunden.")

        st.markdown("---")

        # ── CV vs Holdout side by side ────────────────────────────────────
        col_cv, col_ho = st.columns(2)

        with col_cv:
            st.subheader("Kreuzvalidierung (5-Fold)")
            if cv_data:
                folds = cv_data.get("fold_scores", [])
                if folds:
                    fold_df = pd.DataFrame(folds)
                    fold_df["Fold"] = fold_df["fold"].apply(lambda f: f"Fold {f}")

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="MAE",
                        x=fold_df["Fold"],
                        y=fold_df["mae"],
                        marker_color="#E10600",
                        text=fold_df["mae"].round(2),
                        textposition="outside",
                    ))
                    fig.add_hline(
                        y=cv_data.get("overall_mae", 0),
                        line_dash="dash", line_color="gray",
                        annotation_text=f"Ø {cv_data.get('overall_mae',0):.2f}",
                        annotation_position="top right",
                    )
                    fig.update_layout(
                        height=300,
                        yaxis_title="MAE (Positionen)",
                        margin=dict(l=0, r=0, t=20, b=10),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Fold details
                    fold_display = fold_df[["Fold", "mae", "rmse", "r2", "podium_accuracy", "points_accuracy"]].copy()
                    fold_display.columns = ["Fold", "MAE", "RMSE", "R²", "Podium", "Punkte"]
                    fold_display[["MAE", "RMSE"]] = fold_display[["MAE", "RMSE"]].round(2)
                    fold_display["R²"] = fold_display["R²"].round(3)
                    fold_display["Podium"] = (fold_display["Podium"] * 100).round(1).astype(str) + "%"
                    fold_display["Punkte"] = (fold_display["Punkte"] * 100).round(1).astype(str) + "%"
                    st.dataframe(fold_display, use_container_width=True, hide_index=True)

        with col_ho:
            st.subheader("Holdout-Test (Runden 21–24)")
            if holdout_data:
                h_mae = holdout_data.get("holdout_mae", holdout_data.get("overall_mae", 0))
                h_rmse = holdout_data.get("holdout_rmse", holdout_data.get("overall_rmse", 0))
                h_r2 = holdout_data.get("holdout_r2", holdout_data.get("overall_r2", 0))
                h_pod = holdout_data.get("holdout_podium_accuracy", holdout_data.get("overall_podium_accuracy", 0))
                h_pts = holdout_data.get("holdout_points_accuracy", holdout_data.get("overall_points_accuracy", 0))

                cv_mae = cv_data.get("overall_mae", 0) if cv_data else 0

                metrics = {
                    "MAE": (h_mae, cv_mae, "Positionen"),
                    "RMSE": (h_rmse, cv_data.get("overall_rmse", 0) if cv_data else 0, "Positionen"),
                    "R²": (h_r2, cv_data.get("overall_r2", 0) if cv_data else 0, ""),
                    "Podium-Genauigkeit": (h_pod, cv_data.get("overall_podium_accuracy", 0) if cv_data else 0, ""),
                    "Punkte-Genauigkeit": (h_pts, cv_data.get("overall_points_accuracy", 0) if cv_data else 0, ""),
                }

                for label, (val, cv_val, unit) in metrics.items():
                    is_error = label in ("MAE", "RMSE")
                    delta = val - cv_val
                    delta_str = f"{'+' if delta > 0 else ''}{delta:.3f}" if cv_val else None
                    if unit:
                        display = f"{val:.2f} {unit}"
                    elif label == "R²":
                        display = f"{val:.3f}"
                    else:
                        display = f"{val*100:.1f}%"

                    st.metric(
                        label=label,
                        value=display,
                        delta=delta_str,
                        delta_color="inverse" if is_error else "normal",
                    )
            else:
                st.warning("Holdout-Report nicht gefunden.")

        st.markdown("---")

        # ── Feature Importance ────────────────────────────────────────────
        st.subheader("Feature-Wichtigkeit (Top 15)")
        if cv_data and "feature_importance" in cv_data:
            fi = cv_data["feature_importance"]
            fi_df = (
                pd.DataFrame(list(fi.items()), columns=["Feature", "Wichtigkeit"])
                .sort_values("Wichtigkeit", ascending=False)
                .head(15)
            )
            fig = px.bar(
                fi_df,
                x="Wichtigkeit",
                y="Feature",
                orientation="h",
                color="Wichtigkeit",
                color_continuous_scale="Reds",
            )
            fig.update_layout(
                height=450,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=10, b=10),
                coloraxis_showscale=False,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Alle Features anzeigen"):
                all_fi = (
                    pd.DataFrame(list(fi.items()), columns=["Feature", "Wichtigkeit"])
                    .sort_values("Wichtigkeit", ascending=False)
                )
                all_fi["Wichtigkeit (%)"] = (all_fi["Wichtigkeit"] * 100).round(2)
                st.dataframe(all_fi[["Feature", "Wichtigkeit (%)"]], use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Model files status ────────────────────────────────────────────
        st.subheader("Modell-Dateien")
        models_dir = os.path.join(self.project_root, "artifacts", "models")
        files = [
            ("final_position_predictor.pkl", "Final Position Model"),
            ("dnf_pipeline_calibrated.pkl",  "DNF-Modell (kalibriert)"),
            ("pitstop_predictor_calibrated.pkl", "Pit Stop-Modell"),
        ]
        for fname, label in files:
            exists = os.path.exists(os.path.join(models_dir, fname))
            col_l, col_r = st.columns([4, 1])
            col_l.write(f"**{label}** `{fname}`")
            if exists:
                col_r.success("Vorhanden")
            else:
                col_r.error("Fehlt")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_json(self, path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def _metric_box(self, container, title: str, value: str, subtitle: str = ""):
        container.markdown(f"""
        <div class="metric-box">
            <h4>{title}</h4>
            <p>{value}</p>
            <small>{subtitle}</small>
        </div>
        """, unsafe_allow_html=True)


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    app = PitPredictApp()
    app.run()


if __name__ == "__main__":
    main()
