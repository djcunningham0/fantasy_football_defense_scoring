import streamlit as st
import pandas as pd
import json
from io import StringIO
import plotly.figure_factory as ff
from itertools import cycle
from typing import List, Tuple
from collections import defaultdict


st.title("Fantasy Football Defense Settings")

###########
# Load data
###########

df = pd.read_csv("./data/data.csv")
d_stats = pd.read_csv("./data/defense_stats_2021.csv")

with open("./data/current_settings.json", "r") as f:
    CURRENT_SETTINGS = json.load(f)

with open("./data/stat_category_map.json", "r") as f:
    STAT_MAP = json.load(f)

with open("./data/managers.json", "r") as f:
    MANAGERS = json.load(f)

with open("./data/teams.json", "r") as f:
    TEAMS = json.load(f)


def open_matchup_history() -> pd.DataFrame:
    """Read json and consolidate into single dict rather than split by year"""
    with open("./data/matchup_history.json", "r") as f:
        matchup_history = json.load(f)
    data = {}
    i = 0
    for year in matchup_history:
        for x in year:
            data[i] = x
            i += 1
    return pd.DataFrame(data).T


matchup_df = open_matchup_history()


############################
# choose settings in sidebar
############################

with st.sidebar:
    st.write("# Settings")

    # button to reset to default values
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = 0
    if st.button("reset to defaults"):
        st.session_state["run_id"] += 1

    with st.expander("import/export"):
        download_button = st.empty()
        st.write("---")
        upload_button = st.empty()

    CUSTOM_SETTINGS = {}
    INPUTS = {}

    basic_categories = [
        "Sack",
        "Interception",
        "Fumble Recovery",
        "Touchdown",
        "Kickoff and Punt Return Touchdowns",
        "Safety",
        "Block Kick",
        "Extra Point Returned",
    ]
    cats_basic = [k for k, v in STAT_MAP.items() if v in basic_categories]
    cats_points = [k for k, v in STAT_MAP.items() if v.startswith("Points Allowed")]
    cats_yards = [k for k, v in STAT_MAP.items() if v.startswith("Defensive Yards")]
    cats_other = [k for k in STAT_MAP.keys() if k not in [*cats_basic, *cats_points, *cats_yards]]

    def display_categories(keys: List[str]):
        st.write("---")
        cols = st.columns(2)
        iterator = cycle([0, 1])
        for k in keys:
            name = STAT_MAP[k]
            if name in ["Points Allowed", "Defensive Yards Allowed"]:
                continue
            col = cols[next(iterator)]
            name = format_names(name)
            INPUTS[k] = col.empty()
            CUSTOM_SETTINGS[k] = INPUTS[k].number_input(
                label=name,
                value=CURRENT_SETTINGS[k],
                step=1.0,
                key=f"{name}_{st.session_state['run_id']}"
            )

    def format_names(x):
        x = x.replace(" points", "")
        x = x.replace("Touchdown", "TD")
        x = x.replace("Kickoff and Punt", "Kick/Punt")
        x = x.replace("Defensive Yards", "Yards")
        x = x.replace("- Negative", "< 0")
        return x

    display_categories(cats_basic)
    display_categories(cats_other)
    display_categories(cats_points)
    display_categories(cats_yards)

    def get_download_data(d: dict) -> str:
        d = d.copy()
        for key in list(d.keys()):
            new_key = STAT_MAP[key]
            d[new_key] = d.pop(key)
        return json.dumps(d, indent=2)

    download_button.download_button(
        label="export settings to file",
        data=get_download_data(CUSTOM_SETTINGS),
        file_name="defense_settings.json",
        mime="application/json",
    )

    uploaded_file = upload_button.file_uploader("upload from file", accept_multiple_files=False)
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        uploaded_data = json.loads(string_data)
        for key in list(uploaded_data.keys()):
            new_key = [k for k, v in STAT_MAP.items() if v == key][0]
            uploaded_data[new_key] = uploaded_data.pop(key)
        for k, val in uploaded_data.items():
            name = STAT_MAP[k]
            name = format_names(name)
            CUSTOM_SETTINGS[k] = INPUTS[k].number_input(
                label=name,
                value=float(val),
                step=1.0,
                key=f"{name}_{st.session_state['run_id']}_from_file"
            )


######################
# calculate new scores
######################

def calculate_defense_score(row: pd.Series, settings: dict):
    score = 0
    for key, val in settings.items():
        score += (row[key] * val)
    return score


def choose_winner(row: pd.Series, score_1_col: str, score_2_col: str) -> int:
    if row[score_1_col] > row[score_2_col]:
        return 1  # team 1 wins
    elif row[score_1_col] < row[score_2_col]:
        return 2  # team 2 wins
    else:
        return 0  # tie


d_stats["current defense score"] = d_stats.apply(calculate_defense_score, settings=CURRENT_SETTINGS, axis=1)
d_stats["new defense score"] = d_stats.apply(calculate_defense_score, settings=CUSTOM_SETTINGS, axis=1)
df["current_defense_score"] = df.apply(calculate_defense_score, settings=CURRENT_SETTINGS, axis=1)
df["new_defense_score"] = df.apply(calculate_defense_score, settings=CUSTOM_SETTINGS, axis=1)

df_cols = ["team_key", "league_key", "week", "current_defense_score", "new_defense_score"]
df["week"] = df["week"].astype(str)
matchup_df["week"] = matchup_df["week"].astype(str)
matchup_df = (
    matchup_df
    .merge(
        df[df_cols].rename(columns={
            "team_key": "team_key_1",
            "current_defense_score": "team_1_current_defense_score",
            "new_defense_score": "team_1_new_defense_score",
        }),
        on=["team_key_1", "league_key", "week"],
        how="inner"
    )
    .merge(
        df[df_cols].rename(columns={
            "team_key": "team_key_2",
            "current_defense_score": "team_2_current_defense_score",
            "new_defense_score": "team_2_new_defense_score",
        }),
        on=["team_key_2", "league_key", "week"],
        how="inner"
    )
)
matchup_df["team_1_new_score"] = (
        matchup_df["team_1_score"]
        - matchup_df["team_1_current_defense_score"]
        + matchup_df["team_1_new_defense_score"]
)
matchup_df["team_2_new_score"] = (
        matchup_df["team_2_score"]
        - matchup_df["team_2_current_defense_score"]
        + matchup_df["team_2_new_defense_score"]
)
matchup_df["current_winner"] = matchup_df.apply(
    choose_winner,
    score_1_col="team_1_score",
    score_2_col="team_2_score",
    axis=1
)
matchup_df["new_winner"] = matchup_df.apply(
    choose_winner,
    score_1_col="team_1_new_score",
    score_2_col="team_2_new_score",
    axis=1
)
matchup_df["different_result"] = matchup_df["current_winner"] != matchup_df["new_winner"]


#################
# results summary
#################

total_matchups = matchup_df.shape[0]
changed_matchups = matchup_df[matchup_df["different_result"]].shape[0]
current_avg = df["current_defense_score"].mean()
new_avg = df["new_defense_score"].mean()

st.write("## Summary")
cols = st.columns(3)
cols[0].metric("Current average DEF pts/week", value=f"{current_avg:.2f}",
               help="Average fantasy points scored by defenses per week using "
                    "our league's current settings. Only includes defenses that "
                    "were active in our league.")
cols[1].metric("New average DEF pts/week", value=f"{new_avg:.2f}",
               delta=f"{new_avg - current_avg:.2f}", delta_color="off",
               help="Average fantasy points scored by defenses per week using "
                    "the user supplied settings from the sidebar. Only includes "
                    "defenses that were active in our league.")
cols[2].metric("Matchup results changed", value=changed_matchups,
               delta=f"{changed_matchups/total_matchups:.2%}", delta_color="off",
               help="The number (and percentage) of matchup results that would "
                    "change due under the new defense scoring settings.")

hist_data = [df["current_defense_score"], df["new_defense_score"]]
group_labels = ["current settings", "new settings"]
fig = ff.create_distplot(hist_data, group_labels)
fig.update_layout(title_text="Weekly DEF points distribution")
st.write(fig)


######################
# defense stats impact
######################

st.write("### 2021 defense season scores")
d_stats["points change"] = d_stats["new defense score"] - d_stats["current defense score"]
d_stats["current rank"] = d_stats["current defense score"].rank(method="min", ascending=False).astype(int)
d_stats["new rank"] = d_stats["new defense score"].rank(method="min", ascending=False).astype(int)
d_stats["rank change"] = d_stats["current rank"] - d_stats["new rank"]
out = d_stats[["team", "current rank", "new rank", "rank change",
               "current defense score", "new defense score", "points change"]]
out = out.sort_values("new rank")
st.dataframe(out.set_index("team"))


##################
# standings impact
##################

st.write("### Standings impact")


class TeamRecord:
    def __init__(
            self,
            team_name: str,
            manager_name: str,
            wins: float,
            losses: float,
            ties: float,
            points: float,
    ):
        self.team_name = team_name
        self.manager_name = manager_name
        self.wins = int(wins)
        self.losses = int(losses)
        self.ties = int(ties)
        self.points = points

    def print_record(self):
        return f"{self.wins}-{self.losses}-{self.ties}"

    def __eq__(self, other):
        return self.wins == other.wins and self.points == other.points

    def __gt__(self, other):
        return self.wins > other.wins or (self.wins == other.wins and self.points > other.points)

    def __repr__(self):
        return f"{self.team_name}: {self.print_record()} -- {self.points:.2f} points"


class Standings:
    def __init__(self, team_list: List[TeamRecord] = None):
        self.team_list = team_list or []
        self._sort_teams()

    def add_team(self, team: TeamRecord):
        self.team_list.append(team)
        self._sort_teams()

    def show_standings(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Team": [x.team_name for x in self.team_list],
            # "Manager": [x.manager_name for x in self.team_list],
            "Record": [f"{x.print_record()}" for x in self.team_list],
            "Points": [x.points for x in self.team_list],
        })
        df.index += 1
        return df

    def get_team(self, team_name: str):
        return [x for x in self.team_list if x.team_name == team_name][0]

    def _sort_teams(self):
        self.team_list = sorted(self.team_list)[::-1]

    def __eq__(self, other):
        return [x.team_name for x in self.team_list] == [x.team_name for x in other.team_list]

    def __repr__(self):
        out = [f"({i+1}) {x.team_name}" for i, x in enumerate(self.team_list)]
        return ", ".join(out)


# number of weeks in regular season
weeks_dict = {
    2021: 14,
    2020: 13,
    2019: 13,
    2018: 13,
    2017: 13,
}

# record_dict structure
#  {
#    year_1: {
#      team_key_1: {
#        wins: int,
#        losses: int,
#        ties: int,
#        points: float,
#        new_wins: int,
#        new_losses: int,
#        new_ties: int,
#        new_points: float,
#      },
#      ...
#    },
#    ...
#  }
record_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
for i, row in matchup_df.iterrows():
    year = int(row["year"])
    if int(row["week"]) > weeks_dict[year]:
        continue
    team_key_1 = row["team_key_1"]
    team_key_2 = row["team_key_2"]
    winner = row["current_winner"]
    new_winner = row["new_winner"]
    record_dict[year][team_key_1]["wins"] += (winner == 1)
    record_dict[year][team_key_1]["losses"] += (winner == 2)
    record_dict[year][team_key_1]["losses"] += (winner == 0)
    record_dict[year][team_key_1]["points"] += row["team_1_score"]
    record_dict[year][team_key_1]["new_wins"] += (new_winner == 1)
    record_dict[year][team_key_1]["new_losses"] += (new_winner == 2)
    record_dict[year][team_key_1]["new_losses"] += (new_winner == 0)
    record_dict[year][team_key_1]["new_points"] += row["team_1_new_score"]
    record_dict[year][team_key_2]["wins"] += (winner == 2)
    record_dict[year][team_key_2]["losses"] += (winner == 1)
    record_dict[year][team_key_2]["losses"] += (winner == 0)
    record_dict[year][team_key_2]["points"] += row["team_2_score"]
    record_dict[year][team_key_2]["new_wins"] += (new_winner == 2)
    record_dict[year][team_key_2]["new_losses"] += (new_winner == 1)
    record_dict[year][team_key_2]["new_losses"] += (new_winner == 0)
    record_dict[year][team_key_2]["new_points"] += row["team_2_new_score"]


standings_dict = {}
for year, team_dict in record_dict.items():
    standings_dict[year] = {"standings": Standings(), "new_standings": Standings()}
    for k, team in team_dict.items():
        team_name = TEAMS[k]["team_name"]
        manager_name = MANAGERS[TEAMS[k]["manager_key"]]["nickname"]
        record = TeamRecord(
            team_name=team_name,
            manager_name=manager_name,
            wins=team["wins"],
            losses=team["losses"],
            ties=team["ties"],
            points=team["points"],
        )
        new_record = TeamRecord(
            team_name=team_name,
            manager_name=manager_name,
            wins=team["new_wins"],
            losses=team["new_losses"],
            ties=team["new_ties"],
            points=team["new_points"],
        )
        standings_dict[year]["standings"].add_team(record)
        standings_dict[year]["new_standings"].add_team(new_record)

years = [x for x in standings_dict.keys()]
tabs = st.tabs([str(x) for x in years])
for tab, year in zip(tabs, years):
    with tab:
        standings = standings_dict[year]["standings"]
        new_standings = standings_dict[year]["new_standings"]
        if standings == new_standings:
            st.write("**No change to regular season standings.**")
        else:
            st.write("**At least 2 teams would change places in the regular "
                     "season standings.**")

        for t in standings.team_list:
            new_t = new_standings.get_team(t.team_name)
            if t.print_record() != new_t.print_record():
                st.write(f"- {t.team_name}: {t.print_record()} --> {new_t.print_record()}")

        st.write("---")
        c1, c2 = st.columns(2)
        c1.write("###### Actual standings")
        c2.write("###### New standings")
        c1.write(standings.show_standings())
        c2.write(new_standings.show_standings())


################
# matchup detail
################

st.write("### Affected matchups")


def get_team_dicts(row: pd.Series) -> Tuple[dict, dict]:
    out = []
    for i in ["1", "2"]:
        team_key = row[f"team_key_{i}"]
        team_dict = TEAMS[team_key]
        team_dict["team_key"] = team_key
        manager_key = team_dict["manager_key"]
        team_dict["manager_name"] = MANAGERS[manager_key]["nickname"]
        team_dict["actual_defense_score"] = row[f"team_{i}_current_defense_score"]
        team_dict["new_defense_score"] = row[f"team_{i}_new_defense_score"]
        team_dict["actual_total_score"] = row[f"team_{i}_score"]
        team_dict["new_total_score"] = row[f"team_{i}_new_score"]
        out.append(team_dict)
    return out[0], out[1]


def create_matchup_card(team_1: dict, team_2: dict, week: int, year: int):
    game_id = league_key.split(".")[-1]
    team_1_number = team_1["team_key"].split(".")[-1]
    team_2_number = team_2["team_key"].split(".")[-1]
    link = f"https://football.fantasysports.yahoo.com/{year}/f1/{game_id}/" \
           f"matchup?week={week}&mid1={team_1_number}&mid2={team_2_number}"
    title = f"{year} &#8212 Week {week}"  # &#8212 = em dash

    def team_scores(team_dict, check_actual: int = None, check_new: int = None) -> str:
        act_score = team_dict["actual_total_score"]
        new_score = team_dict["new_total_score"]
        return f"""
            <td>
                <div>{team_dict["actual_defense_score"]}</div>
                <div>{team_dict["new_defense_score"]}</div>
                <div><b>{'✅ ' * (check_actual == 2)}{act_score}{' ✅' * (check_actual == 1)}</b></div>
                <div><b>{'✅ ' * (check_new == 2)}{new_score}{' ✅' * (check_new == 1)}</b></div>
            </td>
        """.strip()

    actual_winner = (
        1 if team_1["actual_total_score"] > team_2["actual_total_score"]
        else 2 if team_1["actual_total_score"] < team_2["actual_total_score"]
        else 0
    )
    new_winner = (
        1 if team_1["new_total_score"] > team_2["new_total_score"]
        else 2 if team_1["new_total_score"] < team_2["new_total_score"]
        else 0
    )
    scores_left = team_scores(
        team_dict=team_1,
        check_actual=1 if actual_winner == 1 else None,
        check_new=1 if new_winner == 1 else None
    )
    scores_right = team_scores(
        team_dict=team_2,
        check_actual=2 if actual_winner == 2 else None,
        check_new=2 if new_winner == 2 else None
    )

    def team_header(team_dict) -> str:
        return f"""
            <td>
                <div><b>{team_dict["team_name"]}</b></div>
                <div>({team_dict["manager_name"]})</div>
                <object data="{team_dict['team_logo']}" type="image/png"
            </td>
        """.strip()

    style = """
        <style>
        table {
            text-align: center;
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            border: none;
        }
        tr {
            border: none;
        }
        td {
            border: none;
        }
        figure {
            overflow: hidden;
            border: 1px solid;
            border-radius: 5px;
            padding: 5px;
        }
        object {
            height: 75px;
            padding: 2px;
            object-fit: contain;
        }
        </style>
    """

    out = f"""
        {style}
        <body>
            <figure>
                <center>
                    <h5><a href="{link}">{title}</a></h5>
                </center>
                <table>
                    <tr>
                        {team_header(team_1)}
                        <td><h4>vs.</h4></td>
                        {team_header(team_2)}
                    </tr>
                    <tr>
                        {scores_left}
                        <td>
                            <div>Actual defense score</div>
                            <div>New defense score</div>
                            <div><b>Actual total score</b></div>
                            <div><b>New total score</b></div>
                        </td>
                        {scores_right}
                    </tr>
                </table>
            </figure>
    """
    return out


with st.expander(expanded=True, label="View the matchups that would be affected "
                                      "by the new settings"):
    matchup_subset = matchup_df[matchup_df["different_result"]]
    matchup_subset["week"] = matchup_subset["week"].astype(int)
    matchup_subset = matchup_subset.sort_values(["year", "week"], ascending=[False, True])

    for i, row in matchup_subset.iterrows():
        league_key = row["league_key"]
        team_1, team_2 = get_team_dicts(row)
        card = create_matchup_card(team_1=team_1, team_2=team_2, week=row["week"], year=row["year"])
        st.markdown(card, unsafe_allow_html=True)
