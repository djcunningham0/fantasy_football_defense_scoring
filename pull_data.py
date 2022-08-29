import os
from authlib.integrations.requests_client import OAuth2Session
import time
import requests
import json
import pandas as pd
from collections import defaultdict

from typing import List

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from dotenv import load_dotenv
load_dotenv()


### Parameters
# MIN_GAME_ID = 348  # earliest year with all current D/ST stats (2015)
MIN_GAME_ID = 371  # earliest year we used our current D/ST settings (2017)
WAIT_AND_RETRY = True  # retry API calls after 30 seconds if they fail?
###


AUTHORIZATION_ENDPOINT = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_ENDPOINT = "https://api.login.yahoo.com/oauth2/get_token"
SCOPE = "email profile fspt-w"  # https://developer.yahoo.com/oauth2/guide/yahoo_scopes/

client = OAuth2Session(
    os.environ.get("YAHOO_CLIENT_ID"),
    os.environ.get("YAHOO_CLIENT_SECRET"),
    redirect_uri="oob",
    scope=SCOPE,
)

uri, state = client.create_authorization_url(AUTHORIZATION_ENDPOINT)
print(uri)

code = input("code copied from Yahoo: ")
token = client.fetch_token(TOKEN_ENDPOINT, grant_type="authorization_code", code=code)


def call_yahoo_api(uri: str, method: str = "get") -> dict:
    """Hit any endpoint for the yahoo api"""
    url = os.path.join("https://fantasysports.yahooapis.com/fantasy/v2/", uri)
    logger.debug(f"calling yahoo api at: {url}")
    r = client.request(method=method, url=url, params={"format": "json"})
    try:
        r.raise_for_status()
    except Exception as e:
        return {"ERROR": e}
    if WAIT_AND_RETRY:
        try:
            return r.json()
        except requests.JSONDecodeError:
            sleep_time = 30
            logger.info(f"JSONDecodeError -- sleeping {sleep_time} seconds")
            time.sleep(sleep_time)
            return call_yahoo_api(uri, method)
    else:
        return r.json()


def get_leagues():
    uri = "users;use_login=1/games;game_codes=nfl/leagues"
    data = call_yahoo_api(uri=uri, method="get")
    data = data["fantasy_content"]["users"]["0"]["user"][1]["games"]
    data = _remove_count_key(data)

    league_list = []
    for i, season in data.items():
        league_data = season["game"][1]["leagues"]
        if len(league_data) == 0:
            continue
        league_data = _remove_count_key(league_data)
        for j, league in league_data.items():
            league = league["league"][0]
            keep_keys = [
                "league_key",
                "game_code",
                "season",
                "name",
                "league_type",
                "start_date",
                "end_date",
                "current_week",
                "start_week",
                "end_week",
                "is_finished",
                "logo_url",
                "num_teams",
                "renew",
                "renewed",
                "scoring_type",
                "url",
            ]
            league = {k: league.get(k, None) for k in keep_keys}
            league_list.append(league)

    league_list = sorted(league_list, key=lambda x: (x.get("season"), x.get("league_key")), reverse=True)
    return league_list


def _remove_count_key(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "count"}


def get_matchup_history_one_league(league: dict) -> List[dict]:
    """
    Output: list of dicts
    {
        "league_key": league_key,
        "week": week,
        "team_key_1": team_key for team 1,
        "team_key_2": team_key for team 2,
        "team_1_score": score for team 1,
        "team_2_score": score for team 2,
    }
    """
    league_key = league.get("league_key")
    start_week = league.get("start_week")
    end_week = league.get("end_week")
    week_list = range(int(start_week), int(end_week) + 1)
    week_list_str = ",".join([str(x) for x in week_list])
    uri = f"league/{league_key}/scoreboard;week={week_list_str}"
    data = call_yahoo_api(uri=uri, method="get")
    try:
        data = data["fantasy_content"]["league"][1]["scoreboard"]["0"]["matchups"]
    except KeyError as e:
        # sometimes the yahoo api fails when you pass a comma-delimited list of weeks...
        # I think it's when the league doesn't start in week 1...
        # resolve it by going week by week and building the full dataset
        data = {}
        for i, week in enumerate(week_list):
            uri = f"league/{league_key}/scoreboard;week={week}"
            tmp_data = call_yahoo_api(uri=uri, method="get")
            tmp_data = tmp_data["fantasy_content"]["league"][1]["scoreboard"]["0"]["matchups"]
            tmp_data = _remove_count_key(tmp_data)
            if len(data) != 0:
                # keys returned by yahoo api are "0", "1", and so on
                # we need to renumber the single week data so it can be combined with previous weeks
                max_key = max([int(x) for x in data.keys()])
                for k in list(tmp_data.keys()):
                    new_key = str(int(k) + max_key + 1)
                    tmp_data[new_key] = tmp_data.pop(k)
            data = data | tmp_data  # merge dictionaries

    data = _remove_count_key(data)
    matchup_list = []
    for k, matchup in data.items():
        matchup = matchup["matchup"]
        if matchup["status"] != "postevent":
            continue  # don't process future matchups
        teams = matchup["0"]["teams"]
        points_1 = teams["0"]["team"][1]
        points_2 = teams["1"]["team"][1]
        matchup_list.append({
            "league_key": league_key,
            "year": league.get("season"),
            "week": matchup["week"],
            "team_key_1": teams["0"]["team"][0][0]["team_key"],
            "team_key_2": teams["1"]["team"][0][0]["team_key"],
            "team_1_score": float(points_1["team_points"]["total"]),
            "team_2_score": float(points_2["team_points"]["total"]),
        })
    return matchup_list


def get_matchup_history_all_leagues(leagues):
    matchup_history_list = []
    for league in leagues:
        matchup_history_list.append(get_matchup_history_one_league(league))
    return matchup_history_list


###############################
# stat ID --> stat name mapping
###############################

def build_stat_dict():
    stats = call_yahoo_api(f"game/{MIN_GAME_ID}/stat_categories")
    stats = stats["fantasy_content"]["game"][1]["stat_categories"]["stats"]
    stats = [x for x in stats if x["stat"]["position_types"][0]["position_type"] == "DT"]
    out = {}
    for x in stats:
        key = str(x["stat"]["stat_id"])
        out[key] = x["stat"]["name"]
    return out


logger.info("==== building stat dict ====")
stat_dict = build_stat_dict()
with open("./data/stat_category_map.json", "w") as f:
    json.dump(stat_dict, f, indent=2)


##################
# current settings
##################

def get_current_defense_settings():
    defense_stats = build_stat_dict()
    league_key = "406.l.231515"  # 2021 league  # TODO: don't hard code
    uri = f"league/{league_key}/settings"
    data = call_yahoo_api(uri)["fantasy_content"]["league"][1]["settings"][0]["stat_modifiers"]["stats"]
    active_stats = {}
    for x in data:
        key = str(x["stat"]["stat_id"])
        if key in defense_stats:
            active_stats[key] = float(x["stat"]["value"])

    # fill in the unused stats with zero
    out = {}
    for key in defense_stats:
        val = active_stats[key] if key in active_stats else 0.0
        out[key] = val
    return out


logger.info("==== getting current settings ====")
current_settings = get_current_defense_settings()
with open("./data/current_settings.json", "w") as f:
    json.dump(current_settings, f, indent=2)


#######################
# team and manager data
#######################


def get_teams_and_managers(leagues):
    team_list = []
    manager_list = []
    for league in leagues:
        league_key = league["league_key"]
        uri = f"league/{league_key}/teams"
        data = call_yahoo_api(uri=uri, method="get")
        teams = data["fantasy_content"]["league"][1]["teams"]
        teams = _remove_count_key(teams)
        for k, team in teams.items():
            team = _remove_count_key(team)
            team = team["team"][0]
            manager = team[19]["managers"][0]["manager"]
            manager = _process_manager_data(data=manager, league_key=league_key)
            manager_list.append(manager)
            out = {
                "league_key": league_key,
                "team_key": team[0]["team_key"],
                "team_name": team[2]["name"],
                "team_logo": team[5]["team_logos"][0]["team_logo"]["url"],
                "manager_key": manager["manager_key"],
            }
            team_list.append(out)

    team_dict = {}
    for x in team_list:
        team_dict[x["team_key"]] = {
            "league_key": x["league_key"],
            "team_name": x["team_name"],
            "team_logo": x["team_logo"],
            "manager_key": x["manager_key"]
        }

    # get distinct managers
    manager_list = set([frozenset(x.items()) for x in manager_list])  # convert to hashable frozenset so we can set
    manager_list = [dict(x) for x in manager_list]  # convert back to dict
    manager_dict = {}
    for x in manager_list:
        manager_dict[x["manager_key"]] = {
            "nickname": x["nickname"],
            # "image_url": x["image_url"]
        }
    return team_dict, manager_dict


def _process_manager_data(data: dict, league_key: str) -> dict:
    """
    data:
    {
      "guid": str,              # sometimes populated, sometimes "--" (not populated for deleted accounts?)
      "image_url": str,         # sometimes
      "is_current_login": str,  # sometimes ("1" or not populated)
      "manager_id": str,        # always ("1" through "n_teams")
      "nickname": str,          # always, but sometimes "--hidden--" or "-- hidden --"
    }

    Output:
    {
      "manager_key": guid if populated, else "{league_key}_{manager_id}",
      "image_url": image_url if populated, else None
      "nickname": nickname, with --hidden-- as None
    """
    if nickname := data.get("nickname", ""):
        nickname = nickname if nickname.replace(" ", "") != "--hidden--" else None
    return {
        "manager_key": data.get("guid", "").replace("-", "") or f"{league_key}_{data.get('manager_id')}",
        "image_url": data.get("image_url"),
        "nickname": nickname,
    }


logger.info("==== pulling team and manager data ====")
leagues = get_leagues()
leagues = [x for x in leagues if int(x["league_key"][:3]) >= MIN_GAME_ID]
leagues = [x for x in leagues if int(x["season"]) < 2022]  # don't get the current season
teams, managers = get_teams_and_managers(leagues)
with open("./data/teams.json", "w") as f:
    json.dump(teams, f, indent=2)
with open("./data/managers.json", "w") as f:
    json.dump(managers, f, indent=2)


######################
# matchup history data
######################

logger.info("==== pulling matchup history data ====")
matchup_history = get_matchup_history_all_leagues(leagues)
with open("./data/matchup_history.json", "w") as f:
    json.dump(matchup_history, f, indent=2)


########################
# get weekly team scores
########################


def get_defense_for_team(team_key: str, week: int):
    uri = f"team/{team_key}/roster;week={week}"
    roster = call_yahoo_api(uri)
    roster = roster["fantasy_content"]["team"][1]["roster"]["0"]["players"]
    roster = _remove_count_key(roster)
    try:
        d_key = [x for x in roster if roster[x]["player"][1]["selected_position"][1]["position"] == "DEF"][0]
    except IndexError:
        # no defense selected
        return None
    d_player_key = roster[d_key]["player"][0][0]["player_key"]
    return d_player_key


def get_defense_stats(d_player_key: str, week: int):
    if d_player_key is None:
        # team did not play a defense this week
        return defaultdict(float)

    uri = f"player/{d_player_key}/stats;type=week;week={week}"
    stats = call_yahoo_api(uri)
    stats = stats["fantasy_content"]["player"][1]["player_stats"]["stats"]
    stat_dict = {}
    for x in stats:
        stat_dict[x["stat"]["stat_id"]] = int(x["stat"]["value"])
    return stat_dict


logger.info("==== pulling weekly team scores ====")
data_dict = defaultdict(list)
for i, league in enumerate(leagues):
    league_key = league["league_key"]
    year = league["season"]
    print(f"\n{year}--{league_key} ({i+1} / {len(leagues)}):", end=" ")
    matchup_history = get_matchup_history_one_league(league)
    for matchup in matchup_history:
        print(".", end="", flush=True)
        week = matchup["week"]
        team_1 = matchup["team_key_1"]
        team_2 = matchup["team_key_2"]
        for team in [team_1, team_2]:
            defense = get_defense_for_team(team_key=team, week=week)
            stats = get_defense_stats(d_player_key=defense, week=week)
            data_dict["league_key"].append(league_key)
            data_dict["year"].append(year)
            data_dict["week"].append(week)
            data_dict["team_key"].append(team)
            for x in stat_dict:
                data_dict[x].append(stats[x])

df = pd.DataFrame(data_dict)
df.to_csv("./data/data.csv", index=False)
