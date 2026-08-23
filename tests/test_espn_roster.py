from types import SimpleNamespace
from unittest.mock import Mock

from notifier.logging_utils import NotifierError
from notifier.roster import _find_my_espn_team, _load_espn, _normalized_swid


def _config(*, team_id=None, swid="{MY-OWNER-ID}"):
    return SimpleNamespace(
        espn_league_id=4242,
        espn_year=2026,
        espn_swid=swid,
        espn_s2="private-session-cookie",
        espn_team_id=team_id,
    )


def _payload():
    return {
        "settings": {"name": "Sunday Friends"},
        "teams": [
            {
                "id": 7,
                "owners": ["{MY-OWNER-ID}"],
                "location": "Arian's",
                "nickname": "Team",
                "roster": {
                    "entries": [
                        {
                            "lineupSlotId": 0,
                            "playerPoolEntry": {
                                "player": {
                                    "fullName": "Brock Purdy",
                                    "defaultPositionId": 1,
                                    "eligibleSlots": [0, 7, 20, 21],
                                    "proTeamId": 25,
                                }
                            },
                        },
                        {
                            "lineupSlotId": 21,
                            "playerPoolEntry": {
                                "player": {
                                    "fullName": "Christian McCaffrey",
                                    "defaultPositionId": 2,
                                    "eligibleSlots": [2, 3, 23, 20, 21],
                                    "proTeamId": 25,
                                }
                            },
                        },
                    ]
                },
            },
            {
                "id": 11,
                "owners": ["{RIVAL-OWNER-ID}"],
                "name": "The Rivals",
                "roster": {
                    "entries": [
                        {
                            "lineupSlotId": 4,
                            "playerPoolEntry": {
                                "player": {
                                    "fullName": "Mike Evans",
                                    "defaultPositionId": 3,
                                    "eligibleSlots": [4, 5, 23, 20, 21],
                                    "proTeamId": 27,
                                }
                            },
                        }
                    ]
                },
            },
        ],
    }


def test_direct_espn_response_preserves_every_team_roster_and_auth() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = _payload()
    session = Mock()
    session.get.return_value = response
    config = _config()

    league, players = _load_espn(config, session)

    assert league.name == "Sunday Friends"
    assert league.my_team_name == "Arian's Team"
    assert [(p.name, p.position, p.pro_team, p.lineup_slot, p.on_my_team) for p in players] == [
        ("Brock Purdy", "QB", "SF", "QB", True),
        ("Christian McCaffrey", "RB", "SF", "IR", True),
        ("Mike Evans", "WR", "TB", "WR", False),
    ]
    call = session.get.call_args
    assert call.kwargs["params"] == {"view": ["mTeam", "mRoster", "mSettings"]}
    assert call.kwargs["cookies"] == {
        "SWID": "{MY-OWNER-ID}",
        "espn_s2": "private-session-cookie",
    }
    assert call.kwargs["timeout"] == 25


def test_espn_team_id_override_supports_co_managed_and_changed_owner_ids() -> None:
    teams = _payload()["teams"]
    selected = _find_my_espn_team(teams, _config(team_id=11, swid="{OLD-ID}"))
    assert selected["id"] == 11


def test_espn_owner_matching_accepts_encoded_swid_and_owner_objects() -> None:
    assert _normalized_swid("%7BMY-OWNER-ID%7D") == "my-owner-id"
    team = _find_my_espn_team(
        [{"id": 3, "owners": [{"id": "{MY-OWNER-ID}"}]}],
        _config(swid="%7BMY-OWNER-ID%7D"),
    )
    assert team["id"] == 3


def test_espn_team_lookup_fails_with_an_actionable_override_hint() -> None:
    try:
        _find_my_espn_team(_payload()["teams"], _config(swid="{NOT-A-MEMBER}"))
    except NotifierError as error:
        assert "Set ESPN_TEAM_ID explicitly" in str(error)
    else:
        raise AssertionError("missing owner should not silently select another team")
