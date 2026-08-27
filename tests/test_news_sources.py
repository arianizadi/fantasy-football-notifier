from __future__ import annotations

import queue
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

from notifier.models import LeagueRef, NewsItem, RosterPlayer, RosterSnapshot
from notifier.pipeline import _depth_report_text
from notifier.plays import DepthCharts
from notifier.sources.rotowire import parse_feed, reattribute_beneficiary_report
from notifier.sources.reporters import ALL_REPORTERS, PlayerNameIndex, build_stream_rules
from notifier.sources.twitter import TwitterStream


def test_stream_rules_include_verified_packers_breaking_reporters() -> None:
    expected = {"mattschneidman", "ByRyanWood", "RobDemovsky", "by_JBH"}
    assert expected.issubset(set(ALL_REPORTERS))
    rendered = " ".join(rule["value"] for rule in build_stream_rules())
    for handle in expected:
        assert f"from:{handle}" in rendered


def test_rotowire_feed_preserves_source_time_and_cleans_markup() -> None:
    xml = """
    <rss><channel><item>
      <title>George Kittle: Returns to practice</title>
      <guid>kittle-1</guid>
      <description><![CDATA[<b>Kittle</b> returned. Visit RotoWire.com for more analysis.]]></description>
      <link>https://www.rotowire.com//football/player.php?id=1</link>
      <pubDate>Sun, 23 Aug 2026 17:30:00 +0000</pubDate>
    </item></channel></rss>
    """

    item = parse_feed(xml)[0]

    assert item.player_name == "George Kittle"
    assert item.headline == "Returns to practice"
    assert item.body == "Kittle returned."
    assert item.published_at == datetime(2026, 8, 23, 17, 30, tzinfo=timezone.utc)
    assert item.url == "https://www.rotowire.com/football/player.php?id=1"


def test_rotowire_feed_preserves_explicit_zone_with_12_hour_clock() -> None:
    xml = """
    <rss><channel>
      <item>
        <title>Kayshon Boutte: Traded to Houston</title>
        <guid>boutte-pdt</guid>
        <description>Traded Monday.</description>
        <pubDate>Mon, 24 Aug 2026 11:51:00 AM PDT</pubDate>
      </item>
      <item>
        <title>Example Player: Afternoon update</title>
        <guid>player-pst</guid>
        <description>Updated Monday.</description>
        <pubDate>Mon, 14 Dec 2026 1:05:00 PM PST</pubDate>
      </item>
      <item>
        <title>Example Player: Unknown timezone</title>
        <guid>player-unknown-zone</guid>
        <description>Updated Monday.</description>
        <pubDate>Mon, 24 Aug 2026 11:51:00 AM XYZ</pubDate>
      </item>
    </channel></rss>
    """

    by_guid = {item.guid: item for item in parse_feed(xml)}

    assert by_guid["rotowire:boutte-pdt"].published_at == datetime(
        2026, 8, 24, 18, 51, tzinfo=timezone.utc
    )
    assert by_guid["rotowire:player-pst"].published_at == datetime(
        2026, 12, 14, 21, 5, tzinfo=timezone.utc
    )
    assert by_guid["rotowire:player-unknown-zone"].published_at is None


def _raiders_running_backs() -> dict[str, dict[str, object]]:
    return {
        "1": {
            "full_name": "Ashton Jeanty",
            "position": "RB",
            "team": "LV",
            "depth_chart_order": 1,
            "status": "Active",
        },
        "2": {
            "full_name": "Mike Washington",
            "position": "RB",
            "team": "LV",
            "depth_chart_order": 2,
            "status": "Active",
        },
        "3": {
            "full_name": "Dylan Laube",
            "position": "RB",
            "team": "LV",
            "depth_chart_order": 4,
            "status": "Active",
        },
    }


def test_rotowire_beneficiary_article_is_centered_on_injured_starter() -> None:
    item = parse_feed(
        """
        <rss><channel><item>
          <title>Mike Washington: Sees extra work after Jeanty injury</title>
          <guid>washington-1</guid>
          <description>Washington took most of the carries after Ashton Jeanty
          (knee) left Sunday's practice, Sam Warren of The Athletic reports.</description>
          <link>https://www.rotowire.com/football/player/mike-washington-999</link>
        </item></channel></rss>
        """
    )[0]

    normalized = reattribute_beneficiary_report(item, _raiders_running_backs())

    assert normalized.player_name == "Ashton Jeanty"
    assert normalized.headline == "Sees extra work after Jeanty injury"
    assert normalized.body == item.body
    assert normalized.guid == item.guid

    league = LeagueRef("sleeper", "1", "Test League", "Mine")
    snapshot = RosterSnapshot(
        generated_at=None,
        leagues=[league],
        players=[
            RosterPlayer(
                "Ashton Jeanty",
                "RB",
                "LV",
                "RB",
                True,
                "Mine",
                league.key,
            )
        ],
    )
    charts = DepthCharts(_raiders_running_backs(), snapshot)
    record, plays = charts.build(
        subject_names=(normalized.player_name,),
        snapshot=snapshot,
        report_text=_depth_report_text(normalized),
    )
    assert record is not None
    assert record["full_name"] == "Ashton Jeanty"
    mike = next(
        candidate
        for candidate in plays[0].beneficiaries
        if candidate.name == "Mike Washington"
    )
    assert mike.named_in_report is True
    context = charts.team_context(record, snapshot)
    assert context is not None
    assert next(entry for entry in context.same_position if entry.is_subject).name == (
        "Ashton Jeanty"
    )


def test_rotowire_does_not_guess_from_surname_only_or_unrelated_injury() -> None:
    base = NewsItem(
        source="rotowire",
        guid="rotowire:washington-2",
        player_name="Mike Washington",
        headline="Sees extra work after Jeanty injury",
        body="Washington took most of the carries after Jeanty left practice.",
        url="https://www.rotowire.com/football/player/mike-washington-999",
        published_at=None,
    )
    unrelated = replace(
        base,
        guid="rotowire:washington-3",
        headline="Handles extra work after morning practice",
        body=(
            "Washington took most of the carries. Ashton Jeanty (knee) left "
            "Sunday's practice in a separate drill."
        ),
    )

    assert reattribute_beneficiary_report(base, _raiders_running_backs()) == base
    assert reattribute_beneficiary_report(unrelated, _raiders_running_backs()) == unrelated


def test_twitter_payload_keeps_created_at_and_player_match() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["George Kittle"])
    payload = {
        "data": {
            "id": "42",
            "author_id": "7",
            "created_at": "2026-08-23T17:30:00.000Z",
            "text": "The 49ers activated George Kittle from active/PUP.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.guid == "twitter:42:George Kittle"
    assert item.player_name == "George Kittle"
    assert item.published_at == datetime(2026, 8, 23, 17, 30, tzinfo=timezone.utc)
    assert item.url == "https://x.com/Reporter/status/42"
    stream._session.close()


def test_twitter_report_without_player_match_is_not_subject_confident() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: [])
    payload = {
        "data": {
            "id": "unmatched-context",
            "author_id": "7",
            "text": "The receiver-needy team added depth after an earlier injury.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.player_name == ""
    assert item.subject_confident is False
    stream._session.close()


def test_transaction_with_position_descriptor_has_clear_player_subject() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Noah Brown"])
    payload = {
        "data": {
            "id": "noah-brown-release",
            "author_id": "7",
            "text": "Raiders released veteran WR Noah Brown.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.player_name == "Noah Brown"
    assert item.subject_confident is True
    stream._session.close()


def test_multi_player_injury_post_becomes_one_starter_centered_report() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(
        find=lambda _text: ["Arizona Starter", "Michael Carter", "Bam Knight"]
    )
    text = (
        "Arizona Starter was ruled out. Michael Carter and Bam Knight could "
        "split the available work."
    )
    payload = {
        "data": {"id": "43", "author_id": "7", "text": text},
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    items = stream._to_items(payload)

    assert len(items) == 1
    assert items[0].player_name == "Arizona Starter"
    assert items[0].subject_confident is True
    stream._session.close()


def test_ambiguous_multi_player_post_is_one_fail_closed_report() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Michael Carter", "Bam Knight"])
    payload = {
        "data": {
            "id": "44",
            "author_id": "7",
            "text": "Michael Carter and Bam Knight are both backfield options.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    items = stream._to_items(payload)

    assert len(items) == 1
    assert items[0].subject_confident is False
    stream._session.close()


def test_two_players_ruled_out_together_remains_ambiguous() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Player Alpha", "Player Beta"])
    payload = {
        "data": {
            "id": "45",
            "author_id": "7",
            "text": "Player Alpha and Player Beta were ruled out.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.subject_confident is False
    stream._session.close()


def test_single_matched_replacement_does_not_become_injured_subject() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    payload = {
        "data": {
            "id": "46",
            "author_id": "7",
            "text": "CMC ruled out; Jordan Mason is expected to handle the backfield.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.player_name == "Jordan Mason"
    assert item.subject_confident is False
    stream._session.close()


def test_replacement_language_cannot_inherit_unmatched_players_absence() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    reports = [
        "Jordan Mason will start with CMC ruled out.",
        "Jordan Mason benefits with CMC injured.",
        "Jordan Mason is the pickup after CMC was ruled out.",
        "Jordan Mason will start because CMC is doubtful.",
    ]

    for index, report in enumerate(reports, start=50):
        item = stream._to_items(
            {
                "data": {"id": str(index), "author_id": "7", "text": report},
                "includes": {"users": [{"id": "7", "username": "Reporter"}]},
            }
        )[0]
        assert item.subject_confident is False
    stream._session.close()


def test_direct_prefix_injury_headlines_remain_confident() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    reports = [
        "Torn ACL for Jordan Mason.",
        "Concussion for Jordan Mason.",
        "High ankle sprain for Jordan Mason.",
    ]

    for index, report in enumerate(reports, start=60):
        item = stream._to_items(
            {
                "data": {"id": str(index), "author_id": "7", "text": report},
                "includes": {"users": [{"id": "7", "username": "Reporter"}]},
            }
        )[0]
        assert item.player_name == "Jordan Mason"
        assert item.subject_confident is True
    stream._session.close()


def test_replacements_after_absence_cue_do_not_make_subject_ambiguous() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(
        find=lambda text: [
            name
            for name in ["James Conner", "Michael Carter", "Bam Knight"]
            if name in text
        ]
    )
    reports = [
        "James Conner ruled out and Michael Carter will start.",
        (
            "James Conner was ruled out, while Michael Carter and Bam Knight "
            "could split work."
        ),
        (
            "James Conner (ankle) was ruled out; Michael Carter and Bam Knight "
            "could split work."
        ),
    ]

    for index, report in enumerate(reports, start=70):
        item = stream._to_items(
            {
                "data": {"id": str(index), "author_id": "7", "text": report},
                "includes": {"users": [{"id": "7", "username": "Reporter"}]},
            }
        )[0]
        assert item.player_name == "James Conner"
        assert item.subject_confident is True
    stream._session.close()


def test_direct_status_variants_remain_confident() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    reports = [
        "Jordan Mason (ankle) ruled out.",
        "Jordan Mason — ankle — ruled out.",
        "Jordan Mason, who was ruled out.",
        "Jordan Mason entered concussion protocol.",
        "Jordan Mason has entered concussion protocol.",
        "Jordan Mason is a game-time decision.",
    ]

    for index, report in enumerate(reports, start=80):
        item = stream._to_items(
            {
                "data": {"id": str(index), "author_id": "7", "text": report},
                "includes": {"users": [{"id": "7", "username": "Reporter"}]},
            }
        )[0]
        assert item.subject_confident is True
    stream._session.close()


def test_non_roster_release_language_fails_closed() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    reports = [
        "Jordan Mason released a statement about CMC.",
        "Jordan Mason waived goodbye to CMC.",
        "Jordan Mason was released from the hospital.",
        "The 49ers released Jordan Mason injury update.",
        "The 49ers released Jordan Mason highlight video.",
        "Jordan Mason was released from concussion protocol by the 49ers.",
        "The 49ers released Jordan Mason from concussion protocol.",
        "Jordan Mason was released by the 49ers medical staff.",
        "The 49ers released Jordan Mason after he cleared concussion protocol.",
    ]

    for index, report in enumerate(reports, start=90):
        item = stream._to_items(
            {
                "data": {"id": str(index), "author_id": "7", "text": report},
                "includes": {"users": [{"id": "7", "username": "Reporter"}]},
            }
        )[0]
        assert item.subject_confident is False
    stream._session.close()


def test_explicit_team_release_language_remains_confident() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    reports = [
        "The 49ers released Jordan Mason.",
        "Jordan Mason was released by the 49ers.",
        "SF waived Jordan Mason from the roster.",
        "The 49ers cut Jordan Mason.",
        "Jordan Mason was suspended four games by the NFL.",
    ]

    for index, report in enumerate(reports, start=100):
        item = stream._to_items(
            {
                "data": {"id": str(index), "author_id": "7", "text": report},
                "includes": {"users": [{"id": "7", "username": "Reporter"}]},
            }
        )[0]
        assert item.subject_confident is True
    stream._session.close()


def test_worked_out_does_not_override_named_injured_subject() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(
        find=lambda _text: ["James Conner", "Michael Carter", "Bam Knight"]
    )
    payload = {
        "data": {
            "id": "47",
            "author_id": "7",
            "text": (
                "James Conner suffered a high ankle sprain. Michael Carter "
                "worked out with Bam Knight."
            ),
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.player_name == "James Conner"
    assert item.subject_confident is True
    stream._session.close()


def test_direct_single_player_ruled_out_remains_confident() -> None:
    stream = TwitterStream("fake", queue.Queue())
    stream._names = SimpleNamespace(find=lambda _text: ["Jordan Mason"])
    payload = {
        "data": {
            "id": "48",
            "author_id": "7",
            "text": "Jordan Mason was ruled out for Sunday.",
        },
        "includes": {"users": [{"id": "7", "username": "Reporter"}]},
    }

    item = stream._to_items(payload)[0]

    assert item.subject_confident is True
    stream._session.close()


def test_player_index_does_not_join_a_name_across_sentences() -> None:
    index = PlayerNameIndex(
        {
            "1": {
                "full_name": "Will Shipley",
                "team": "PHI",
                "position": "RB",
            }
        }
    )

    assert index.find("Will Shipley practiced.") == ["Will Shipley"]
    assert index.find("The team will. Shipley practiced.") == []
