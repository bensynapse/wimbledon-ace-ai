"""
Live Tennis API source — offline tests.

Stdlib unittest only (no new dependency) and no network: the HTTP call is
stubbed. Run with:

    python -m unittest discover tests

Covers the two things that matter for an optional source: it is inert without
a key, and it maps the payload faithfully — dropping what it cannot use rather
than filling it in.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_sources.livetennis import LiveTennisSource  # noqa: E402


def _match(**overrides) -> dict:
    """A GET /matches item, shaped as the API documents it."""
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    match = {
        "id": 21131,
        "tournament": "Wimbledon",
        "surface": "grass",
        "indoor": False,
        "format": "BO5",
        "round": "Round of 16",
        "status": "upcoming",
        "event_status": None,
        "is_doubles": False,
        "scheduled_time": f"{tomorrow}T13:30:00Z",
        "players": {
            "p1": {"id": 501, "name": "Alex de Minaur"},
            "p2": {"id": 502, "name": "Flavio Cobolli"},
        },
        "score": None,
        "winner": None,
    }
    match.update(overrides)
    return match


def _payload(*matches) -> dict:
    return {
        "data": list(matches),
        "meta": {"limit": 200, "offset": 0, "count": len(matches),
                 "total": len(matches), "has_more": False},
    }


class InertWithoutKeyTest(unittest.TestCase):
    def test_disabled_and_silent(self):
        source = LiveTennisSource(api_key="")

        self.assertFalse(source.enabled)
        with mock.patch("data_sources.livetennis.requests.get") as get:
            self.assertEqual(source.get_upcoming_fixtures("atp"), [])
            get.assert_not_called()


class MappingTest(unittest.TestCase):
    def _fixtures(self, *matches, tour="atp", days_ahead=2):
        source = LiveTennisSource(api_key="test-key")
        response = mock.Mock()
        response.json.return_value = _payload(*matches)
        response.raise_for_status.return_value = None

        with mock.patch("data_sources.livetennis.requests.get",
                        return_value=response) as get:
            fixtures = source.get_upcoming_fixtures(tour, days_ahead=days_ahead)
            self.last_call = get.call_args
        return fixtures

    def test_maps_an_upcoming_fixture(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        fixtures = self._fixtures(_match())

        self.assertEqual(fixtures, [{
            "match_id": "21131",
            "date": tomorrow,
            "time": "13:30",
            "player1": "Alex de Minaur",
            "player2": "Flavio Cobolli",
            "player1_key": "501",
            "player2_key": "502",
            "tournament": "Wimbledon",
            "round": "Round of 16",
            "surface": "grass",
            "status": "upcoming",
        }])

    def test_sends_key_as_header_not_query_string(self):
        self._fixtures(_match())
        _, kwargs = self.last_call

        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertNotIn("token", kwargs["params"])
        self.assertEqual(kwargs["params"]["tour"], "atp")
        self.assertEqual(kwargs["params"]["status"], "upcoming")

    def test_skips_doubles(self):
        self.assertEqual(self._fixtures(_match(is_doubles=True)), [])

    def test_skips_matches_beyond_the_window(self):
        far = (date.today() + timedelta(days=30)).isoformat()
        self.assertEqual(
            self._fixtures(_match(scheduled_time=f"{far}T13:30:00Z")), [])

    def test_skips_unscheduled_or_unparseable_times(self):
        self.assertEqual(self._fixtures(_match(scheduled_time=None)), [])
        self.assertEqual(self._fixtures(_match(scheduled_time="soon")), [])

    def test_skips_fixtures_missing_a_player_name(self):
        self.assertEqual(
            self._fixtures(_match(players={"p1": {"id": 1}, "p2": {"id": 2, "name": "X"}})),
            [])

    def test_leaves_surface_empty_when_unknown(self):
        # The collector infers it from the tournament name; the source does not guess.
        self.assertEqual(self._fixtures(_match(surface=None))[0]["surface"], "")

    def test_rejects_an_unsupported_tour(self):
        self.assertEqual(self._fixtures(_match(), tour="mixed"), [])

    def test_network_failure_returns_empty_not_an_exception(self):
        source = LiveTennisSource(api_key="test-key")
        with mock.patch("data_sources.livetennis.requests.get",
                        side_effect=OSError("connection reset")):
            self.assertEqual(source.get_upcoming_fixtures("atp"), [])


if __name__ == "__main__":
    unittest.main()
