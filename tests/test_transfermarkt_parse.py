from ingestion.transfermarkt.models import ClubRecord
from ingestion.transfermarkt.parse import (
    parse_clubs,
    parse_players,
)


def test_parse_clubs_deduplicates_links() -> None:
    html = """
    <html><body>
      <a href="/arsenal/startseite/verein/11/saison_id/2025">Arsenal FC</a>
      <a href="/arsenal/startseite/verein/11/saison_id/2025">Arsenal FC</a>
      <a href="/chelsea/startseite/verein/631/saison_id/2025">Chelsea FC</a>
    </body></html>
    """

    clubs = parse_clubs(
        html=html,
        base_url="https://www.transfermarkt.com",
        season=2025,
    )

    assert [club.club_id for club in clubs] == [11, 631]
    assert clubs[0].roster_url.endswith("/arsenal/startseite/verein/11/saison_id/2025")


def test_parse_players_deduplicates_profile_links() -> None:
    html = """
    <html><body>
      <a href="/player-one/profil/spieler/101">Player One</a>
      <a href="/player-one/profil/spieler/101">Player One</a>
      <a href="/player-two/profil/spieler/202">Player Two</a>
    </body></html>
    """
    club = ClubRecord(
        club_id=11,
        club_name="Arsenal FC",
        profile_path="/arsenal/startseite/verein/11",
        roster_url="https://www.transfermarkt.com/arsenal/startseite/verein/11/saison_id/2025",
    )

    players = parse_players(html=html, club=club, season=2025)

    assert [player.player_id for player in players] == [101, 202]
    assert players[0].club_id == 11
