## Team news input

Place league-specific news files in this folder to enrich pre-match and live
analysis with injuries, suspensions, and form trends.

### File format

Create a file named `<league>.json` (for example `premier_league.json`) with:

```json
{
  "teams": {
    "Arsenal": {
      "injuries": [
        { "name": "Player A", "position": "FW", "impact": 0.06 }
      ],
      "suspensions": [
        { "name": "Player B", "position": "MF", "impact": 0.05 }
      ],
      "doubts": [
        { "name": "Player C", "position": "DF", "impact": 0.03 }
      ],
      "key_players": [
        { "name": "Player D", "status": "out", "impact": 0.07 }
      ],
      "card_risk": 1,
      "form": {
        "shots_per90": 15.2,
        "xg_for_last5": 8.1,
        "xg_against_last5": 5.9
      }
    }
  }
}
```

Impact values should be between `0.00` and `0.10` per player. The system caps
total absence impact automatically.
