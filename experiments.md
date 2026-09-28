# Experiments: baseline results

Video 1 split: train even seasons, validate odd seasons, 1966-2023, without the 2023 Super Bowl.
Video 2 split: train odd seasons, validate even seasons, 1966-2024, without the 2024 Super Bowl.
Baseline "constant": predict the validation win rate for every row, 0.5 cutoff, accuracy.
Lag: 5 for video 1, 10 for video 2.

| Exp | Data | Ties | Reset each season | Team key | Rows | Filter | Baseline method | Video 1 | Video 2 |
|---|---|---|---|---|---|---|---|---|---|
| Author | Kaggle | ? | yes | ? | ? | ? | constant | 52.0% | 49.9% |
| 1 | nfl-history.csv | loss | yes | name | home view | none | constant | 57.5% | 56.6% |
| 1 | nfl-history.csv | loss | yes | name | per team | none | constant | 50.3% | 50.3% |
| 2 | Kaggle | removed | yes | name | home view | none | constant | 57.9% | 57.0% |
| 2 | Kaggle | removed | yes | name | per team | none | constant | 50.0% | 50.0% |
| 3 | Kaggle | loss | yes | name | home view | none | constant | 57.5% | 56.6% |
| 3 | Kaggle | loss | yes | name | per team | none | constant | 50.3% | 50.3% |
| 4 | Kaggle | loss | yes | name | home view | none | constant | 57.5% | 56.6% |
| 5 | Kaggle | removed | yes | name | home view | none | constant | 57.9% | 57.0% |
| 6 | Kaggle | 0.5 | yes | name | home view | none | constant | 57.9% | 57.0% |
| 7 | Kaggle | 0.5 | no | name | home view | none | constant | 57.9% | 57.0% |
| 7 | Kaggle | 0.5 | no | team_id | home view | none | constant | 57.9% | 57.0% |
| 7 | Kaggle | 0.5 | no | team_id | per team | none | constant | 50.0% | 50.0% |
| 8 | Kaggle | 0.5 | no | team_id | home view | start season 1966-2020 | constant | 53.5% to 57.9% | - |
| 8 | Kaggle | 0.5 | no | team_id | home view | start season 2023 | constant | 56.3% | - |
| 9 | Kaggle | 0.5 | yes | team_id | per team | team has >= 5 earlier games | constant | 50.0% | 50.1% |
| 9 | Kaggle | loss | yes | team_id | per team | none | constant | 50.3% | 50.3% |
| 9 | Kaggle | 0.5 | yes | team_id | home view | none | constant, expected accuracy | 51.2% | 51.0% |
| 10 | Kaggle | 0.5 | yes | team_id | per team | none | own avg diff > opp avg diff | 61.3% | 61.6% |
| 10 | Kaggle | 0.5 | yes | team_id | per team | team has >= 5 earlier games | own avg diff > opp avg diff | 63.0% | 64.2% |
| 10 | Kaggle | 0.5 | yes | team_id | home view | none | home avg diff >= away avg diff | 61.7% | 61.9% |
| 10 | Kaggle | 0.5 | yes | team_id | per team | both teams >= lag earlier games | own avg diff > opp avg diff | 62.9% | 65.1% |
| 10 | Kaggle | 0.5 | yes | team_id | home view | both teams >= lag earlier games | home avg diff >= away avg diff | 63.0% | 65.0% |
| 11 | Kaggle | 0.5 | yes | team_id | home view | home team >= lag earlier games | constant | 58.4% | 58.1% |
| 11 | Kaggle | 0.5 | yes | team_id | home view | away team >= lag earlier games | constant | 58.4% | 58.0% |
| 11 | Kaggle | 0.5 | yes | team_id | home view | both teams >= lag earlier games | constant | 58.5% | 58.1% |
| 11 | Kaggle | 0.5 | yes | team_id | per team | both teams >= lag earlier games | constant | 50.0% | 50.0% |
