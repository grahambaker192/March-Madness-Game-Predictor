import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = 'https://basketball.realgm.com/ncaa/team_stats/2024/Averages/Team_Totals/0/0/minutes/desc'
response = requests.get(URL)
soup = BeautifulSoup(response.content, 'html.parser')

columns = [
    "#",
    "Team",
    "GP",
    "MPG",
    "PPG",
    "FGM",
    "FGA",
    "FG%",
    "3PM",
    "3PA",
    "3P%",
    "FTM",
    "FTA",
    "FT%",
    "ORB",
    "DRB",
    "RPG",
    "APG",
    "SPG",
    "BPG",
    "TOV",
    "PF"
]

data = []

table = soup.find('table').tbody

trs = table.find_all('tr')
for tr in trs:
    tds = tr.find_all('td')
    row = [td.text.replace('\n', '') for td in tds]
    data.append(row)

df = pd.DataFrame(data, columns=columns)
df.to_csv('currentStats.csv', index=False)
