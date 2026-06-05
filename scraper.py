import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

## for Team Codes

# CSK(Chennai Super Kings),         DDV(Delhi Capitals)
# KKR(Kolkata Knight Riders),       LSG(Lucknow Super Giants)
# MIN(Mumbai Indians),              KXI(Punjab Kings)
# RRO(Rajasthan Royals),            GTI(Gujarat Titans),
# RCB(Royal Challengers Bengaluru)

teams=["RCB", "GTI"]

def get_match_list(team_code):

    url = f"https://www.howstat.com/Cricket/Statistics/IPL/MatchList.asp?Season=2026&Team1={team_code}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Could not fetch match list for {team_code}: {e}")
        return []
    
    soup = BeautifulSoup(res.content, 'html.parser')
    
    match_table = soup.find("table", {"class": "TableLined"})

    if match_table is None:
        print(f"  [WARNING] Match table not found for team: {team_code}")
        return []
    

    rows = match_table.find_all('tr') # type: ignore

    matches_list = []
    
    # to get latest match first reversed loop
    for row in reversed(rows):

        #since we need only 10 matches data
        if len(matches_list) >= 10:
            break
        
            
        cells = row.find_all('td')
        
        # Safe check: Verify the row has all 6 columns before accessing indices
        if len(cells) >= 6:

            # cell 0 contain index of matches  so we go to cell 1 of <td> lists
            link_tag = cells[1].find('a', href=True) # Date is safely targeted in cells[1]
            
            if link_tag and 'MatchScorecard.asp' in link_tag['href']:
                date = cells[1].text.strip()
                team1 = cells[2].text.strip()
                team2 = cells[3].text.strip()
                venue = cells[4].text.strip()
                result = cells[5].text.strip()
                
                scorecard_url = "http://www.howstat.com/Cricket/Statistics/IPL/" + link_tag['href'] # type: ignore
                
                matches_list.append({
                    "Date": date,
                    "Team 1": team1,
                    "Team 2": team2,
                    "Venue": venue,
                    "Result": result,
                    "Scorecard_URL": scorecard_url
                })
                
    print(f"  Found {len(matches_list)} matches for {team_code}")
    return matches_list



def extract_scorer_and_score(url):

    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Could not fetch scorecard: {e}")
        return "Unknown", 0

    soup=BeautifulSoup(res.content,'html.parser')

    highest_score=0
    batsman="Unknown"

    # teams runs and name is in table classed naem ScorecardMain contain of both team 1 and 2 
    all_tables = soup.find_all('table', {"class": "ScorecardMain"})

    for table in all_tables:
        rows=table.find_all('tr')

        for row in rows:
            cells=row.find_all('td')

            if len(cells) >= 3:
                # player name contain in column one means column 0 as in python
                player_name = cells[0].text.strip()
                # player runs contain in column 3 means column 2 as in python
                runs_text = cells[2].text.strip()
            
                if runs_text.isdigit():
                    name_check = player_name.upper()
                    
                    if "TOTAL" in name_check or "EXTRAS" in name_check:
                        continue
                    runs = int(runs_text)
                    
                    # Update our highest score tracker if this batsman scored more
                    if runs > highest_score: # type: ignore
                        highest_score = runs
                        batsman = player_name

    return batsman, highest_score


all_matches_data = []

# 2. Collect match lists for both teams sequentially
for team in teams:
    
    team_matches = get_match_list(team) 
   
    all_matches_data.extend(team_matches) 

print(f"\nTotal matches to process across both teams: {len(all_matches_data)}")
print("Starting detailed scorecard extraction.\n")


for idx, match in enumerate(all_matches_data, 1):
    
    batsman, score = extract_scorer_and_score(match["Scorecard_URL"]) # pyright: ignore[reportUndefinedVariable]
    
    match["Top_Scorer"] = batsman
    match["Top_Score"] = score
    
    time.sleep(1)


# store using pandas dataframe and save to csv

df=pd.DataFrame(all_matches_data,columns=["Date", "Team 1", "Team 2", "Venue", "Result", "Top_Scorer", "Top_Score"])

df.to_csv("match_data.csv",index=False)

print(f"\nSaved {len(df)} matches to match_data.csv")

