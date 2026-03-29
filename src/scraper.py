import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

URLS = {
    'batters_central':  'https://npb.jp/bis/2024/stats/bat_c.html',
    'batters_pacific':  'https://npb.jp/bis/2024/stats/bat_p.html',
    'pitchers_central': 'https://npb.jp/bis/2024/stats/pit_c.html',
    'pitchers_pacific': 'https://npb.jp/bis/2024/stats/pit_p.html',
}

def scrape_table(url):
    print(f"取得中: {url}")
    res = requests.get(url, headers=HEADERS, timeout=15)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'lxml')
    tables = soup.find_all('table')
    if not tables:
        print("テーブルが見つかりませんでした")
        return None
    df = pd.read_html(StringIO(str(tables[0])))[0]

    # 1行目（説明文）と2行目（ヘッダー）を除去してカラム名を設定
    df.columns = df.iloc[1]
    df = df.iloc[2:].reset_index(drop=True)

    # カラム名のスペースを除去
    df.columns = [str(c).replace('\u3000', '').replace(' ', '') for c in df.columns]

    # 重複カラムの2つ目（チーム名）を「チーム」に rename
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            new_cols.append('チーム')
        else:
            seen[c] = True
            new_cols.append(c)
    df.columns = new_cols

    # nanカラムを除去
    df = df.loc[:, df.columns != 'nan']

    return df

def scrape_all():
    batters_list = []
    pitchers_list = []

    for key, url in URLS.items():
        df = scrape_table(url)
        if df is None:
            continue
        if 'batters' in key:
            league = 'セ・リーグ' if 'central' in key else 'パ・リーグ'
            df['リーグ'] = league
            batters_list.append(df)
        else:
            league = 'セ・リーグ' if 'central' in key else 'パ・リーグ'
            df['リーグ'] = league
            pitchers_list.append(df)
        time.sleep(1)

    if batters_list:
        batters = pd.concat(batters_list, ignore_index=True)
        batters.to_csv('data/batters_2024.csv', index=False, encoding='utf-8-sig')
        print(f"打者データ保存完了: {len(batters)}件")
        print(batters[['選手', 'チーム', '打率', '本塁打', '打点', 'リーグ']].head(5).to_string())

    if pitchers_list:
        pitchers = pd.concat(pitchers_list, ignore_index=True)
        pitchers.to_csv('data/pitchers_2024.csv', index=False, encoding='utf-8-sig')
        print(f"\n投手データ保存完了: {len(pitchers)}件")
        print(pitchers[['投手', 'チーム', '防御率', '勝利', '敗北', 'リーグ']].head(5).to_string())

if __name__ == "__main__":
    scrape_all()
