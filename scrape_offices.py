# -*- coding: utf-8 -*-
"""
【GitHub Actions対応版】
タマホームの公式サイトから都道府県別の営業所数をスクレイピングし、
結果をDataFrameで返し、CSVとして保存し、ログに出力するスクリプト。
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin
import os
from datetime import datetime

def scrape_tamahome_offices():
    """
    タマホームの公式サイトから都道府県別の営業所数をスクレイピングし、
    結果をDataFrameで返す関数
    """
    # スクレイピング対象はモデルハウス一覧ページ
    BASE_URL = "https://www.tamahome.jp/modelhouse/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # --- ステップ1: 各都道府県へのリンクを取得 ---
    print(f"URLにアクセス中: {BASE_URL}")
    try:
        response = requests.get(BASE_URL, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        print("-> トップページへのアクセス成功")
    except requests.exceptions.RequestException as e:
        print(f"エラー: サイトにアクセスできませんでした。 {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    prefecture_links = []
    link_list_div = soup.find('div', class_='link-list')
    if not link_list_div:
        print("エラー: 都道府県リンクの一覧が見つかりませんでした。サイトの構造が変更された可能性があります。")
        return None

    links = link_list_div.find_all('a')
    for link in links:
        href = link.get('href')
        name = link.get_text(strip=True)
        if href and name:
            full_url = urljoin(BASE_URL, href)
            prefecture_links.append({"name": name, "url": full_url})

    print(f"-> {len(prefecture_links)}件の都道府県ページを検出しました。")

    # --- ステップ2: 各都道府県ページを巡回して営業所数をカウント ---
    office_data = []
    print("\n各都道府県の営業所数を集計中...")

    for i, pref in enumerate(prefecture_links):
        pref_name = pref['name']
        pref_url = pref['url']
        print(f"  ({i+1}/{len(prefecture_links)}) {pref_name} のページを処理中...", end="")

        try:
            page_response = requests.get(pref_url, headers=headers, timeout=10)
            page_response.raise_for_status()
            page_response.encoding = page_response.apparent_encoding

            page_soup = BeautifulSoup(page_response.text, 'html.parser')

            # 各営業所は 'c-modelhouse' というクラス名のdivで表現されている
            offices = page_soup.find_all('div', class_='c-modelhouse')
            count = len(offices)

            office_data.append({
                "都道府県": pref_name,
                "営業所数": count
            })
            print(f" -> {count}件")

        except requests.exceptions.RequestException as e:
            print(f" -> エラー: {pref_name}のページにアクセスできませんでした。スキップします。")

        time.sleep(0.5)

    print("-> データ抽出完了")

    if not office_data:
        print("エラー: 営業所データが1件も抽出できませんでした。")
        return None

    df = pd.DataFrame(office_data)
    df = df[df['営業所数'] > 0].copy()

    return df

# --- メイン処理 ---
if __name__ == '__main__':
    # スクレイピング実行
    df_offices = scrape_tamahome_offices()

    if df_offices is not None and not df_offices.empty:
        # --- 結果をCSVに保存 ---
        # 保存先ディレクトリをGitHub Actionsのローカルパスに変更
        SAVE_DIR = './TamaHome_CSV_Data' # 既存のスクリプトと同じパスを使用

        # 保存先ディレクトリが存在しない場合は作成
        os.makedirs(SAVE_DIR, exist_ok=True)

        # ファイル名に日付を追加
        date_str = datetime.now().strftime('%Y%m%d')
        file_name = f'tamahome_offices_{date_str}.csv'
        save_path = os.path.join(SAVE_DIR, file_name)

        try:
            # CSVファイルとして保存 (Excelでの文字化け対策に encoding='utf-8-sig')
            df_offices.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n--- CSV保存完了 ---")
            print(f"-> ファイルを '{save_path}' に保存しました。")
        except Exception as e:
            print(f"\n--- CSV保存エラー ---")
            print(f"-> ファイルの保存に失敗しました: {e}")

        # --- 結果を画面に表示 (displayの代わりにprintを使用) ---
        print("\n--- タマホーム 都道府県別 営業所数 ---")
        df_offices.index = range(1, len(df_offices) + 1)
        print(df_offices.to_string()) # display(df_offices)の代わりにto_string()を使用

        total_offices = df_offices['営業所数'].sum()
        print("\n------------------------------------")
        print(f"全国合計営業所数: {total_offices} 箇所")
        print("------------------------------------")
        print(f"※ {datetime.now().strftime('%Y年%m月%d日')}時点の https://www.tamahome.jp/modelhouse/ のデータに基づきます。")

    elif df_offices is not None and df_offices.empty:
         print("\n営業所が見つかりませんでした。")
    else:
        print("\n営業所情報のスクレイピングに失敗しました。")