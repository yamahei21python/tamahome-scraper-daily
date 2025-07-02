# -*- coding: utf-8 -*-
"""
【GitHub Actions対応版】
タマホームの公式ウェブサイトから分譲物件のデータを取得し、
日別のフォルダを作成して、その中に3つのDataFrameをCSVファイルとして保存します。
（URLを右端に保存、不明物件削除機能付き）
"""

# ==============================================================================
# 必要なライブラリのインポート
# ==============================================================================
import requests
import pandas as pd
import re
import json
import time
import os
from datetime import datetime
from zoneinfo import ZoneInfo # Python 3.9以降で利用可能
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

# Google DriveのマウントはGitHub Actionsでは不要なので削除
# from google.colab import drive
# drive.mount('/content/drive')


# ==============================================================================
# 定数定義
# ==============================================================================
BASE_URL = "https://www.tamahome.jp/kodatebunjyo/"
AREA_PATHS = [
    "hokkaido/", "tohoku/", "hokuriku/", "kanto/", "tokai/nagano/",
    "tokai/aichi/", "tokai/gifu/", "tokai/shizuoka/", "tokai/mie/",
    "kinki/", "chugoku/", "shikoku/", "kyushu/"
]
ROMAJI_TO_PREF_MAP = {
    'hokkaido': '北海道', 'aomori': '青森県', 'iwate': '岩手県', 'miyagi': '宮城県', 'akita': '秋田県',
    'yamagata': '山形県', 'fukushima': '福島県', 'ibaraki': '茨城県', 'tochigi': '栃木県', 'gunma': '群馬県',
    'saitama': '埼玉県', 'chiba': '千葉県', 'tokyo': '東京都', 'kanagawa': '神奈川県', 'niigata': '新潟県',
    'toyama': '富山県', 'ishikawa': '石川県', 'fukui': '福井県', 'yamanashi': '山梨県', 'nagano': '長野県',
    'gifu': '岐阜県', 'shizuoka': '静岡県', 'aichi': '愛知県', 'mie': '三重県', 'shiga': '滋賀県',
    'kyoto': '京都府', 'osaka': '大阪府', 'hyogo': '兵庫県', 'nara': '奈良県', 'wakayama': '和歌山県',
    'tottori': '鳥取県', 'shimane': '島根県', 'okayama': '岡山県', 'hiroshima': '広島県', 'yamaguchi': '山口県',
    'tokushima': '徳島県', 'kagawa': '香川県', 'ehime': '愛媛県', 'kochi': '高知県', 'fukuoka': '福岡県',
    'saga': '佐賀県', 'nagasaki': '長崎県', 'kumamoto': '熊本県', 'oita': '大分県', 'miyazaki': '宮崎県',
    'kagoshima': '鹿児島県', 'okinawa': '沖縄県'
}
PREFECTURE_ORDER = list(ROMAJI_TO_PREF_MAP.values())
PREF_REGEX_PATTERN = "|".join(p for p in PREFECTURE_ORDER)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
REQUEST_DELAY = 0.2


# ==============================================================================
# データ整形・変換関数
# ==============================================================================
def clean_completion_date(date_str: Any) -> str:
    if not isinstance(date_str, str): return '情報なし'
    dates_found = re.findall(r'(\d{4})[年|月]\s*(\d{1,2})月', date_str)
    if dates_found:
        latest_date = None
        for year, month in dates_found:
            try:
                current_date = datetime(int(year), int(month), 1)
                if latest_date is None or current_date > latest_date: latest_date = current_date
            except ValueError: continue
        if latest_date: return latest_date.strftime('%Y%m')
    if any(keyword in date_str for keyword in ['完成済', '完成済み']): return '完成済'
    return '情報なし'

def determine_property_attribute(completion_info: str, jst_now: datetime) -> str:
    if completion_info == "完成済": return "新築"
    if not completion_info.isdigit(): return "不明"
    try:
        completion_date = datetime.strptime(completion_info, '%Y%m')
    except ValueError: return "不明"
    is_older_than_one_year = (jst_now.year - completion_date.year) > 1 or \
                             ((jst_now.year - completion_date.year) == 1 and jst_now.month > completion_date.month)
    is_future = completion_date > jst_now.replace(tzinfo=None)
    if is_future: return "将来"
    elif is_older_than_one_year: return "中古"
    else: return "新築"

def clean_price(price_str: Any) -> str:
    if not isinstance(price_str, str): return '0'
    price_matches = re.findall(r'([\d,.]+)\s*万円', price_str)
    if not price_matches:
        temp_str = price_str.replace('　', ' ').replace('(税込)', '').replace('（税込）', '').strip()
        price_matches = re.findall(r'[\d,.]+', temp_str)
    if not price_matches: return '0'
    numbers = [int(p.replace(',', '').replace('.', '')) for p in price_matches]
    if not numbers: return '0'
    elif len(numbers) == 1: return f"{numbers[0]:,}"
    else: return f"{min(numbers):,} ～ {max(numbers):,}"

def calculate_average_price(price_range_str: str) -> int:
    if not isinstance(price_range_str, str): return 0
    numbers = [int(n.replace(',', '')) for n in re.findall(r'[\d,]+', price_range_str)]
    if not numbers: return 0
    return int((min(numbers) + max(numbers)) / 2)

def extract_first_number(text: Any) -> int:
    if not isinstance(text, str): return 0
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else 0


# ==============================================================================
# データ取得・スクレイピング関数
# ==============================================================================
def _fetch_property_list_from_areas(session: requests.Session) -> List[Dict[str, str]]:
    all_properties = []
    print("--- ステップ1: 全エリアから物件リストを収集中 ---")
    for area_path in AREA_PATHS:
        area_url = f"{BASE_URL}{area_path}"; area_name = area_path.strip('/').split('/')[-1]
        print(f"■ 対象エリア: {area_name}"); time.sleep(REQUEST_DELAY)
        try:
            res = session.get(area_url, headers=HEADERS); res.raise_for_status()
            match = re.compile(r"let items = JSON\.parse\('(.*?)'\);", re.DOTALL).search(res.text)
            if not match or not match.group(1):
                print(f" -> 物件リストが見つかりませんでした。"); continue
            properties = json.loads(match.group(1).encode().decode('unicode-escape'))
            all_properties.extend(properties)
        except (requests.RequestException, json.JSONDecodeError) as e: print(f" -> エラーが発生しました: {e}")
    unique_properties = list({prop['url']: prop for prop in all_properties if prop.get('url')}.values())
    print(f"\nユニークな物件数は {len(unique_properties)} 件です。")
    return unique_properties

def scrape_all_properties_details() -> Optional[pd.DataFrame]:
    with requests.Session() as session:
        property_list = _fetch_property_list_from_areas(session)
        if not property_list: return None
        print("\n--- ステップ2: 各物件の詳細ページから情報を取得 ---")
        detailed_results = []
        for i, prop_info in enumerate(property_list, 1):
            prop_name = prop_info.get('name', '物件名不明'); prop_url = prop_info.get('url', '').replace('\\', '')
            print(f"  - 処理中 ({i}/{len(property_list)}): {prop_name}"); time.sleep(REQUEST_DELAY)
            try:
                pref_name = '不明'
                try:
                    path_parts = prop_url.split('/'); bunjyo_index = path_parts.index('kodatebunjyo')
                    romaji_pref = path_parts[bunjyo_index + 1]
                    if romaji_pref in ROMAJI_TO_PREF_MAP: pref_name = ROMAJI_TO_PREF_MAP[romaji_pref]
                except (ValueError, IndexError): pass
                if pref_name == '不明':
                    address_from_json = prop_info.get('address', '')
                    if pref_match := re.search(PREF_REGEX_PATTERN, address_from_json): pref_name = pref_match.group(0)
                res = session.get(prop_url, headers=HEADERS); res.raise_for_status()
                soup = BeautifulSoup(res.content, 'html.parser')
                outline_heading = soup.find(lambda tag: tag.name in ['h2', 'h3'] and '物件概要' in tag.get_text())
                def get_info(table: Optional[BeautifulSoup], header_text: str) -> str:
                    if not table: return '情報なし'
                    header_tag = table.find('th', string=re.compile(header_text))
                    return header_tag.find_next_sibling('td').get_text(strip=True) if header_tag and header_tag.find_next_sibling('td') else '情報なし'

                base_data = {"都道府県": pref_name, "物件名": prop_name, "URL": prop_url}

                if not outline_heading:
                    print("    -> 物件概要が見つからないため、デフォルト値で登録します。")
                    default_data = {"完成時期": "情報なし", "総戸数（または総区画数）": "0", "販売戸数（または販売区画数）": "0", "価格(税込)": "0"}
                    detailed_results.append({**base_data, **default_data})
                    continue
                outline_table = outline_heading.find_next('table')
                if pref_name == '不明' and outline_table:
                    for key in ['所在地', 'お問合せ']:
                        address_from_table = get_info(outline_table, key)
                        if new_pref_match := re.search(PREF_REGEX_PATTERN, address_from_table):
                            pref_name = new_pref_match.group(0); base_data["都道府県"] = pref_name; break

                scraped_data = {
                    "完成時期": get_info(outline_table, '完成時期'),
                    "総戸数（または総区画数）": get_info(outline_table, r'総戸数|総区画数'),
                    "販売戸数（または販売区画数）": get_info(outline_table, r'販売戸数|販売区画数'),
                    "価格(税込)": get_info(outline_table, r'価格\(税込\)')
                }
                detailed_results.append({**base_data, **scraped_data})
            except requests.RequestException as e: print(f"    -> エラー: {prop_url} の取得に失敗しました: {e}")
    if not detailed_results: return None
    print("\n--- ステップ3: 取得結果の整形 ---")
    return pd.DataFrame(detailed_results)


# ==============================================================================
# データ処理と表示
# ==============================================================================
def create_and_process_dataframes(raw_df: pd.DataFrame, jst_now: datetime) -> tuple:
    print("\n--- ステップ4: 詳細リストの作成と整形 ---")
    df = raw_df.copy(); df['物件名'] = df['物件名'].str.replace('タマタウン', '').str.strip()
    df['完成時期'] = df['完成時期'].apply(clean_completion_date); df['属性'] = df['完成時期'].apply(determine_property_attribute, args=(jst_now,))
    df['価格'] = df['価格(税込)'].apply(clean_price); df['総戸数'] = df['総戸数（または総区画数）'].apply(extract_first_number)
    df['販売戸数'] = df['販売戸数（または販売区画数）'].apply(extract_first_number)
    print("-> 総戸数と販売戸数の論理チェックと修正を実行...")
    swap_mask = df['総戸数'] < df['販売戸数']
    df.loc[swap_mask, ['総戸数', '販売戸数']] = df.loc[swap_mask, ['販売戸数', '総戸数']].values
    df['価格（平均）'] = df['価格'].apply(calculate_average_price)

    final_columns = ['都道府県', '物件名', '属性', '完成時期', '総戸数', '販売戸数', '価格（平均）', '価格', 'URL']
    detailed_list_df = df[final_columns]

    existing_prefs = [p for p in PREFECTURE_ORDER if p in detailed_list_df['都道府県'].unique()]
    detailed_list_df['都道府県'] = pd.Categorical(detailed_list_df['都道府県'], categories=existing_prefs, ordered=True)
    detailed_list_df = detailed_list_df.sort_values(by=["都道府県", "完成時期"]).reset_index(drop=True)

    print("\n--- 全物件詳細リスト（フィルタリング前）---")
    # GitHub Actionsではdisplayは使えないのでprintで代用
    print(detailed_list_df.head(10).to_string()) # 先頭10行を表示

    print("\n\n--- ステップ5: サマリーテーブルの作成 ---")
    df_filtered = detailed_list_df[(detailed_list_df['都道府県'] != '不明') & (detailed_list_df['属性'] != '不明')].copy()

    # 都道府県別サマリー
    summary_df = df_filtered.copy(); summary_df['価格x販売戸数'] = summary_df['価格（平均）'] * summary_df['販売戸数']
    prefecture_summary_df = summary_df.groupby(['都道府県', '属性'], observed=True).agg(総戸数=('総戸数', 'sum'),販売戸数=('販売戸数', 'sum'),価格x販売戸数_合計=('価格x販売戸数', 'sum')).reset_index()
    prefecture_summary_df['価格（平均）'] = prefecture_summary_df.apply(lambda r: r['価格x販売戸数_合計']/r['販売戸数'] if r['販売戸数']>0 else 0, axis=1).astype(int)
    prefecture_summary_df['売れ残り率'] = prefecture_summary_df.apply(lambda r: f"{(r['販売戸数']/r['総戸数']*100):.1f}%" if r['総戸数']>0 else "0.0%", axis=1)
    prefecture_summary_df = prefecture_summary_df[['都道府県', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）']]

    # 月次サマリー
    monthly_df = df_filtered[df_filtered['完成時期'].str.isdigit()].copy()
    if not monthly_df.empty:
        monthly_df['価格x販売戸数'] = monthly_df['価格（平均）'] * monthly_df['販売戸数']
        monthly_summary_df = monthly_df.groupby(['完成時期', '属性']).agg(総戸数=('総戸数', 'sum'),販売戸数=('販売戸数', 'sum'),価格x販売戸数_合計=('価格x販売戸数', 'sum')).reset_index()
        monthly_summary_df['価格（平均）'] = monthly_summary_df.apply(lambda r: r['価格x販売戸数_合計']/r['販売戸数'] if r['販売戸数']>0 else 0, axis=1).astype(int)
        monthly_summary_df['売れ残り率'] = monthly_summary_df.apply(lambda r: f"{(r['販売戸数']/r['総戸数']*100):.1f}%" if r['総戸数']>0 else "0.0%", axis=1)
        monthly_summary_df = monthly_summary_df[['完成時期', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）']]
    else: monthly_summary_df = pd.DataFrame()

    print("-> サマリーテーブルを表示します。")
    print(prefecture_summary_df.head().to_string())
    print(monthly_summary_df.head().to_string())

    return df_filtered, prefecture_summary_df, monthly_summary_df

# ==============================================================================
# ファイルへの保存関数
# ==============================================================================
def save_dataframes_to_local(detailed_df, prefecture_summary_df, monthly_summary_df, base_save_path, jst_now):
    date_str = jst_now.strftime('%Y%m%d')
    daily_folder_path = os.path.join(base_save_path, date_str)
    print(f"\n\n--- ステップ6: DataFrameをローカルフォルダに保存 ---")
    print(f"保存先フォルダ: {daily_folder_path}")
    os.makedirs(daily_folder_path, exist_ok=True)
    try:
        path_detailed = os.path.join(daily_folder_path, f'tamahome_detailed_list_{date_str}.csv')
        path_pref_summary = os.path.join(daily_folder_path, f'tamahome_prefecture_summary_{date_str}.csv')
        path_month_summary = os.path.join(daily_folder_path, f'tamahome_monthly_summary_{date_str}.csv')
        detailed_df.to_csv(path_detailed, index=False, encoding='utf-8-sig')
        print(f"-> 詳細リストを '{path_detailed}' として保存しました。")
        if not prefecture_summary_df.empty:
            prefecture_summary_df.to_csv(path_pref_summary, index=False, encoding='utf-8-sig')
            print(f"-> 都道府県別サマリーを '{path_pref_summary}' として保存しました。")
        else: print("-> 都道府県別サマリーは空のため、保存をスキップしました。")
        if not monthly_summary_df.empty:
            monthly_summary_df.to_csv(path_month_summary, index=False, encoding='utf-8-sig')
            print(f"-> 月次サマリーを '{path_month_summary}' として保存しました。")
        else: print("-> 月次サマリーは空のため、保存をスキップしました。")
    except Exception as e: print(f"ファイルへの保存中にエラーが発生しました: {e}")

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    """スクリプト全体の処理を実行するメイン関数。"""

    # 保存パスをGitHub Actionsの実行環境内の相対パスに変更
    BASE_SAVE_PATH = './TamaHome_CSV_Data'
    jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
    pd.set_option('display.max_rows', 10)

    raw_property_df = scrape_all_properties_details()
    if raw_property_df is None or raw_property_df.empty:
        print("\n物件データが取得できませんでした。処理を終了します。")
        return

    detailed_df, prefecture_summary_df, monthly_summary_df = create_and_process_dataframes(raw_property_df, jst_now)

    print("\n--- フィルタリング後の最終データ ---")
    print(f"-> クリーンな物件データ: {len(detailed_df)} 件")
    print(detailed_df.head().to_string())

    save_dataframes_to_local(
        detailed_df,
        prefecture_summary_df,
        monthly_summary_df,
        BASE_SAVE_PATH,
        jst_now
    )
    print("\n\n" + "*"*60 + "\nすべての処理が正常に完了しました。\n" + "*"*60)

# スクリプトを実行
if __name__ == '__main__':
    main()