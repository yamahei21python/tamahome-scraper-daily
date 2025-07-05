# -*- coding: utf-8 -*-
"""
タマホーム物件情報自動取得・データベース保存スクリプト

タマホームの公式ウェブサイトから「分譲物件データ」と「営業所データ」を
スクレイピングし、指定のデータベースに保存します。

- 分譲物件データ: 毎日取得し、3つのテーブルに分けて保存。
- 営業所データ: 毎月1日、またはデータが存在しない場合に取得・更新。

実行には環境変数 `DATABASE_URL` が必要です。
(例: `postgresql://user:password@host:port/dbname`)
"""

import os
import re
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text

# --- 環境依存ライブラリのインポート ---
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8以前の場合
    from backports.zoneinfo import ZoneInfo

# Colab/Jupyter環境でのみ利用可能なライブラリ
try:
    from IPython.display import display
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    # display関数の代替を定義
    def display(df):
        print(df)

# ==============================================================================
# 1. 定数定義
# ==============================================================================
PROPERTY_BASE_URL = "https://www.tamahome.jp/kodatebunjyo/"
OFFICE_BASE_URL = "https://www.tamahome.jp/modelhouse/"
AREA_PATHS = [
    "hokkaido/", "tohoku/", "hokuriku/", "kanto/", "tokai/nagano/", "tokai/aichi/",
    "tokai/gifu/", "tokai/shizuoka/", "tokai/mie/", "kinki/", "chugoku/", "shikoku/", "kyushu/"
]
ROMAJI_TO_PREF_MAP = {
    'hokkaido': '北海道', 'aomori': '青森県', 'iwate': '岩手県', 'miyagi': '宮城県', 'akita': '秋田県', 'yamagata': '山形県',
    'fukushima': '福島県', 'ibaraki': '茨城県', 'tochigi': '栃木県', 'gunma': '群馬県', 'saitama': '埼玉県', 'chiba': '千葉県',
    'tokyo': '東京都', 'kanagawa': '神奈川県', 'niigata': '新潟県', 'toyama': '富山県', 'ishikawa': '石川県', 'fukui': '福井県',
    'yamanashi': '山梨県', 'nagano': '長野県', 'gifu': '岐阜県', 'shizuoka': '静岡県', 'aichi': '愛知県', 'mie': '三重県',
    'shiga': '滋賀県', 'kyoto': '京都府', 'osaka': '大阪府', 'hyogo': '兵庫県', 'nara': '奈良県', 'wakayama': '和歌山県',
    'tottori': '鳥取県', 'shimane': '島根県', 'okayama': '岡山県', 'hiroshima': '広島県', 'yamaguchi': '山口県', 'tokushima': '徳島県',
    'kagawa': '香川県', 'ehime': '愛媛県', 'kochi': '高知県', 'fukuoka': '福岡県', 'saga': '佐賀県', 'nagasaki': '長崎県',
    'kumamoto': '熊本県', 'oita': '大分県', 'miyazaki': '宮崎県', 'kagoshima': '鹿児島県', 'okinawa': '沖縄県'
}
PREFECTURE_ORDER = list(ROMAJI_TO_PREF_MAP.values())
PREF_REGEX_PATTERN = "|".join(p for p in PREFECTURE_ORDER)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
REQUEST_DELAY = 0.2

# ==============================================================================
# 2. データ整形・変換関数
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
        is_older_than_one_year = (jst_now.year - completion_date.year) > 1 or \
                                 ((jst_now.year - completion_date.year) == 1 and jst_now.month > completion_date.month)
        is_future = completion_date > jst_now.replace(tzinfo=None)
        if is_future: return "将来"
        elif is_older_than_one_year: return "中古"
        else: return "新築"
    except ValueError: return "不明"

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
# 3. データ取得・スクレイピング関数
# ==============================================================================
def scrape_all_properties_details() -> Optional[pd.DataFrame]:
    # (この関数の内容は変更ないため、コードの簡潔化のため省略)
    # ... (元のコードをここに配置)
    with requests.Session() as session:
        all_properties = []
        print("--- Step 1: Collecting property lists from all areas ---")
        for area_path in AREA_PATHS:
            area_url = f"{PROPERTY_BASE_URL}{area_path}"
            print(f"■ Target Area: {area_path.strip('/')}")
            time.sleep(REQUEST_DELAY)
            try:
                res = session.get(area_url, headers=HEADERS)
                res.raise_for_status()
                match = re.compile(r"let items = JSON\.parse\('(.*?)'\);", re.DOTALL).search(res.text)
                if not match or not match.group(1): continue
                properties = json.loads(match.group(1).encode().decode('unicode-escape'))
                all_properties.extend(properties)
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f" -> Error: {e}")
        unique_properties = list({prop['url']: prop for prop in all_properties if prop.get('url')}.values())
        print(f"\nFound {len(unique_properties)} unique properties.")
        if not unique_properties: return None
        print("\n--- Step 2: Scraping details from each property page ---")
        detailed_results = []
        for i, prop_info in enumerate(unique_properties, 1):
            prop_name = prop_info.get('name', 'N/A')
            prop_url = prop_info.get('url', '').replace('\\', '')
            print(f"  - Processing ({i}/{len(unique_properties)}): {prop_name}")
            time.sleep(REQUEST_DELAY)
            try:
                pref_name = '不明'
                try:
                    path_parts = prop_url.split('/')
                    bunjyo_index = path_parts.index('kodatebunjyo')
                    romaji_pref = path_parts[bunjyo_index + 1]
                    if romaji_pref in ROMAJI_TO_PREF_MAP: pref_name = ROMAJI_TO_PREF_MAP[romaji_pref]
                except (ValueError, IndexError): pass
                if pref_name == '不明':
                    address_from_json = prop_info.get('address', '')
                    if pref_match := re.search(PREF_REGEX_PATTERN, address_from_json): pref_name = pref_match.group(0)
                res = session.get(prop_url, headers=HEADERS)
                res.raise_for_status()
                soup = BeautifulSoup(res.content, 'html.parser')
                outline_heading = soup.find(lambda tag: tag.name in ['h2', 'h3'] and '物件概要' in tag.get_text())
                def get_info(table: Optional[BeautifulSoup], header_text: str) -> str:
                    if not table: return '情報なし'
                    header_tag = table.find('th', string=re.compile(header_text))
                    return header_tag.find_next_sibling('td').get_text(strip=True) if header_tag and header_tag.find_next_sibling('td') else '情報なし'
                base_data = {"都道府県": pref_name, "物件名": prop_name, "URL": prop_url}
                if not outline_heading:
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
            except requests.RequestException as e: print(f"    -> Error scraping {prop_url}: {e}")
    if not detailed_results: return None
    print("\n--- Step 3: Formatting results ---")
    return pd.DataFrame(detailed_results)

def scrape_tamahome_offices() -> Optional[pd.DataFrame]:
    # (この関数の内容は変更ないため、コードの簡潔化のため省略)
    # ... (元のコードをここに配置)
    print("Scraping office information...")
    try:
        response = requests.get(OFFICE_BASE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
    except requests.exceptions.RequestException as e:
        print(f"Error accessing site: {e}")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    link_list_div = soup.find('div', class_='link-list')
    if not link_list_div:
        print("Error: Could not find prefecture link list.")
        return None
    prefecture_links = [{"name": link.get_text(strip=True), "url": urljoin(OFFICE_BASE_URL, link.get('href'))}
                        for link in link_list_div.find_all('a') if link.get('href')]
    office_data = []
    print(f"Aggregating office counts for {len(prefecture_links)} prefectures...")
    for i, pref in enumerate(prefecture_links, 1):
        try:
            time.sleep(REQUEST_DELAY)
            page_response = requests.get(pref['url'], headers=HEADERS, timeout=10)
            page_response.raise_for_status()
            page_response.encoding = page_response.apparent_encoding
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            count = len(page_soup.find_all('div', class_='c-modelhouse'))
            office_data.append({"都道府県": pref['name'], "営業所数": count})
        except requests.exceptions.RequestException: continue
    if not office_data: return None
    df = pd.DataFrame(office_data)
    return df[df['営業所数'] > 0].copy()

# ==============================================================================
# 4. データ処理 (3つのDataFrameを生成)
# ==============================================================================
def create_and_process_dataframes(raw_df: pd.DataFrame, jst_now: datetime) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # (この関数の内容は変更ないため、コードの簡潔化のため省略)
    # ... (元のコードをここに配置)
    print("\n--- Step 4: Creating and formatting detailed list and summaries ---")
    df = raw_df.copy()
    df['物件名'] = df['物件名'].str.replace('タマタウン', '').str.strip()
    df['完成時期'] = df['完成時期'].apply(clean_completion_date)
    df['属性'] = df['完成時期'].apply(determine_property_attribute, args=(jst_now,))
    df['価格'] = df['価格(税込)'].apply(clean_price)
    df['総戸数'] = df['総戸数（または総区画数）'].apply(extract_first_number)
    df['販売戸数'] = df['販売戸数（または販売区画数）'].apply(extract_first_number)
    swap_mask = df['総戸数'] < df['販売戸数']
    df.loc[swap_mask, ['総戸数', '販売戸数']] = df.loc[swap_mask, ['販売戸数', '総戸数']].values
    df['価格（平均）'] = df['価格'].apply(calculate_average_price)
    final_columns = ['都道府県', '物件名', '属性', '完成時期', '総戸数', '販売戸数', '価格（平均）', '価格', 'URL']
    detailed_df_base = df[final_columns]
    detailed_df = detailed_df_base[(detailed_df_base['都道府県'] != '不明') & (detailed_df_base['属性'] != '不明')].copy()

    # --- 都道府県別サマリー ---
    summary_df = detailed_df.copy()
    summary_df['価格x販売戸数'] = summary_df['価格（平均）'] * summary_df['販売戸数']
    prefecture_summary_df = summary_df.groupby(['都道府県', '属性']).agg(
        総戸数=('総戸数', 'sum'), 販売戸数=('販売戸数', 'sum'), 価格x販売戸数_合計=('価格x販売戸数', 'sum')).reset_index()
    prefecture_summary_df['価格（平均）'] = prefecture_summary_df.apply(lambda r: r['価格x販売戸数_合計']/r['販売戸数'] if r['販売戸数']>0 else 0, axis=1).astype(int)
    prefecture_summary_df['売れ残り率'] = prefecture_summary_df.apply(lambda r: f"{(r['販売戸数']/r['総戸数']*100):.1f}%" if r['総戸数']>0 else "0.0%", axis=1)
    prefecture_summary_df = prefecture_summary_df[['都道府県', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）']]

    # --- 月次サマリー ---
    monthly_df = detailed_df[detailed_df['完成時期'].str.isdigit()].copy()
    if not monthly_df.empty:
        monthly_df['価格x販売戸数'] = monthly_df['価格（平均）'] * monthly_df['販売戸数']
        monthly_summary_df = monthly_df.groupby(['完成時期', '属性']).agg(
            総戸数=('総戸数', 'sum'), 販売戸数=('販売戸数', 'sum'), 価格x販売戸数_合計=('価格x販売戸数', 'sum')).reset_index()
        monthly_summary_df['価格（平均）'] = monthly_summary_df.apply(lambda r: r['価格x販売戸数_合計']/r['販売戸数'] if r['販売戸数']>0 else 0, axis=1).astype(int)
        monthly_summary_df['売れ残り率'] = monthly_summary_df.apply(lambda r: f"{(r['販売戸数']/r['総戸数']*100):.1f}%" if r['総戸数']>0 else "0.0%", axis=1)
        monthly_summary_df = monthly_summary_df[['完成時期', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）']]
    else:
        monthly_summary_df = pd.DataFrame(columns=['完成時期', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）'])

    print("-> DataFrames generated successfully.")
    return detailed_df, prefecture_summary_df, monthly_summary_df

# ==============================================================================
# 5. データベース保存関数
# ==============================================================================
def save_data_to_db(connection, df: pd.DataFrame, table_name: str, jst_now: datetime, unique_keys: list):
    # (この関数の内容は変更ないため、コードの簡潔化のため省略)
    # ... (元のコードをここに配置)
    """汎用DB保存関数"""
    if df.empty:
        print(f"-> No data to save for '{table_name}'. Skipping.")
        return 0
    df_to_save = df.copy()
    df_to_save['scrape_date'] = jst_now.date()
    COLUMN_MAP = {
        '都道府県': 'prefecture', '物件名': 'property_name', '属性': 'attribute', '完成時期': 'completion_period',
        '総戸数': 'total_units', '販売戸数': 'units_for_sale', '価格（平均）': 'price_avg', '価格': 'price_range',
        'URL': 'url', '売れ残り率': 'unsold_rate'
    }
    df_to_save.rename(columns=COLUMN_MAP, inplace=True)
    if 'unsold_rate' in df_to_save.columns:
        df_to_save['unsold_rate'] = df_to_save['unsold_rate'].str.replace('%', '').astype(float)
    all_db_cols = [
        'scrape_date', 'prefecture', 'property_name', 'attribute', 'completion_period', 'total_units',
        'units_for_sale', 'price_avg', 'price_range', 'url', 'unsold_rate'
    ]
    cols_to_insert = [col for col in all_db_cols if col in df_to_save.columns]
    df_to_save = df_to_save[cols_to_insert]
    records = df_to_save.to_dict(orient='records')
    stmt = text(f"""
        INSERT INTO {table_name} ({', '.join(cols_to_insert)})
        VALUES ({', '.join([f':{c}' for c in cols_to_insert])})
        ON CONFLICT ({', '.join(unique_keys)}) DO NOTHING;
    """)
    result = connection.execute(stmt, records)
    print(f"-> Saved {result.rowcount} new rows to '{table_name}'.")
    return result.rowcount

def save_offices_to_db(connection, df_offices: pd.DataFrame, jst_now: datetime):
    # (この関数の内容は変更ないため、コードの簡潔化のため省略)
    # ... (元のコードをここに配置)
    """営業所情報をDBに保存する（Upsert）"""
    if df_offices.empty: return
    df_to_save = df_offices.copy()
    df_to_save.rename(columns={'都道府県': 'prefecture', '営業所数': 'office_count'}, inplace=True)
    df_to_save['updated_at'] = jst_now
    records = df_to_save.to_dict(orient='records')
    stmt = text("""
        INSERT INTO tamahome_offices (prefecture, office_count, updated_at)
        VALUES (:prefecture, :office_count, :updated_at)
        ON CONFLICT (prefecture) DO UPDATE SET
            office_count = EXCLUDED.office_count,
            updated_at = EXCLUDED.updated_at;
    """)
    result = connection.execute(stmt, records)
    print(f"-> Upserted {result.rowcount} rows in 'tamahome_offices'.")

# ==============================================================================
# 6. メイン処理
# ==============================================================================
def get_database_url():
    """環境変数またはColabシークレットからDATABASE_URLを取得する。"""
    db_url = None
    # Colab環境のシークレットを優先
    if IS_COLAB:
        try:
            from google.colab import userdata
            db_url = userdata.get('DATABASE_URL')
        except (ImportError, KeyError):
            pass
    # Colabシークレットにない場合、または非Colab環境の場合は環境変数を試す
    if not db_url:
        db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        raise ValueError("データベース接続URLが見つかりません。"
                         "環境変数 'DATABASE_URL' を設定するか、"
                         "Colabの場合はシークレットに 'DATABASE_URL' を設定してください。")
    return db_url

def main(args):
    """スクリプト全体の処理を実行するメイン関数。"""
    try:
        DATABASE_URL = get_database_url()
        jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
        pd.set_option('display.max_rows', 10)

        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("✅ Database connection successful.")

            if args.setup_db:
                print("\n--- [Setup Mode] Checking and creating tables if they don't exist ---")
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS tamahome_offices (
                        prefecture VARCHAR(10) PRIMARY KEY, office_count INTEGER NOT NULL, updated_at TIMESTAMPTZ NOT NULL );
                    CREATE TABLE IF NOT EXISTS tamahome_properties_detailed (
                        id SERIAL PRIMARY KEY, scrape_date DATE NOT NULL, url TEXT NOT NULL, prefecture VARCHAR(10), property_name TEXT,
                        attribute VARCHAR(10), completion_period VARCHAR(10), total_units INTEGER, units_for_sale INTEGER,
                        price_avg BIGINT, price_range TEXT, UNIQUE (scrape_date, url) );
                    CREATE TABLE IF NOT EXISTS tamahome_summary_prefecture (
                        id SERIAL PRIMARY KEY, scrape_date DATE NOT NULL, prefecture VARCHAR(10) NOT NULL, attribute VARCHAR(10) NOT NULL,
                        total_units INTEGER, units_for_sale INTEGER, unsold_rate REAL, price_avg BIGINT,
                        UNIQUE (scrape_date, prefecture, attribute) );
                    CREATE TABLE IF NOT EXISTS tamahome_summary_monthly (
                        id SERIAL PRIMARY KEY, scrape_date DATE NOT NULL, completion_period VARCHAR(10) NOT NULL, attribute VARCHAR(10) NOT NULL,
                        total_units INTEGER, units_for_sale INTEGER, unsold_rate REAL, price_avg BIGINT,
                        UNIQUE (scrape_date, completion_period, attribute) );
                """))
                connection.commit()
                print("-> Tables are ready.")
                return # セットアップモードの場合はここで終了
            
            # --- 営業所情報の取得処理 ---
            print("\n" + "="*80 + "\n--- Part 1: Office Information ---")
            office_count_in_db = connection.execute(text("SELECT COUNT(*) FROM tamahome_offices")).scalar()
            if args.force_update or jst_now.day == 1 or office_count_in_db == 0:
                if args.force_update: print("Forced update requested. Updating office data.")
                elif jst_now.day == 1: print("It's the 1st of the month. Updating office data.")
                else: print("Office data not found in DB. Fetching new data.")
                df_offices = scrape_tamahome_offices()
                if df_offices is not None and not df_offices.empty:
                    save_offices_to_db(connection, df_offices, jst_now)
                    display(df_offices)
            else:
                print("Office data already exists and it's not the 1st of the month. Skipping.")

            # --- 分譲物件情報の取得と保存 ---
            print("\n" + "="*80 + "\n--- Part 2: Property Information ---")
            raw_property_df = scrape_all_properties_details()
            if raw_property_df is None or raw_property_df.empty:
                print("\nCould not retrieve property data. Terminating.")
                return

            detailed_df, pref_summary_df, month_summary_df = create_and_process_dataframes(raw_property_df, jst_now)

            print("\n--- Displaying a sample of generated data ---")
            print("\n[Detailed List]")
            display(detailed_df.head())
            print("\n[Prefecture Summary]")
            display(pref_summary_df.head())
            print("\n[Monthly Summary]")
            display(month_summary_df.head())
            
            print("\n--- Step 5: Saving data to database ---")
            save_data_to_db(connection, detailed_df, 'tamahome_properties_detailed', jst_now, ['scrape_date', 'url'])
            save_data_to_db(connection, pref_summary_df, 'tamahome_summary_prefecture', jst_now, ['scrape_date', 'prefecture', 'attribute'])
            save_data_to_db(connection, month_summary_df, 'tamahome_summary_monthly', jst_now, ['scrape_date', 'completion_period', 'attribute'])
            
            connection.commit()
            print("\n✅ All data has been successfully saved to the database.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        return

    print("\n\n" + "*"*60 + "\nAll processes completed successfully.\n" + "*"*60)

# ==============================================================================
# 7. スクリプト実行エントリーポイント
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape property data from TamaHome and save to a database.')
    parser.add_argument(
        '--setup-db',
        action='store_true',
        help='Only creates database tables if they do not exist and then exits.'
    )
    parser.add_argument(
        '--force-update',
        action='store_true',
        help='Forces an update of the office information, ignoring the day of the month.'
    )
    args = parser.parse_args()
    
    main(args)
