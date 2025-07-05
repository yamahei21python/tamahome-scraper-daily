# -*- coding: utf-8 -*-
"""
【統合版】タマホーム物件情報自動取得・比較分析・DB保存スクリプト

タマホームの公式ウェブサイトからデータを取得し、データベースに保存します。
さらに、前日のデータと比較して日次レポートを生成し、それもDBに保存します。

- データ取得: 物件詳細、サマリー、営業所情報を取得。
- 比較分析: 最新日と前日のデータを比較し、新規・終了・更新物件を特定。
- レポート保存: 分析結果を日次レポートとして専用テーブルに保存。
"""

# ==============================================================================
# 1. 必要なライブラリのインポート
# ==============================================================================
import os
import re
import json
import time
import argparse
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text

# --- 環境依存ライブラリ ---
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8以前の場合
    from backports.zoneinfo import ZoneInfo

try:
    from IPython.display import display, HTML
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    def display(df):
        # DataFrameでない場合はそのままprint
        if isinstance(df, pd.DataFrame):
            print(df.to_string())
        else:
            print(df)
            
    def HTML(s):
        # HTMLタグを除去して簡易的に表示
        clean_s = re.sub('<.*?>', '', s)
        print(clean_s)


# ==============================================================================
# 2. 定数定義
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
# 3. データ整形・変換関数
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
# 4. データ取得 & 処理関数
# ==============================================================================
def scrape_all_properties_details() -> Optional[pd.DataFrame]:
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
    from urllib.parse import urljoin
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

def create_and_process_dataframes(raw_df: pd.DataFrame, jst_now: datetime) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
# 5. DB保存 & 比較分析関数
# ==============================================================================
def save_data_to_db(connection, df: pd.DataFrame, table_name: str, jst_now: datetime, unique_keys: list):
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

def analyze_and_save_daily_report(connection, jst_now: datetime):
    """前日データと比較し、分析レポートを生成してDBに保存する"""
    print("\n" + "="*80)
    print("--- Part 3: Daily Comparison Analysis ---")
    
    today = jst_now.date()
    # DBから最新と2番目に新しい日付を取得
    date_query = text("SELECT DISTINCT scrape_date FROM tamahome_properties_detailed ORDER BY scrape_date DESC LIMIT 2;")
    result = connection.execute(date_query).fetchall()

    if len(result) < 2 or result[0][0] != today:
        print("-> Not enough data for comparison. Skipping report generation.")
        return

    latest_date, previous_date = result[0][0], result[1][0]
    print(f"Comparing data between {previous_date} and {latest_date}...")

    # データを読み込み
    df_latest = pd.read_sql(text("SELECT * FROM tamahome_properties_detailed WHERE scrape_date = :date"), connection, params={'date': latest_date})
    df_previous = pd.read_sql(text("SELECT * FROM tamahome_properties_detailed WHERE scrape_date = :date"), connection, params={'date': previous_date})
    
    # 複合キーでマージ
    df_latest['composite_key'] = df_latest['prefecture'] + '_' + df_latest['property_name'] + '_' + df_latest['url']
    df_previous['composite_key'] = df_previous['prefecture'] + '_' + df_previous['property_name'] + '_' + df_previous['url']
    merged_df = pd.merge(df_previous, df_latest, on='composite_key', how='outer', suffixes=('_prev', '_latest'), indicator=True)

    # レポート用データの準備リスト
    reports_to_save = []

    # 1. 新規物件
    new_properties = merged_df[merged_df['_merge'] == 'right_only']
    print(f"-> Found {len(new_properties)} new properties.")
    for _, row in new_properties.iterrows():
        content = {
            'prefecture': row['prefecture_latest'], 'property_name': row['property_name_latest'],
            'url': row['url_latest'], 'attribute': row['attribute_latest'],
            'price_range': row['price_range_latest'], 'units_for_sale': int(row['units_for_sale_latest']) if pd.notna(row['units_for_sale_latest']) else 0
        }
        reports_to_save.append({'report_date': today, 'report_type': 'new', 'content': json.dumps(content, ensure_ascii=False)})

    # 2. 掲載終了物件
    removed_properties = merged_df[merged_df['_merge'] == 'left_only']
    print(f"-> Found {len(removed_properties)} removed properties.")
    for _, row in removed_properties.iterrows():
        content = {
            'prefecture': row['prefecture_prev'], 'property_name': row['property_name_prev'],
            'url': row['url_prev'], 'attribute': row['attribute_prev'],
            'price_range': row['price_range_prev'], 'units_for_sale': int(row['units_for_sale_prev']) if pd.notna(row['units_for_sale_prev']) else 0
        }
        reports_to_save.append({'report_date': today, 'report_type': 'removed', 'content': json.dumps(content, ensure_ascii=False)})
        
    # 3. 更新物件
    updated_count = 0
    both_df = merged_df[merged_df['_merge'] == 'both'].copy()
    compare_cols = ['attribute', 'completion_period', 'total_units', 'units_for_sale', 'price_avg', 'price_range']
    for _, row in both_df.iterrows():
        updates = {}
        for col in compare_cols:
            val_prev, val_latest = row[f'{col}_prev'], row[f'{col}_latest']
            if pd.isna(val_prev) and pd.isna(val_latest): continue
            if str(val_prev) != str(val_latest):
                updates[col] = {'from': str(val_prev) if pd.notna(val_prev) else None, 'to': str(val_latest) if pd.notna(val_latest) else None}
        if updates:
            updated_count += 1
            content = {
                'prefecture': row['prefecture_latest'], 'property_name': row['property_name_latest'],
                'url': row['url_latest'], 'changes': updates
            }
            reports_to_save.append({'report_date': today, 'report_type': 'updated', 'content': json.dumps(content, ensure_ascii=False)})
    print(f"-> Found {updated_count} updated properties.")

    # DBにレポートを保存
    if reports_to_save:
        # 今日のレポートを一旦削除してからインサート（冪等性を保つため）
        connection.execute(text("DELETE FROM tamahome_daily_reports WHERE report_date = :today"), {'today': today})
        
        stmt = text("""
            INSERT INTO tamahome_daily_reports (report_date, report_type, content)
            VALUES (:report_date, :report_type, :content::jsonb)
        """)
        result = connection.execute(stmt, reports_to_save)
        print(f"-> Saved {result.rowcount} report entries to the database.")


# ==============================================================================
# 6. メイン処理
# ==============================================================================
def get_database_url():
    """環境変数またはColabシークレットからDATABASE_URLを取得する。"""
    db_url = None
    if IS_COLAB:
        try:
            from google.colab import userdata
            db_url = userdata.get('DATABASE_URL')
        except (ImportError, KeyError):
            pass
    if not db_url:
        db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        raise ValueError("DATABASE_URL not found. Please set it as an environment variable or a Colab secret.")
    return db_url

def setup_database_tables(connection):
    """データベースに必要なテーブルを作成する"""
    print("\n--- Checking and creating tables if they don't exist ---")
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
        CREATE TABLE IF NOT EXISTS tamahome_daily_reports (
            id SERIAL PRIMARY KEY,
            report_date DATE NOT NULL,
            report_type VARCHAR(20) NOT NULL,
            content JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_report_date ON tamahome_daily_reports (report_date);
    """))
    print("-> Tables are ready.")

def main(args):
    """スクリプト全体の処理を実行するメイン関数。"""
    try:
        DATABASE_URL = get_database_url()
        jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
        pd.set_option('display.max_rows', 10)

        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("✅ Database connection successful.")
            
            # DBセットアップモード
            if args.setup_db:
                setup_database_tables(connection)
                connection.commit()
                return

            # --- Part 1: データ取得 & 保存 ---
            print("\n" + "="*80 + "\n--- Part 1: Office Information ---")
            office_count_in_db = connection.execute(text("SELECT COUNT(*) FROM tamahome_offices")).scalar()
            if args.force_update or jst_now.day == 1 or office_count_in_db == 0:
                if args.force_update: print("Forced update requested. Updating office data.")
                elif jst_now.day == 1: print("It's the 1st of the month. Updating office data.")
                else: print("Office data not found in DB. Fetching new data.")
                df_offices = scrape_tamahome_offices()
                if df_offices is not None and not df_offices.empty:
                    save_offices_to_db(connection, df_offices, jst_now)
            else:
                print("Skipping office data update.")
            
            print("\n" + "="*80 + "\n--- Part 2: Property Information ---")
            raw_property_df = scrape_all_properties_details()
            if raw_property_df is None or raw_property_df.empty:
                print("\nCould not retrieve property data. Terminating.")
                return
            
            detailed_df, pref_summary_df, month_summary_df = create_and_process_dataframes(raw_property_df, jst_now)
            
            print("\n--- Displaying a sample of generated data ---")
            display(detailed_df.head())
            
            print("\n--- Saving data to database ---")
            save_data_to_db(connection, detailed_df, 'tamahome_properties_detailed', jst_now, ['scrape_date', 'url'])
            save_data_to_db(connection, pref_summary_df, 'tamahome_summary_prefecture', jst_now, ['scrape_date', 'prefecture', 'attribute'])
            save_data_to_db(connection, month_summary_df, 'tamahome_summary_monthly', jst_now, ['scrape_date', 'completion_period', 'attribute'])
            
            # --- Part 3: 比較分析 & レポート保存 ---
            analyze_and_save_daily_report(connection, jst_now)

            # --- 最終コミット ---
            print("\n" + "="*80)
            print("Committing all changes to the database...")
            connection.commit()
            print("✅ All data and reports have been successfully saved.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        return

    print("\n\n" + "*"*60 + "\nAll processes completed successfully.\n" + "*"*60)

# ==============================================================================
# 7. スクリプト実行エントリーポイント
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape, analyze, and save TamaHome property data.')
    parser.add_argument('--setup-db', action='store_true', help='Only creates/updates database tables and exits.')
    parser.add_argument('--force-update', action='store_true', help='Forces an update of the office information.')
    args = parser.parse_args()
    main(args)
