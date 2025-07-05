# -*- coding: utf-8 -*-
"""
【完全統合版 / エラー修正済】タマホーム物件情報 自動処理スクリプト

データ取得、比較分析、PDFレポート生成を一つのスクリプトで実行します。
コマンドライン引数により、実行するタスクを選択可能です。
"""

# ==============================================================================
# 1. 必要なライブラリのインポート
# ==============================================================================
import os
import re
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

# --- 主要ライブラリ ---
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text

# --- グラフ描画ライブラリ (冒頭でインポートする) ---
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image as PILImage
try:
    import japanize_matplotlib
except ImportError:
    print("Warning: japanize_matplotlib not found. Japanese characters may not display correctly.")

# --- 環境依存ライブラリ ---
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

try:
    from IPython.display import display, HTML
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

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
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
REQUEST_DELAY = 0.2
COLUMN_MAP_TO_JP = {
    'url': 'URL', 'prefecture': '都道府県', 'property_name': '物件名', 'attribute': '属性',
    'completion_period': '完成時期', 'total_units': '総戸数', 'units_for_sale': '販売戸数',
    'price_avg': '価格（平均）', 'price_range': '価格', 'office_count': '営業所数',
    'unsold_rate': '売れ残り率'
}

# ==============================================================================
# 3. ヘルパー関数
# ==============================================================================
def custom_display(df):
    if IS_COLAB:
        display(df)
    elif isinstance(df, pd.DataFrame):
        print(df.to_string())
    else:
        print(df)

def get_database_url():
    db_url = None
    if IS_COLAB:
        try: from google.colab import userdata; db_url = userdata.get('DATABASE_URL')
        except (ImportError, KeyError): pass
    if not db_url: db_url = os.environ.get('DATABASE_URL')
    if not db_url: raise ValueError("DATABASE_URL not found.")
    return db_url

# ==============================================================================
# 4. データ取得 & 処理モジュール
# ==============================================================================
class Scraper:
    # (このクラス内の関数は、これまでのスクリプトの関数とほぼ同じです)
    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
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

    @staticmethod
    def calculate_average_price(price_range_str: str) -> int:
        if not isinstance(price_range_str, str): return 0
        numbers = [int(n.replace(',', '')) for n in re.findall(r'[\d,]+', price_range_str)]
        if not numbers: return 0
        return int((min(numbers) + max(numbers)) / 2)
        
    @staticmethod
    def extract_first_number(text: Any) -> int:
        if not isinstance(text, str): return 0
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0

    @staticmethod
    def scrape_all_properties_details() -> Optional[pd.DataFrame]:
        from urllib.parse import urljoin
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
            PREF_REGEX_PATTERN = "|".join(p for p in PREFECTURE_ORDER)
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

    @staticmethod
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

    @classmethod
    def create_and_process_dataframes(cls, raw_df: pd.DataFrame, jst_now: datetime) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("\n--- Step 4: Creating and formatting detailed list and summaries ---")
        df = raw_df.copy()
        df['物件名'] = df['物件名'].str.replace('タマタウン', '').str.strip()
        df['完成時期'] = df['完成時期'].apply(cls.clean_completion_date)
        df['属性'] = df['完成時期'].apply(cls.determine_property_attribute, args=(jst_now,))
        df['価格'] = df['価格(税込)'].apply(cls.clean_price)
        df['総戸数'] = df['総戸数（または総区画数）'].apply(cls.extract_first_number)
        df['販売戸数'] = df['販売戸数（または販売区画数）'].apply(cls.extract_first_number)
        swap_mask = df['総戸数'] < df['販売戸数']
        df.loc[swap_mask, ['総戸数', '販売戸数']] = df.loc[swap_mask, ['販売戸数', '総戸数']].values
        df['価格（平均）'] = df['価格'].apply(cls.calculate_average_price)
        final_columns = ['都道府県', '物件名', '属性', '完成時期', '総戸数', '販売戸数', '価格（平均）', '価格', 'URL']
        detailed_df_base = df[final_columns]
        detailed_df = detailed_df_base[(detailed_df_base['都道府県'] != '不明') & (detailed_df_base['属性'] != '不明')].copy()
        summary_df = detailed_df.copy()
        summary_df['価格x販売戸数'] = summary_df['価格（平均）'] * summary_df['販売戸数']
        prefecture_summary_df = summary_df.groupby(['都道府県', '属性'], observed=False).agg(総戸数=('総戸数', 'sum'), 販売戸数=('販売戸数', 'sum'), 価格x販売戸数_合計=('価格x販売戸数', 'sum')).reset_index()
        prefecture_summary_df['価格（平均）'] = prefecture_summary_df.apply(lambda r: r['価格x販売戸数_合計']/r['販売戸数'] if r['販売戸数']>0 else 0, axis=1).astype(int)
        prefecture_summary_df['売れ残り率'] = prefecture_summary_df.apply(lambda r: f"{(r['販売戸数']/r['総戸数']*100):.1f}%" if r['総戸数']>0 else "0.0%", axis=1)
        prefecture_summary_df = prefecture_summary_df[['都道府県', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）']]
        monthly_df = detailed_df[detailed_df['完成時期'].str.isdigit()].copy()
        if not monthly_df.empty:
            monthly_df['価格x販売戸数'] = monthly_df['価格（平均）'] * monthly_df['販売戸数']
            monthly_summary_df = monthly_df.groupby(['完成時期', '属性'], observed=False).agg(総戸数=('総戸数', 'sum'), 販売戸数=('販売戸数', 'sum'), 価格x販売戸数_合計=('価格x販売戸数', 'sum')).reset_index()
            monthly_summary_df['価格（平均）'] = monthly_summary_df.apply(lambda r: r['価格x販売戸数_合計']/r['販売戸数'] if r['販売戸数']>0 else 0, axis=1).astype(int)
            monthly_summary_df['売れ残り率'] = monthly_summary_df.apply(lambda r: f"{(r['販売戸数']/r['総戸数']*100):.1f}%" if r['総戸数']>0 else "0.0%", axis=1)
            monthly_summary_df = monthly_summary_df[['完成時期', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）']]
        else:
            monthly_summary_df = pd.DataFrame(columns=['完成時期', '属性', '総戸数', '販売戸数', '売れ残り率', '価格（平均）'])
        print("-> DataFrames generated successfully.")
        return detailed_df, prefecture_summary_df, monthly_summary_df

# ==============================================================================
# 5. 可視化モジュール
# ==============================================================================
class Visualizer:
    @staticmethod
    def analyze_attribute_by_month(df: pd.DataFrame) -> plt.Figure:
        print(" - グラフ 1/4: 販売中物件の属性別戸数割合（グラデーション円グラフ）を作成中...")
        df_for_sale = df[(df['販売戸数'] > 0) & (df['完成時期'].str.isdigit())].copy()
        df_plot = df_for_sale[df_for_sale['属性'].isin(['将来', '新築', '中古'])].copy()
        if df_plot.empty: return None
        attr_order = ['将来', '新築', '中古']; df_plot['属性'] = pd.Categorical(df_plot['属性'], categories=attr_order, ordered=True)
        monthly_attr_totals = df_plot.groupby(['属性', '完成時期'], observed=False)['販売戸数'].sum().sort_index(ascending=[True, False]); color_map = {}
        months_by_attr = df_plot.groupby('属性', observed=False)['完成時期'].unique().apply(lambda x: sorted(x, reverse=True))
        future_months = months_by_attr.get('将来', []); new_months = months_by_attr.get('新築', []); used_months = months_by_attr.get('中古', [])
        if len(future_months) > 0:
            future_cmap = plt.get_cmap('Reds_r', len(future_months) + 3)
            for i, month in enumerate(future_months): color_map[('将来', month)] = future_cmap(i)
        if len(new_months) > 0:
            new_cmap = plt.get_cmap('Greens_r', len(new_months) + 3)
            for i, month in enumerate(new_months): color_map[('新築', month)] = new_cmap(i)
        if len(used_months) > 0:
            used_cmap = plt.get_cmap('Blues_r', len(used_months) + 3)
            for i, month in enumerate(used_months): color_map[('中古', month)] = used_cmap(i)
        fig, ax = plt.subplots(figsize=(12, 10)); pie_values = monthly_attr_totals.values; pie_colors = [color_map.get(idx, 'grey') for idx in monthly_attr_totals.index]
        ax.pie(pie_values, colors=pie_colors, startangle=90, counterclock=False, wedgeprops={'edgecolor': None})
        attribute_totals = df_plot.groupby('属性', observed=False)['販売戸数'].sum().reindex(attr_order); total_sales = attribute_totals.sum(); start_angle = 90
        for attr, attr_total in attribute_totals.items():
            if pd.isna(attr_total) or attr_total == 0: continue
            slice_angle = (attr_total / total_sales) * 360; mid_angle_deg = start_angle - (slice_angle / 2); start_angle -= slice_angle
            x = 1.1 * np.cos(np.deg2rad(mid_angle_deg)); y = 1.1 * np.sin(np.deg2rad(mid_angle_deg))
            pct = (attr_total / total_sales) * 100; label_text = f"{attr}\n{pct:.1f}%\n({attr_total:,.0f}戸)"
            ax.text(x, y, label_text, ha='center', va='center', fontsize=12, weight="bold")
        ax.axis('equal'); plt.title('1. 販売中物件の属性別販売戸数割合（完成月グラデーション）', fontsize=16)
        return fig

    @staticmethod
    def analyze_combined_prefecture_view_detailed(df_detailed: pd.DataFrame, df_summary: pd.DataFrame, df_offices: pd.DataFrame) -> plt.Figure:
        print(" - グラフ 2/4: 都道府県別の価格・供給状況・営業所数の詳細分析を作成中...")
        all_prefs = df_detailed['都道府県'].unique(); geo_order = [pref for pref in PREFECTURE_ORDER if pref in all_prefs]
        if not geo_order: return None
        rate_geo_sorted = df_summary.groupby('都道府県')['売れ残り率'].mean().reindex(geo_order).dropna(); df_price = df_detailed[df_detailed['都道府県'].isin(geo_order)]
        fig, axes = plt.subplots(3, 1, figsize=(18, 24), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]}); fig.suptitle('2. 都道府県別の価格・供給状況・営業所数の分析', fontsize=20, y=0.97)
        ax1 = axes[0]; ax1.set_title('価格分布', fontsize=14); median_prices = df_price.groupby('都道府県')['価格（平均）'].median().reindex(geo_order); norm = mcolors.Normalize(vmin=median_prices.min(), vmax=median_prices.max()); cmap = plt.colormaps['coolwarm_r']; palette = {pref: cmap(norm(median_prices.get(pref, 0))) for pref in geo_order}; sns.boxplot(x='都道府県', y='価格（平均）', data=df_price, order=geo_order, palette=palette, ax=ax1, hue='都道府県', legend=False); ax1.set_ylabel('価格（平均） (万円)'); ax1.set_xlabel(''); ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y):,}'))
        ax2 = axes[1]; ax2.set_title('供給戸数（属性別）・売れ残り率', fontsize=14); ax2_twin = ax2.twinx(); df_plot_attr = df_detailed.copy(); df_plot_attr['販売済み戸数'] = df_plot_attr['総戸数'] - df_plot_attr['販売戸数']; df_melted_attr = pd.melt(df_plot_attr, id_vars=['都道府県', '属性'], value_vars=['販売戸数', '販売済み戸数'], var_name='販売状況_raw', value_name='戸数'); df_melted_attr['販売状況'] = df_melted_attr['販売状況_raw'].map({'販売戸数': '販売中', '販売済み戸数': '販売済み'}); df_melted_attr['plot_category'] = df_melted_attr['属性'] + '-' + df_melted_attr['販売状況']; sales_pivot_attr = df_melted_attr.pivot_table(index='都道府県', columns='plot_category', values='戸数', aggfunc='sum').fillna(0); column_order_attr = ['中古-販売中', '新築-販売中', '将来-販売中', '中古-販売済み', '新築-販売済み', '将来-販売済み']; [sales_pivot_attr.update({col: 0}) for col in column_order_attr if col not in sales_pivot_attr]; sales_pivot_attr = sales_pivot_attr[column_order_attr]; sales_geo_sorted_attr = sales_pivot_attr.reindex(geo_order).dropna(how='all'); color_map_bar_attr = {'将来-販売中': '#b22222', '将来-販売済み': '#ff6347', '新築-販売中': '#006400', '新築-販売済み': '#90ee90', '中古-販売中': '#00008b', '中古-販売済み': '#add8e6'}; colors_bar_attr = [color_map_bar_attr.get(col, 'grey') for col in sales_geo_sorted_attr.columns]; sales_geo_sorted_attr.plot(kind='bar', stacked=True, ax=ax2, color=colors_bar_attr, legend=False, zorder=2); ax2.set_ylabel('総戸数'); h,l = [Patch(facecolor=color_map_bar_attr[n], label=n) for n in column_order_attr if n in color_map_bar_attr and sales_geo_sorted_attr[n].sum() > 0], [n for n in column_order_attr if n in color_map_bar_attr and sales_geo_sorted_attr[n].sum() > 0]; ax2.legend(handles=h, labels=l, loc='upper left', title='属性-販売状況', ncol=2); rate_to_plot_2 = rate_geo_sorted.reindex(sales_geo_sorted_attr.index); ax2_twin.plot(ax2.get_xticks(), rate_to_plot_2.values, color='gold', marker='o', linestyle='--', label='売れ残り率'); ax2_twin.set_ylabel('売れ残り率 (%)', color='black'); ax2_twin.tick_params(axis='y', labelcolor='black'); h2, l2 = ax2_twin.get_legend_handles_labels(); ax2_twin.legend(h2, l2, loc='upper right')
        ax3 = axes[2]; ax3.set_title('販売中戸数（完成月別）と営業所数', fontsize=14); df_plot_month = df_detailed[(df_detailed['販売戸数'] > 0) & (df_detailed['完成時期'].str.isdigit())].copy()
        if not df_plot_month.empty:
            sorted_months = sorted(df_plot_month['完成時期'].unique()); month_to_attr = df_plot_month[['完成時期', '属性']].drop_duplicates().set_index('完成時期')['属性'].to_dict(); future_months = sorted([m for m, a in month_to_attr.items() if a == '将来']); new_months = sorted([m for m, a in month_to_attr.items() if a == '新築']); used_months = sorted([m for m, a in month_to_attr.items() if a == '中古']); color_map_bar_month = {}; future_cmap = plt.get_cmap('Reds', len(future_months)+2) if future_months else None; new_cmap = plt.get_cmap('Greens', len(new_months)+2) if new_months else None; used_cmap = plt.get_cmap('Blues', len(used_months)+2) if used_months else None
            if future_cmap: [color_map_bar_month.update({m: future_cmap(i+2)}) for i, m in enumerate(future_months)]
            if new_cmap: [color_map_bar_month.update({m: new_cmap(i+2)}) for i, m in enumerate(new_months)]
            if used_cmap: [color_map_bar_month.update({m: used_cmap(i+2)}) for i, m in enumerate(used_months)]
            supply_pivot_month = df_plot_month.pivot_table(index='都道府県', columns='完成時期', values='販売戸数', aggfunc='sum').fillna(0)[sorted_months]; supply_geo_sorted_month = supply_pivot_month.reindex(geo_order).dropna(how='all'); colors_bar_month = [color_map_bar_month.get(col, 'grey') for col in supply_geo_sorted_month.columns]; supply_geo_sorted_month.plot(kind='bar', stacked=True, ax=ax3, color=colors_bar_month, zorder=2); ax3.legend(title='完成時期', loc='upper left', ncol=max(1, len(sorted_months)//10))
            if df_offices is not None and not df_offices.empty:
                ax3_twin = ax3.twinx(); office_counts = df_offices.set_index('都道府県')['営業所数']; office_counts_to_plot = office_counts.reindex(supply_geo_sorted_month.index).fillna(0); max_left = supply_geo_sorted_month.sum(axis=1).max(); max_right = office_counts_to_plot.max(); new_limit_left = max(max_left, max_right * 5); final_limit_left = np.ceil(new_limit_left / 10.0) * 10.0 if new_limit_left > 0 else 10; final_limit_right = final_limit_left / 5.0; ax3.set_ylim(0, final_limit_left); ax3_twin.set_ylim(0, final_limit_right); ax3_twin.plot(ax3.get_xticks(), office_counts_to_plot.values, color='darkviolet', marker='D', linestyle=':', label='営業所数'); ax3_twin.set_ylabel('営業所数', color='black'); ax3_twin.tick_params(axis='y', labelcolor='black'); ax3_twin.grid(False); ax3_twin.legend(loc='upper right')
        ax3.set_ylabel('販売中戸数'); ax3.set_xlabel('都道府県'); plt.setp(ax3.get_xticklabels(), rotation=90); [ax.grid(True, linestyle='--', alpha=0.6) for ax in [ax1, ax2, ax3]]; ax2_twin.grid(False)
        if 'ax3_twin' in locals(): ax3_twin.grid(False)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96]); return fig
        
    @staticmethod
    def analyze_combined_5_and_6(df_detailed: pd.DataFrame, df_monthly: pd.DataFrame) -> plt.Figure:
        print(" - グラフ 3/4: 価格分布と供給戸数の連携グラフを作成中..."); df_for_plots = df_detailed[df_detailed['完成時期'].str.isdigit()].copy()
        if df_for_plots.empty or df_monthly.empty: return None
        df_plot_line = df_monthly.copy(); rate_by_month = df_plot_line.groupby('完成時期')['売れ残り率'].mean(); rate_by_month.index = rate_by_month.index.astype(str); all_months = sorted(list(set(df_for_plots['完成時期'].unique()) | set(rate_by_month.index))); rate_by_month = rate_by_month.reindex(all_months, fill_value=np.nan); fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True); fig.suptitle('3. 完成時期別の価格・供給戸数と売れ残り率の分析', fontsize=20, y=0.95)
        ax1_top = axes[0]; ax2_top = ax1_top.twinx(); df_plot_box = df_for_plots.copy(); df_plot_box['完成時期'] = pd.Categorical(df_plot_box['完成時期'], categories=all_months, ordered=True); sns.boxplot(data=df_plot_box, x='完成時期', y='価格（平均）', hue='属性', palette={'将来': 'tomato', '新築': 'mediumseagreen', '中古': 'cornflowerblue', '不明': 'grey'}, ax=ax1_top); ax1_top.set_ylabel('価格（平均） (万円)'); ax1_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); ax2_top.plot(range(len(all_months)), rate_by_month.values, color='gold', marker='o', linestyle='--', label='売れ残り率 (実績)', zorder=10); x_num, y_val, mask = np.arange(len(all_months)), rate_by_month.values, ~np.isnan(rate_by_month.values)
        if np.sum(mask) > 1: sns.regplot(x=x_num[mask], y=y_val[mask], ax=ax2_top, scatter=False, ci=None, color='orange', line_kws={'linestyle':'-', 'linewidth': 3, 'alpha': 0.4}, label='売れ残り率 (トレンド)')
        ax2_top.set_ylabel('売れ残り率 (%)'); ax2_top.set_ylim(bottom=0); h1,l1 = ax1_top.get_legend_handles_labels(); h2,l2 = ax2_top.get_legend_handles_labels(); ax1_top.legend(h1,l1,loc='upper left'); ax2_top.legend(h2,l2,loc='upper right')
        ax1_bottom = axes[1]; ax2_bottom = ax1_bottom.twinx(); df_plot_bar = df_for_plots.copy(); df_plot_bar['販売済み戸数'] = df_plot_bar['総戸数'] - df_plot_bar['販売戸数']; df_melted = pd.melt(df_plot_bar, id_vars=['完成時期', '属性'], value_vars=['販売戸数', '販売済み戸数'], var_name='販売状況_raw', value_name='戸数'); df_melted['plot_category'] = df_melted['属性'] + '-' + df_melted['販売状況_raw'].map({'販売戸数': '販売中', '販売済み戸数': '販売済み'}); plot_df = df_melted.pivot_table(index='完成時期', columns='plot_category', values='戸数', aggfunc='sum').fillna(0).reindex(all_months, fill_value=0); col_order = [f"{a}-{s}" for a in ['将来','新築','中古'] for s in ['販売中','販売済み']]; [plot_df.update({c: 0}) for c in col_order if c not in plot_df]; color_map = {'将来-販売中': '#b22222', '将来-販売済み': '#ff6347', '新築-販売中': '#006400', '新築-販売済み': '#90ee90', '中古-販売中': '#00008b', '中古-販売済み': '#add8e6'}; plot_df[col_order].plot(kind='bar', stacked=True, color=[color_map.get(c, 'grey') for c in col_order], ax=ax1_bottom, legend=None); ax1_bottom.set_ylabel('物件戸数'); ax1_bottom.set_xlabel('完成時期'); ax1_bottom.tick_params(axis='x', rotation=45); handles = [Patch(facecolor=color_map[n], label=n) for n in col_order if n in color_map]; ax1_bottom.legend(handles=handles, loc='upper left', ncol=2); ax2_bottom.plot(range(len(all_months)), rate_by_month.values, color='gold', marker='o', linestyle='--', zorder=10)
        if np.sum(mask) > 1: sns.regplot(x=x_num[mask], y=y_val[mask], ax=ax2_bottom, scatter=False, ci=None, color='orange', line_kws={'linestyle':'-', 'linewidth': 3, 'alpha': 0.4})
        ax2_bottom.set_ylim(bottom=0); ax2_bottom.set_ylabel('売れ残り率 (%)'); [ax.grid(True, linestyle='--', alpha=0.6) for ax in [ax1_top, ax1_bottom]]; [ax.grid(False) for ax in [ax2_top, ax2_bottom]]; fig.tight_layout(rect=[0, 0.03, 1, 0.93]); return fig

    @staticmethod
    def analyze_bubble_chart(df_detailed: pd.DataFrame, df_summary: pd.DataFrame, df_offices: pd.DataFrame) -> tuple:
        print(" - グラフ 4/4: バブルチャート分析を作成中..."); sales = df_detailed.groupby('都道府県')['販売戸数'].sum(); prices = df_detailed.groupby('都道府県')['価格（平均）'].mean(); rates = df_summary.groupby('都道府県')['売れ残り率'].mean(); bubble_df = pd.concat([sales, prices, rates], axis=1).dropna(); bubble_df.columns = ['販売戸数', '価格（平均）', '売れ残り率']
        if df_offices is not None and not df_offices.empty: bubble_df = bubble_df.join(df_offices.set_index('都道府県')['営業所数']).fillna(0)
        else: bubble_df['営業所数'] = 0
        bubble_df['営業所数'] = bubble_df['営業所数'].astype(int)
        if bubble_df.empty: return None, None
        if bubble_df['営業所数'].sum() == 0: print("警告: 営業所数がどの都道府県にも正しく割り当てられていません。結合キーを確認してください。")
        office_counts_unique = sorted(bubble_df['営業所数'].unique()); positive_counts = [c for c in office_counts_unique if c > 0]; colors = plt.get_cmap('viridis', len(positive_counts) + 2) if positive_counts else None; palette_custom = {c: colors(i+1) for i, c in enumerate(positive_counts)} if colors else {};
        if 0 in office_counts_unique: palette_custom[0] = 'lightgrey'
        fig, ax = plt.subplots(figsize=(16, 10)); plot = sns.scatterplot(data=bubble_df, x='売れ残り率', y='価格（平均）', size='販売戸数', sizes=(50, 2000), hue='営業所数', palette=palette_custom, hue_order=office_counts_unique, alpha=0.8, edgecolor='dimgrey', linewidth=1, ax=ax); plt.title('4. 販売戸数、価格、売れ残り率、営業所数のバブルチャート分析', fontsize=20); plt.xlabel('売れ残り率 (%)'); plt.ylabel('平均価格 (万円)'); plt.grid(True, linestyle='--', alpha=0.6)
        for i in range(bubble_df.shape[0]): plt.text(x=bubble_df['売れ残り率'].iloc[i] + 0.1, y=bubble_df['価格（平均）'].iloc[i] + 0.1, s=bubble_df.index[i], fontdict=dict(color='black',size=10))
        handles, labels = plot.get_legend_handles_labels(); plt.legend(handles=handles, labels=labels, title="営業所数 / 販売戸数")
        top3_sales = bubble_df.nlargest(3, '販売戸数'); top3_rates = bubble_df.nlargest(3, '売れ残り率'); top3_prices = bubble_df.nlargest(3, '価格（平均）'); top3_offices = bubble_df.nlargest(3, '営業所数'); lines = [f"・販売規模（円の大きさ）\n [Top 3]\n" + "\n".join([f" {i}. {p} ({r['販売戸数']:,}戸)" for i, (p, r) in enumerate(top3_sales.iterrows(), 1)])]; lines.append(f"\n・売れ残り率（横軸）\n [Top 3 - 課題]\n" + "\n".join([f" {i}. {p} ({r['売れ残り率']:.1f}%)" for i, (p, r) in enumerate(top3_rates.iterrows(), 1)])); lines.append(f"\n・平均価格（縦軸）\n [Top 3]\n" + "\n".join([f" {i}. {p} ({r['価格（平均）']:,.0f}万円)" for i, (p, r) in enumerate(top3_prices.iterrows(), 1)])); lines.append(f"\n・営業所数（色の濃淡）\n [Top 3]\n" + "\n".join([f" {i}. {p} ({r['営業所数']:,}箇所)" for i, (p, r) in enumerate(top3_offices.iterrows(), 1)])); return fig, "\n".join(lines)


# ==============================================================================
# 6. データベース操作 & タスク実行モジュール
# ==============================================================================
class TaskRunner:
    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = create_engine(self.db_url)

    def setup_database_tables(self):
        with self.engine.connect() as connection:
            print("\n--- [Setup Mode] Checking and creating tables if they don't exist ---")
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS tamahome_offices ( prefecture VARCHAR(10) PRIMARY KEY, office_count INTEGER NOT NULL, updated_at TIMESTAMPTZ NOT NULL );
                CREATE TABLE IF NOT EXISTS tamahome_properties_detailed ( id SERIAL PRIMARY KEY, scrape_date DATE NOT NULL, url TEXT NOT NULL, prefecture VARCHAR(10), property_name TEXT, attribute VARCHAR(10), completion_period VARCHAR(10), total_units INTEGER, units_for_sale INTEGER, price_avg BIGINT, price_range TEXT, UNIQUE (scrape_date, url) );
                CREATE TABLE IF NOT EXISTS tamahome_summary_prefecture ( id SERIAL PRIMARY KEY, scrape_date DATE NOT NULL, prefecture VARCHAR(10) NOT NULL, attribute VARCHAR(10) NOT NULL, total_units INTEGER, units_for_sale INTEGER, unsold_rate REAL, price_avg BIGINT, UNIQUE (scrape_date, prefecture, attribute) );
                CREATE TABLE IF NOT EXISTS tamahome_summary_monthly ( id SERIAL PRIMARY KEY, scrape_date DATE NOT NULL, completion_period VARCHAR(10) NOT NULL, attribute VARCHAR(10) NOT NULL, total_units INTEGER, units_for_sale INTEGER, unsold_rate REAL, price_avg BIGINT, UNIQUE (scrape_date, completion_period, attribute) );
                CREATE TABLE IF NOT EXISTS tamahome_daily_reports ( id SERIAL PRIMARY KEY, report_date DATE NOT NULL, report_type VARCHAR(20) NOT NULL, content JSONB NOT NULL );
                CREATE INDEX IF NOT EXISTS idx_report_date ON tamahome_daily_reports (report_date);
            """))
            connection.commit()
            print("-> Tables are ready.")

    def run_scraping_tasks(self, jst_now, force_update=False):
        with self.engine.connect() as connection:
            print("\n" + "="*80 + "\n--- Part 1: Office Information ---")
            office_count_in_db = connection.execute(text("SELECT COUNT(*) FROM tamahome_offices")).scalar()
            if force_update or jst_now.day == 1 or office_count_in_db == 0:
                df_offices = Scraper.scrape_tamahome_offices()
                if df_offices is not None and not df_offices.empty: self._save_offices_to_db(connection, df_offices, jst_now)
            else:
                print("Skipping office data update.")
            print("\n" + "="*80 + "\n--- Part 2: Property Information ---")
            raw_property_df = Scraper.scrape_all_properties_details()
            if raw_property_df is None or raw_property_df.empty:
                print("\nCould not retrieve property data. Terminating property scraping.")
                return
            detailed_df, pref_summary_df, month_summary_df = Scraper.create_and_process_dataframes(raw_property_df, jst_now)
            print("\n--- Saving data to database ---")
            self._save_data_to_db(connection, detailed_df, 'tamahome_properties_detailed', jst_now, ['scrape_date', 'url'])
            self._save_data_to_db(connection, pref_summary_df, 'tamahome_summary_prefecture', jst_now, ['scrape_date', 'prefecture', 'attribute'])
            self._save_data_to_db(connection, month_summary_df, 'tamahome_summary_monthly', jst_now, ['scrape_date', 'completion_period', 'attribute'])
            connection.commit()

    def run_analysis_tasks(self, jst_now):
        with self.engine.connect() as connection:
            print("\n" + "="*80 + "\n--- Part 3: Daily Comparison Analysis ---")
            today = jst_now.date()
            date_query = text("SELECT DISTINCT scrape_date FROM tamahome_properties_detailed ORDER BY scrape_date DESC LIMIT 2;")
            result = connection.execute(date_query).fetchall()
            if len(result) < 2 or result[0][0] != today:
                print("-> Not enough data for comparison. Skipping report generation.")
                return
            latest_date, previous_date = result[0][0], result[1][0]
            df_latest = pd.read_sql(text("SELECT * FROM tamahome_properties_detailed WHERE scrape_date = :date"), connection, params={'date': latest_date})
            df_previous = pd.read_sql(text("SELECT * FROM tamahome_properties_detailed WHERE scrape_date = :date"), connection, params={'date': previous_date})
            reports_to_save = self._analyze_differences(df_latest, df_previous, today)
            if reports_to_save:
                connection.execute(text("DELETE FROM tamahome_daily_reports WHERE report_date = :today"), {'today': today})
                stmt = text("INSERT INTO tamahome_daily_reports (report_date, report_type, content) VALUES (:report_date, :report_type, :content::jsonb)")
                db_result = connection.execute(stmt, reports_to_save)
                print(f"-> Saved {db_result.rowcount} report entries to the database.")
                connection.commit()

    def run_visualization_tasks(self, output_dir):
        # ★★★ 修正箇所: ディレクトリ作成をここに追加 ★★★
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' is ready.")
        
        with self.engine.connect() as connection:
            print("\n" + "="*80 + "\n--- Part 4: PDF Report Visualization ---")
            df_detailed, df_pref, df_monthly, df_offices, latest_date = self._load_data_for_visualization(connection)
            if df_detailed is None:
                print("Could not load data for visualization. Aborting.")
                return

            figures = []
            figures.append(Visualizer.analyze_attribute_by_month(df_detailed))
            figures.append(Visualizer.analyze_combined_prefecture_view_detailed(df_detailed, df_pref, df_offices))
            figures.append(Visualizer.analyze_combined_5_and_6(df_detailed, df_monthly))
            fig_bubble, bubble_text = Visualizer.analyze_bubble_chart(df_detailed, df_pref, df_offices)
            if fig_bubble: figures.append(fig_bubble)
            
            figures = [fig for fig in figures if fig is not None]
            if not figures:
                print("No graphs were generated.")
                return

            self._save_figures_to_pdf(figures, output_dir, latest_date)
            if bubble_text:
                print("\n--- Bubble Chart Analysis ---")
                print(bubble_text)

    def _save_data_to_db(self, connection, df, table_name, jst_now, unique_keys):
        if df.empty: return
        df_to_save = df.copy()
        df_to_save['scrape_date'] = jst_now.date()
        COLUMN_MAP = {'都道府県': 'prefecture', '物件名': 'property_name', '属性': 'attribute', '完成時期': 'completion_period', '総戸数': 'total_units', '販売戸数': 'units_for_sale', '価格（平均）': 'price_avg', '価格': 'price_range', 'URL': 'url', '売れ残り率': 'unsold_rate'}
        df_to_save.rename(columns=COLUMN_MAP, inplace=True)
        if 'unsold_rate' in df_to_save.columns: df_to_save['unsold_rate'] = df_to_save['unsold_rate'].str.replace('%', '').astype(float)
        all_db_cols = ['scrape_date', 'prefecture', 'property_name', 'attribute', 'completion_period', 'total_units', 'units_for_sale', 'price_avg', 'price_range', 'url', 'unsold_rate']
        cols_to_insert = [col for col in all_db_cols if col in df_to_save.columns]
        df_to_save = df_to_save[cols_to_insert]
        records = df_to_save.to_dict(orient='records')
        stmt = text(f"INSERT INTO {table_name} ({', '.join(cols_to_insert)}) VALUES ({', '.join([f':{c}' for c in cols_to_insert])}) ON CONFLICT ({', '.join(unique_keys)}) DO NOTHING;")
        result = connection.execute(stmt, records)
        print(f"-> Saved {result.rowcount} new rows to '{table_name}'.")

    def _save_offices_to_db(self, connection, df_offices, jst_now):
        if df_offices.empty: return
        df_to_save = df_offices.copy(); df_to_save.rename(columns={'都道府県': 'prefecture', '営業所数': 'office_count'}, inplace=True); df_to_save['updated_at'] = jst_now
        records = df_to_save.to_dict(orient='records')
        stmt = text("INSERT INTO tamahome_offices (prefecture, office_count, updated_at) VALUES (:prefecture, :office_count, :updated_at) ON CONFLICT (prefecture) DO UPDATE SET office_count = EXCLUDED.office_count, updated_at = EXCLUDED.updated_at;")
        result = connection.execute(stmt, records)
        print(f"-> Upserted {result.rowcount} rows in 'tamahome_offices'.")

    def _analyze_differences(self, df_latest, df_previous, today):
        df_latest['composite_key'] = df_latest['prefecture'].astype(str) + '_' + df_latest['property_name'].astype(str) + '_' + df_latest['url'].astype(str)
        df_previous['composite_key'] = df_previous['prefecture'].astype(str) + '_' + df_previous['property_name'].astype(str) + '_' + df_previous['url'].astype(str)
        merged_df = pd.merge(df_previous, df_latest, on='composite_key', how='outer', suffixes=('_prev', '_latest'), indicator=True)
        reports_to_save = []
        new_properties = merged_df[merged_df['_merge'] == 'right_only']
        for _, row in new_properties.iterrows(): reports_to_save.append({'report_date': today, 'report_type': 'new', 'content': json.dumps({'prefecture': row['prefecture_latest'], 'property_name': row['property_name_latest'], 'url': row['url_latest'], 'attribute': row['attribute_latest'], 'price_range': row['price_range_latest'], 'units_for_sale': int(row['units_for_sale_latest']) if pd.notna(row['units_for_sale_latest']) else 0}, ensure_ascii=False)})
        removed_properties = merged_df[merged_df['_merge'] == 'left_only']
        for _, row in removed_properties.iterrows(): reports_to_save.append({'report_date': today, 'report_type': 'removed', 'content': json.dumps({'prefecture': row['prefecture_prev'], 'property_name': row['property_name_prev'], 'url': row['url_prev'], 'attribute': row['attribute_prev'], 'price_range': row['price_range_prev'], 'units_for_sale': int(row['units_for_sale_prev']) if pd.notna(row['units_for_sale_prev']) else 0}, ensure_ascii=False)})
        both_df = merged_df[merged_df['_merge'] == 'both'].copy()
        compare_cols = ['attribute', 'completion_period', 'total_units', 'units_for_sale', 'price_avg', 'price_range']
        for _, row in both_df.iterrows():
            updates = {}
            for col in compare_cols:
                val_prev, val_latest = row[f'{col}_prev'], row[f'{col}_latest']
                if pd.isna(val_prev) and pd.isna(val_latest): continue
                if str(val_prev) != str(val_latest): updates[col] = {'from': str(val_prev) if pd.notna(val_prev) else None, 'to': str(val_latest) if pd.notna(val_latest) else None}
            if updates: reports_to_save.append({'report_date': today, 'report_type': 'updated', 'content': json.dumps({'prefecture': row['prefecture_latest'], 'property_name': row['property_name_latest'], 'url': row['url_latest'], 'changes': updates}, ensure_ascii=False)})
        print(f"-> Found {len(new_properties)} new, {len(removed_properties)} removed, {len(reports_to_save) - len(new_properties) - len(removed_properties)} updated properties.")
        return reports_to_save
        
    def _load_data_for_visualization(self, connection):
        latest_date_query = text("SELECT MAX(scrape_date) FROM tamahome_properties_detailed;")
        latest_date = connection.execute(latest_date_query).scalar()
        if not latest_date: return None, None, None, None, None
        def fetch_table(table_name, date): return pd.read_sql(text(f"SELECT * FROM {table_name} WHERE scrape_date = :date"), connection, params={'date': date})
        df_detailed = fetch_table("tamahome_properties_detailed", latest_date)
        df_pref = fetch_table("tamahome_summary_prefecture", latest_date)
        df_monthly = fetch_table("tamahome_summary_monthly", latest_date)
        df_offices = pd.read_sql("SELECT * FROM tamahome_offices", connection)
        if df_detailed.empty: return None, None, None, None, None
        for df in [df_detailed, df_pref, df_monthly, df_offices]: df.rename(columns=COLUMN_MAP_TO_JP, inplace=True)
        if not df_offices.empty: df_offices['都道府県'] = df_offices['都道府県'].apply(lambda x: x+'都' if x=='東京' else (x+'府' if x in ['大阪','京都'] else (x+'県' if not x.endswith(('都','道','府','県')) else x)))
        return df_detailed, df_pref, df_monthly, df_offices, latest_date

    def _save_figures_to_pdf(self, figures, output_dir, latest_date):
        pdf_filename = f"analysis_summary_{latest_date.strftime('%Y%m%d')}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        images_for_pdf = []
        for fig in figures:
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            images_for_pdf.append(PILImage.open(buf))
            plt.close(fig) # メモリを解放
        if images_for_pdf:
            try:
                images_for_pdf[0].save(pdf_path, save_all=True, append_images=images_for_pdf[1:])
                print(f"-> ✅ PDF report saved to '{pdf_path}'.")
            except Exception as e:
                print(f"❌ Error creating PDF: {e}")

# ==============================================================================
# 7. メイン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='All-in-one TamaHome data processing script.')
    parser.add_argument('--setup-db', action='store_true', help='Only creates/updates database tables and exits.')
    parser.add_argument('--scrape-only', action='store_true', help='Only runs scraping and saves to DB.')
    parser.add_argument('--analyze-only', action='store_true', help='Only runs comparison analysis.')
    parser.add_argument('--visualize-only', action='store_true', help='Only runs PDF report generation.')
    parser.add_argument('--force-update', action='store_true', help='Forces an update of the office information.')
    parser.add_argument('-o', '--output-dir', type=str, default='output', help="Directory to save the PDF report.")
    args = parser.parse_args()

    # タスク実行フラグの決定
    run_all = not (args.scrape_only or args.analyze_only or args.visualize_only)
    run_scrape = run_all or args.scrape_only
    run_analyze = run_all or args.analyze_only
    run_visualize = run_all or args.visualize_only
    
    # メイン処理開始
    try:
        runner = TaskRunner(get_database_url())
        
        if args.setup_db:
            runner.setup_database_tables()
        else:
            if run_scrape:
                runner.run_scraping_tasks(datetime.now(ZoneInfo("Asia/Tokyo")), args.force_update)
            
            if run_analyze:
                runner.run_analysis_tasks(datetime.now(ZoneInfo("Asia/Tokyo")))
            
            if run_visualize:
                runner.run_visualization_tasks(args.output_dir)

        print("\n\n" + "*"*60 + "\nAll requested processes completed successfully.\n" + "*"*60)

    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
