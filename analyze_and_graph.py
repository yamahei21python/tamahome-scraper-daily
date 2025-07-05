# -*- coding: utf-8 -*-
"""
【汎用版 / DB / PDF & Artifact】
データベース上のタマホーム物件情報を分析し、
各グラフをメモリ上で生成して直接1つのPDFにまとめて保存します。
GitHub Actionsのアーティファクトとしてアップロードすることを想定しています。
"""
import os
import argparse
from datetime import datetime
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sqlalchemy import create_engine, text

# --- 必要なライブラリのインポートと設定 ---
try:
    from PIL import Image as PILImage
except ImportError:
    print("Pillow not found. Installing...")
    # This might fail in restricted environments, requirements.txt is preferred
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'Pillow'], check=True)
    from PIL import Image as PILImage

try:
    import japanize_matplotlib
except ImportError:
    print("japanize_matplotlib not found. Installing...")
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'japanize_matplotlib'], check=True)
    import japanize_matplotlib

try:
    from matplotlib.patches import Patch
except ImportError:
    pass

# ==============================================================================
# 定数とマッピング
# ==============================================================================
# (このセクションは変更ないため、コードの簡潔化のため省略します)
PREFECTURE_ORDER = [
    '北海道', '青森県', '岩手県', '宮城県', '秋田県', '山形県', '福島県', '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県',
    '新潟県', '富山県', '石川県', '福井県', '山梨県', '長野県', '岐阜県', '静岡県', '愛知県', '三重県', '滋賀県', '京都府', '大阪府', '兵庫県', '奈良県', '和歌山県',
    '鳥取県', '島根県', '岡山県', '広島県', '山口県', '徳島県', '香川県', '愛媛県', '高知県', '福岡県', '佐賀県', '長崎県', '熊本県', '大分県', '宮崎県', '鹿児島県', '沖縄県'
]
COLUMN_MAP_TO_JP = {
    'url': 'URL', 'prefecture': '都道府県', 'property_name': '物件名', 'attribute': '属性',
    'completion_period': '完成時期', 'total_units': '総戸数', 'units_for_sale': '販売戸数',
    'price_avg': '価格（平均）', 'price_range': '価格', 'office_count': '営業所数',
    'unsold_rate': '売れ残り率'
}

# ==============================================================================
# データ読み込み関数
# ==============================================================================
# (この関数の内容は変更ないため、コードの簡潔化のため省略します)
def load_data_from_db(connection) -> tuple:
    # ... (元のload_data_from_db関数をここに配置)
    """データベースから最新日付の分析用データを読み込む。"""
    print("--- データベースから最新日付のデータを検索・読み込みます ---")
    latest_date_query = text("SELECT MAX(scrape_date) FROM tamahome_properties_detailed;")
    latest_date = connection.execute(latest_date_query).scalar()
    if not latest_date:
        print("エラー: データベースに分析対象のデータが見つかりません。")
        return None, None, None, None, None
    print(f"-> 最新のデータ日付 '{latest_date.strftime('%Y-%m-%d')}' を使用します。")
    def fetch_table_by_date(table_name, date):
        query = text(f"SELECT * FROM {table_name} WHERE scrape_date = :date")
        return pd.read_sql(query, connection, params={'date': date})
    df_detailed = fetch_table_by_date("tamahome_properties_detailed", latest_date)
    df_pref_summary = fetch_table_by_date("tamahome_summary_prefecture", latest_date)
    df_monthly_summary = fetch_table_by_date("tamahome_summary_monthly", latest_date)
    df_offices = pd.read_sql("SELECT * FROM tamahome_offices", connection)
    if df_detailed.empty or df_pref_summary.empty or df_monthly_summary.empty:
        print("エラー: 最新日付の必須データの一部が見つかりませんでした。")
        return None, None, None, None, None
    for df in [df_detailed, df_pref_summary, df_monthly_summary, df_offices]:
        df.rename(columns=COLUMN_MAP_TO_JP, inplace=True)
    if not df_offices.empty:
        df_offices['都道府県'] = df_offices['都道府県'].apply(lambda x: x+'都' if x=='東京' else (x+'府' if x in ['大阪','京都'] else (x+'県' if not x.endswith(('都','道','府','県')) else x)))
    print("\nデータの読み込みが完了しました。")
    return df_detailed, df_pref_summary, df_monthly_summary, df_offices, latest_date

# ==============================================================================
# グラフ作成関数
# ==============================================================================
# (これらの関数の内容は変更ないため、コードの簡潔化のため省略します)
def analyze_attribute_by_month(df: pd.DataFrame) -> plt.Figure:
    # ... (元のanalyze_attribute_by_month関数をここに配置)
    print(" - グラフ 1/4: 販売中物件の属性別戸数割合（グラデーション円グラフ）を作成中...")
    df_for_sale = df[(df['販売戸数'] > 0) & (df['完成時期'].str.isdigit())].copy()
    df_plot = df_for_sale[df_for_sale['属性'].isin(['将来', '新築', '中古'])].copy()
    if df_plot.empty: return None
    attr_order = ['将来', '新築', '中古']
    df_plot['属性'] = pd.Categorical(df_plot['属性'], categories=attr_order, ordered=True)
    monthly_attr_totals = df_plot.groupby(['属性', '完成時期'])['販売戸数'].sum().sort_index(ascending=[True, False])
    color_map = {}
    months_by_attr = df_plot.groupby('属性')['完成時期'].unique().apply(lambda x: sorted(x, reverse=True))
    future_months = months_by_attr.get('将来', [])
    if len(future_months) > 0:
        future_cmap = plt.get_cmap('Reds_r', len(future_months) + 3)
        for i, month in enumerate(future_months): color_map[('将来', month)] = future_cmap(i)
    new_months = months_by_attr.get('新築', [])
    if len(new_months) > 0:
        new_cmap = plt.get_cmap('Greens_r', len(new_months) + 3)
        for i, month in enumerate(new_months): color_map[('新築', month)] = new_cmap(i)
    used_months = months_by_attr.get('中古', [])
    if len(used_months) > 0:
        used_cmap = plt.get_cmap('Blues_r', len(used_months) + 3)
        for i, month in enumerate(used_months): color_map[('中古', month)] = used_cmap(i)
    fig, ax = plt.subplots(figsize=(12, 10))
    pie_values = monthly_attr_totals.values
    pie_colors = [color_map.get(idx, 'grey') for idx in monthly_attr_totals.index]
    ax.pie(pie_values, colors=pie_colors, startangle=90, counterclock=False, wedgeprops={'edgecolor': None})
    attribute_totals = df_plot.groupby('属性')['販売戸数'].sum().reindex(attr_order)
    total_sales = attribute_totals.sum()
    start_angle = 90
    for attr, attr_total in attribute_totals.items():
        if pd.isna(attr_total) or attr_total == 0: continue
        slice_angle = (attr_total / total_sales) * 360
        mid_angle_deg = start_angle - (slice_angle / 2)
        start_angle -= slice_angle
        x = 1.1 * np.cos(np.deg2rad(mid_angle_deg)); y = 1.1 * np.sin(np.deg2rad(mid_angle_deg))
        pct = (attr_total / total_sales) * 100
        label_text = f"{attr}\n{pct:.1f}%\n({attr_total:,.0f}戸)"
        ax.text(x, y, label_text, ha='center', va='center', fontsize=12, weight="bold")
    ax.axis('equal')
    plt.title('1. 販売中物件の属性別販売戸数割合（完成月グラデーション）', fontsize=16)
    return fig

def analyze_combined_prefecture_view_detailed(df_detailed: pd.DataFrame, df_summary: pd.DataFrame, df_offices: pd.DataFrame) -> plt.Figure:
    # ... (元のanalyze_combined_prefecture_view_detailed関数をここに配置)
    print(" - グラフ 2/4: 都道府県別の価格・供給状況・営業所数の詳細分析を作成中...")
    all_prefs = df_detailed['都道府県'].unique()
    geo_order = [pref for pref in PREFECTURE_ORDER if pref in all_prefs]
    if not geo_order: return None
    rate_geo_sorted = df_summary.groupby('都道府県')['売れ残り率'].mean().reindex(geo_order).dropna()
    df_price = df_detailed[df_detailed['都道府県'].isin(geo_order)]
    fig, axes = plt.subplots(3, 1, figsize=(18, 24), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.suptitle('2. 都道府県別の価格・供給状況・営業所数の分析', fontsize=20, y=0.97)
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

def analyze_combined_5_and_6(df_detailed: pd.DataFrame, df_monthly: pd.DataFrame) -> plt.Figure:
    # ... (元のanalyze_combined_5_and_6関数をここに配置)
    print(" - グラフ 3/4: 価格分布と供給戸数の連携グラフを作成中..."); df_for_plots = df_detailed[df_detailed['完成時期'].str.isdigit()].copy()
    if df_for_plots.empty or df_monthly.empty: return None
    df_plot_line = df_monthly.copy(); rate_by_month = df_plot_line.groupby('完成時期')['売れ残り率'].mean(); rate_by_month.index = rate_by_month.index.astype(str); all_months = sorted(list(set(df_for_plots['完成時期'].unique()) | set(rate_by_month.index))); rate_by_month = rate_by_month.reindex(all_months, fill_value=np.nan); fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True); fig.suptitle('3. 完成時期別の価格・供給戸数と売れ残り率の分析', fontsize=20, y=0.95)
    ax1_top = axes[0]; ax2_top = ax1_top.twinx(); df_plot_box = df_for_plots.copy(); df_plot_box['完成時期'] = pd.Categorical(df_plot_box['完成時期'], categories=all_months, ordered=True); sns.boxplot(data=df_plot_box, x='完成時期', y='価格（平均）', hue='属性', palette={'将来': 'tomato', '新築': 'mediumseagreen', '中古': 'cornflowerblue', '不明': 'grey'}, ax=ax1_top); ax1_top.set_ylabel('価格（平均） (万円)'); ax1_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}')); ax2_top.plot(range(len(all_months)), rate_by_month.values, color='gold', marker='o', linestyle='--', label='売れ残り率 (実績)', zorder=10); x_num, y_val, mask = np.arange(len(all_months)), rate_by_month.values, ~np.isnan(rate_by_month.values)
    if np.sum(mask) > 1: sns.regplot(x=x_num[mask], y=y_val[mask], ax=ax2_top, scatter=False, ci=None, color='orange', line_kws={'linestyle':'-', 'linewidth': 3, 'alpha': 0.4}, label='売れ残り率 (トレンド)')
    ax2_top.set_ylabel('売れ残り率 (%)'); ax2_top.set_ylim(bottom=0); h1,l1 = ax1_top.get_legend_handles_labels(); h2,l2 = ax2_top.get_legend_handles_labels(); ax1_top.legend(h1,l1,loc='upper left'); ax2_top.legend(h2,l2,loc='upper right')
    ax1_bottom = axes[1]; ax2_bottom = ax1_bottom.twinx(); df_plot_bar = df_for_plots.copy(); df_plot_bar['販売済み戸数'] = df_plot_bar['総戸数'] - df_plot_bar['販売戸数']; df_melted = pd.melt(df_plot_bar, id_vars=['完成時期', '属性'], value_vars=['販売戸数', '販売済み戸数'], var_name='販売状況_raw', value_name='戸数'); df_melted['plot_category'] = df_melted['属性'] + '-' + df_melted['販売状況_raw'].map({'販売戸数': '販売中', '販売済み戸数': '販売済み'}); plot_df = df_melted.pivot_table(index='完成時期', columns='plot_category', values='戸数', aggfunc='sum').fillna(0).reindex(all_months, fill_value=0); col_order = [f"{a}-{s}" for a in ['将来','新築','中古'] for s in ['販売中','販売済み']]; [plot_df.update({c: 0}) for c in col_order if c not in plot_df]; color_map = {'将来-販売中': '#b22222', '将来-販売済み': '#ff6347', '新築-販売中': '#006400', '新築-販売済み': '#90ee90', '中古-販売中': '#00008b', '中古-販売済み': '#add8e6'}; plot_df[col_order].plot(kind='bar', stacked=True, color=[color_map.get(c, 'grey') for c in col_order], ax=ax1_bottom, legend=None); ax1_bottom.set_ylabel('物件戸数'); ax1_bottom.set_xlabel('完成時期'); ax1_bottom.tick_params(axis='x', rotation=45); handles = [Patch(facecolor=color_map[n], label=n) for n in col_order if n in color_map]; ax1_bottom.legend(handles=handles, loc='upper left', ncol=2); ax2_bottom.plot(range(len(all_months)), rate_by_month.values, color='gold', marker='o', linestyle='--', zorder=10)
    if np.sum(mask) > 1: sns.regplot(x=x_num[mask], y=y_val[mask], ax=ax2_bottom, scatter=False, ci=None, color='orange', line_kws={'linestyle':'-', 'linewidth': 3, 'alpha': 0.4})
    ax2_bottom.set_ylim(bottom=0); ax2_bottom.set_ylabel('売れ残り率 (%)'); [ax.grid(True, linestyle='--', alpha=0.6) for ax in [ax1_top, ax1_bottom]]; [ax.grid(False) for ax in [ax2_top, ax2_bottom]]; fig.tight_layout(rect=[0, 0.03, 1, 0.93]); return fig

def analyze_bubble_chart(df_detailed: pd.DataFrame, df_summary: pd.DataFrame, df_offices: pd.DataFrame) -> tuple:
    # ... (元のanalyze_bubble_chart関数をここに配置)
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
# メイン処理
# ==============================================================================
def main(args):
    """スクリプト全体の処理を実行するメイン関数。"""
    
    # 1. 環境変数からDB URLを取得
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        raise ValueError("環境変数 'DATABASE_URL' が設定されていません。")

    # 2. PDF保存先パスを決定
    # コマンドライン引数 > 環境変数 GITHUB_WORKSPACE > デフォルト'output'
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.environ.get('GITHUB_WORKSPACE', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. DBからデータを読み込み
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("✅ データベースへの接続に成功しました。")
            df_detailed, df_pref, df_monthly, df_offices, latest_date = load_data_from_db(connection)
    except Exception as e:
        print(f"❌ データベース処理中にエラーが発生しました: {e}")
        return

    if df_detailed is None:
        print("\n分析を中止します。")
        return

    # 4. グラフ生成
    print(f"\n--- 分析グラフの生成を開始します ---")
    figures = []
    bubble_analysis_text = ""
    
    figures.append(analyze_attribute_by_month(df_detailed))
    figures.append(analyze_combined_prefecture_view_detailed(df_detailed, df_pref, df_offices))
    figures.append(analyze_combined_5_and_6(df_detailed, df_monthly))
    
    fig_bubble, bubble_text = analyze_bubble_chart(df_detailed, df_pref, df_offices)
    if fig_bubble:
        figures.append(fig_bubble)
        if bubble_text:
            bubble_analysis_text = bubble_text

    figures = [fig for fig in figures if fig is not None]

    if not figures:
        print("生成されたグラフがありません。処理を終了します。")
        return

    # 5. PDFにまとめて保存
    print("\n--- 全グラフをPDFにまとめて保存します ---")
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
            print(f"-> ✅ PDFを '{pdf_path}' に保存しました。")
        except Exception as e:
            print(f"❌ エラー: PDFの作成に失敗しました - {e}")

    # 6. バブルチャートのテキスト分析結果を出力
    if bubble_analysis_text:
        print("\n--- バブルチャート分析結果 ---")
        print(bubble_analysis_text)
    
    print("\n--- すべての処理が完了しました ---")

# ==============================================================================
# スクリプト実行エントリーポイント
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze TamaHome data from DB and generate a PDF report.')
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help="Directory to save the generated PDF file. Defaults to './output'."
    )
    args = parser.parse_args()
    main(args)
