# -*- coding: utf-8 -*-
"""
【GitHub Actions対応版】
ローカルフォルダ上のタマホーム物件情報CSVと特定の営業所数CSVを分析し、
各グラフを画像として保存し、最後にPDFにまとめて保存します。
（GitHub Actionsのログにはテキスト情報のみ出力）
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
# from IPython.display import display, Image # GitHub Actionsでは不要なので削除
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import warnings # For future warnings from matplotlib/seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Pillowはrequirements.txtでインストールするので、条件付きインストールは不要
from PIL import Image as PILImage

# japanize_matplotlibもrequirements.txtでインストールするので、条件付きインストールは不要
import japanize_matplotlib

# from matplotlib.patches import Patch # 通常は自動でインポートされるが明示的に
# try:
#     from matplotlib.patches import Patch
# except ImportError:
#     pass # Python 3.9+では必要だが、古いバージョンでは不要な場合があるためpass

# ==============================================================================
# 都道府県の地理的順序を定義
# ==============================================================================
PREFECTURE_ORDER = [
    '北海道', '青森県', '岩手県', '宮城県', '秋田県', '山形県', '福島県', '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県',
    '新潟県', '富山県', '石川県', '福井県', '山梨県', '長野県', '岐阜県', '静岡県', '愛知県', '三重県', '滋賀県', '京都府', '大阪府', '兵庫県', '奈良県', '和歌山県',
    '鳥取県', '島根県', '岡山県', '広島県', '山口県', '徳島県', '香川県', '愛媛県', '高知県', '福岡県', '佐賀県', '長崎県', '熊本県', '大分県', '宮崎県', '鹿児島県', '沖縄県'
]

# ==============================================================================
# データ読み込み関数 (ローカルパスからの読み込みに修正)
# ==============================================================================
def load_data_from_local(base_local_path: str) -> tuple:
    """指定されたローカルパスから最新の日付フォルダを検索し、各種CSVファイルを読み込む。"""
    print(f"--- ローカルのベースパス '{base_local_path}' から最新の日別フォルダを検索します ---")
    
    if not os.path.exists(base_local_path):
        print(f"エラー: ベースパス '{base_local_path}' が見つかりません。")
        return None, None, None, None, None, None # データフォルダパスもNoneを返す

    all_subdirs = glob.glob(os.path.join(base_local_path, '*/'))
    date_folders = [d for d in all_subdirs if os.path.isdir(d) and os.path.basename(os.path.normpath(d)).isdigit() and len(os.path.basename(os.path.normpath(d))) == 8]
    if not date_folders:
        print(f"エラー: ベースパス '{base_local_path}' に日付名のフォルダが見つかりません。")
        return None, None, None, None, None, None

    latest_folder = sorted(date_folders, reverse=True)[0]
    print(f"-> 最新のデータフォルダ '{os.path.basename(os.path.normpath(latest_folder))}' を使用します。")

    detailed_file_list = glob.glob(os.path.join(latest_folder, 'tamahome_detailed_list_*.csv'))
    pref_summary_file_list = glob.glob(os.path.join(latest_folder, 'tamahome_prefecture_summary_*.csv'))
    monthly_summary_file_list = glob.glob(os.path.join(latest_folder, 'tamahome_monthly_summary_*.csv'))

    # 各ファイルのパスを特定
    detailed_file = detailed_file_list[0] if detailed_file_list else None
    pref_summary_file = pref_summary_file_list[0] if pref_summary_file_list else None
    monthly_summary_file = monthly_summary_file_list[0] if monthly_summary_file_list else None


    if not all([detailed_file, pref_summary_file, monthly_summary_file]):
        print("\nエラー: 最新フォルダ内に物件情報の必須CSVファイルの一部またはすべてが見つかりません。")
        if not detailed_file: print("  - tamahome_detailed_list_*.csv が見つかりません。")
        if not pref_summary_file: print("  - tamahome_prefecture_summary_*.csv が見つかりません。")
        if not monthly_summary_file: print("  - tamahome_monthly_summary_*.csv が見つかりません。")
        return None, None, None, None, None, None


    df_detailed = pd.read_csv(detailed_file)
    df_pref_summary = pd.read_csv(pref_summary_file)
    df_monthly_summary = pd.read_csv(monthly_summary_file)

    # ★★★★★ 最新の営業所数CSVを動的に検索して読み込み ★★★★★
    df_offices = None
    # 営業所数CSVは日別フォルダ内ではなく、BASE_LOCAL_PATH直下にある可能性も考慮
    office_file_pattern = os.path.join(base_local_path, 'tamahome_offices_*.csv')
    office_file_list = sorted(glob.glob(office_file_pattern), reverse=True) # 最新のものを取得

    if office_file_list:
        latest_office_file = office_file_list[0]
        print(f"営業所数リスト: {os.path.basename(latest_office_file)}")
        try:
            df_offices = pd.read_csv(latest_office_file)
            # 都道府県名の表記を統一
            df_offices['都道府県'] = df_offices['都道府県'].apply(lambda x: x + '都' if x == '東京' else (x + '府' if x in ['大阪', '京都'] else (x + '県' if not x.endswith(('都', '道', '府', '県')) else x)))
        except Exception as e:
            print(f"エラー: 営業所数CSVファイル '{os.path.basename(latest_office_file)}' の読み込みに失敗しました - {e}")
            df_offices = None # 失敗した場合はNoneを維持
    else:
        print(f"警告: 営業所数CSVファイルが見つかりませんでした (検索パターン: {office_file_pattern})")

    print("\nデータの読み込みが完了しました。")
    return df_detailed, df_pref_summary, df_monthly_summary, df_offices, latest_folder


# ==============================================================================
# グラフ作成関数
# ==============================================================================
def analyze_attribute_by_month(df: pd.DataFrame, date_str: str, save_path: str) -> str:
    """1. 販売中物件の属性別販売中戸数割合をグラデーション円グラフで可視化する。"""
    print(" - グラフ 1/4: 販売中物件の属性別戸数割合（グラデーション円グラフ）を作成中...")
    df_for_sale = df[(df['販売戸数'] > 0) & (df['完成時期'].str.isdigit())].copy()
    df_plot = df_for_sale[df_for_sale['属性'].isin(['将来', '新築', '中古'])].copy()
    if df_plot.empty:
        print(" -> スキップ: グラフ1の対象データがありません。")
        return None
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
    output_path = os.path.join(save_path, f'analysis_01_attribute_pie_chart_{date_str}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f" -> 一時保存完了: {os.path.basename(output_path)}")
    return output_path

def analyze_combined_prefecture_view_detailed(df_detailed: pd.DataFrame, df_summary: pd.DataFrame, df_offices: pd.DataFrame, date_str: str, save_path: str) -> str:
    """2. 都道府県別の価格と供給状況、営業所数を3段構成で詳細分析する。"""
    print(" - グラフ 2/4: 都道府県別の価格・供給状況・営業所数の詳細分析を作成中...")

    # --- 0. 共通のデータ準備 ---
    all_prefs = df_detailed['都道府県'].unique()
    geo_order = [pref for pref in PREFECTURE_ORDER if pref in all_prefs]
    if not geo_order:
        print(" -> スキップ: 対象となる都道府県データがありません。")
        return None

    rate_geo_sorted = pd.Series(dtype=float)
    if '売れ残り率' in df_summary.columns:
        if '売れ残り率_数値' not in df_summary.columns:
            df_summary['売れ残り率_数値'] = df_summary['売れ残り率'].str.replace('%', '').astype(float)
        rate_by_pref = df_summary.groupby('都道府県')['売れ残り率_数値'].mean()
        rate_geo_sorted = rate_by_pref.reindex(geo_order).dropna()

    df_price = df_detailed[df_detailed['都道府県'].isin(geo_order)]

    if df_price.empty:
        print(" -> スキップ: グラフ2の対象データ（価格情報）がありません。")
        return None

    # --- グラフ描画設定 ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 24), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    fig.suptitle('2. 都道府県別の価格・供給状況・営業所数の分析', fontsize=20, y=0.97)

    # --- 上段: 価格分布 ---
    ax1 = axes[0]
    ax1.set_title('価格分布', fontsize=14)
    median_prices = df_price.groupby('都道府県')['価格（平均）'].median().reindex(geo_order)
    norm = mcolors.Normalize(vmin=median_prices.min(), vmax=median_prices.max()); cmap = plt.colormaps['coolwarm_r']
    palette = {pref: cmap(norm(median_prices.get(pref, 0))) for pref in geo_order}
    sns.boxplot(x='都道府県', y='価格（平均）', data=df_price, order=geo_order, palette=palette, ax=ax1, hue='都道府県', legend=False)
    ax1.set_ylabel('価格（平均） (万円)'); ax1.set_xlabel('')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y):,}'))

    # --- 中段: 供給戸数 ---
    ax2 = axes[1]
    ax2.set_title('供給戸数（属性別）・売れ残り率', fontsize=14)
    ax2_twin = ax2.twinx()
    df_plot_attr = df_detailed.copy()
    df_plot_attr['販売済み戸数'] = df_plot_attr['総戸数'] - df_plot_attr['販売戸数']
    df_melted_attr = pd.melt(df_plot_attr, id_vars=['都道府県', '属性'], value_vars=['販売戸数', '販売済み戸数'], var_name='販売状況_raw', value_name='戸数')
    df_melted_attr['販売状況'] = df_melted_attr['販売状況_raw'].map({'販売戸数': '販売中', '販売済み戸数': '販売済み'})
    df_melted_attr['plot_category'] = df_melted_attr['属性'] + '-' + df_melted_attr['販売状況']
    sales_pivot_attr = df_melted_attr.pivot_table(index='都道府県', columns='plot_category', values='戸数', aggfunc='sum').fillna(0)
    column_order_attr = ['中古-販売中', '新築-販売中', '将来-販売中', '中古-販売済み', '新築-販売済み', '将来-販売済み']
    for col in column_order_attr:
        if col not in sales_pivot_attr.columns: sales_pivot_attr[col] = 0
    sales_pivot_attr = sales_pivot_attr[column_order_attr]
    sales_geo_sorted_attr = sales_pivot_attr.reindex(geo_order).dropna(how='all')
    color_map_bar_attr = {'将来-販売中': '#b22222', '将来-販売済み': '#ff6347', '新築-販売中': '#006400', '新築-販売済み': '#90ee90', '中古-販売中': '#00008b', '中古-販売済み': '#add8e6'}
    colors_bar_attr = [color_map_bar_attr.get(col, 'grey') for col in sales_geo_sorted_attr.columns]
    sales_geo_sorted_attr.plot(kind='bar', stacked=True, ax=ax2, color=colors_bar_attr, legend=False, zorder=2)
    ax2.set_ylabel('総戸数')
    from matplotlib.patches import Patch # ここで明示的にインポート
    handles_bar_attr = [Patch(facecolor=color_map_bar_attr[name], label=name) for name in column_order_attr if name in color_map_bar_attr and sales_geo_sorted_attr[name].sum() > 0]
    ax2.legend(handles=handles_bar_attr, loc='upper left', title='属性-販売状況', ncol=2)
    if not rate_geo_sorted.empty:
        rate_to_plot_2 = rate_geo_sorted.reindex(sales_geo_sorted_attr.index)
        ax2_twin.plot(ax2.get_xticks(), rate_to_plot_2.values, color='gold', marker='o', linestyle='--', label='売れ残り率')
        ax2_twin.set_ylabel('売れ残り率 (%)', color='black')
        ax2_twin.tick_params(axis='y', labelcolor='black')
        h2, l2 = ax2_twin.get_legend_handles_labels()
        ax2_twin.legend(h2, l2, loc='upper right')

    # --- 下段: 完成月別の販売中戸数と営業所数 ---
    ax3 = axes[2]
    ax3.set_title('販売中戸数（完成月別）と営業所数', fontsize=14)
    df_plot_month = df_detailed[(df_detailed['販売戸数'] > 0) & (df_detailed['完成時期'].str.isdigit())].copy()

    if not df_plot_month.empty:
        sorted_months = sorted(df_plot_month['完成時期'].unique())
        month_to_attr = df_plot_month[['完成時期', '属性']].drop_duplicates().set_index('完成時期')['属性'].to_dict()
        future_months = sorted([m for m, a in month_to_attr.items() if a == '将来'])
        new_months = sorted([m for m, a in month_to_attr.items() if a == '新築'])
        used_months = sorted([m for m, a in month_to_attr.items() if a == '中古'])

        color_map_bar_month = {}
        if future_months:
            future_cmap = plt.get_cmap('Reds', len(future_months) + 2)
            for i, month in enumerate(future_months): color_map_bar_month[month] = future_cmap(i + 2)
        if new_months:
            new_cmap = plt.get_cmap('Greens', len(new_months) + 2)
            for i, month in enumerate(new_months): color_map_bar_month[month] = new_cmap(i + 2)
        if used_months:
            used_cmap = plt.get_cmap('Blues', len(used_months) + 2)
            for i, month in enumerate(used_months): color_map_bar_month[month] = used_cmap(i + 2)

        supply_pivot_month = df_plot_month.pivot_table(index='都道府県', columns='完成時期', values='販売戸数', aggfunc='sum').fillna(0)
        supply_pivot_month = supply_pivot_month[sorted_months]
        supply_geo_sorted_month = supply_pivot_month.reindex(geo_order).dropna(how='all')

        colors_bar_month = [color_map_bar_month.get(col, 'grey') for col in supply_geo_sorted_month.columns]
        supply_geo_sorted_month.plot(kind='bar', stacked=True, ax=ax3, color=colors_bar_month, zorder=2)
        ax3.legend(title='完成時期', loc='upper left', ncol=max(1, len(sorted_months) // 10))

        if df_offices is not None and not df_offices.empty:
            ax3_twin = ax3.twinx()
            office_counts = df_offices.set_index('都道府県')['営業所数']
            office_counts_to_plot = office_counts.reindex(supply_geo_sorted_month.index).fillna(0)

            max_left = supply_geo_sorted_month.sum(axis=1).max() if not supply_geo_sorted_month.empty else 0
            max_right = office_counts_to_plot.max() if not office_counts_to_plot.empty else 0

            # スケール調整：左軸（販売中戸数）が右軸（営業所数）の5倍になるように調整
            # ただし、0除算を避ける
            if max_right > 0:
                # 営業所数の最大値を基準に、販売中戸数の新しい上限を計算
                new_limit_left = max_right * 5.0
            else:
                new_limit_left = max_left # 営業所数が0の場合は販売中戸数に合わせる

            # 販売中戸数の上限が0の場合（データがない場合）
            if new_limit_left == 0:
                ax3.set_ylim(0, 10) # 最小限の範囲を設定
                ax3_twin.set_ylim(0, 2)
            else:
                # キリの良い数字に丸める (例: 10単位で切り上げ)
                final_limit_left = np.ceil(new_limit_left / 10.0) * 10.0
                final_limit_right = final_limit_left / 5.0 # 右軸もそれに合わせて調整
                ax3.set_ylim(0, final_limit_left)
                ax3_twin.set_ylim(0, final_limit_right)

            ax3_twin.plot(ax3.get_xticks(), office_counts_to_plot.values, color='darkviolet', marker='D', linestyle=':', label='営業所数')
            ax3_twin.set_ylabel('営業所数', color='black')
            ax3_twin.tick_params(axis='y', labelcolor='black')
            ax3_twin.grid(False)
            ax3_twin.legend(loc='upper right')
        else:
            print(" -> 営業所データがないため、下段グラフの営業所数は表示しません。")

    ax3.set_ylabel('販売中戸数')
    ax3.set_xlabel('都道府県')
    plt.setp(ax3.get_xticklabels(), rotation=90)

    # --- 全体の調整と保存 ---
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle='--', alpha=0.6)
    ax2_twin.grid(False)
    if 'ax3_twin' in locals(): ax3_twin.grid(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = os.path.join(save_path, f'analysis_02_prefecture_detailed_{date_str}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f" -> 一時保存完了: {os.path.basename(output_path)}")
    return output_path

def analyze_combined_5_and_6(df_detailed: pd.DataFrame, df_monthly: pd.DataFrame, date_str: str, save_path: str) -> str:
    """3. 完成時期別の価格・供給戸数と売れ残り率の連携グラフを作成する。"""
    print(" - グラフ 3/4: 価格分布と供給戸数の連携グラフを作成中...")
    df_for_plots = df_detailed[df_detailed['完成時期'].str.isdigit()].copy()
    if df_for_plots.empty or df_monthly.empty or '売れ残り率' not in df_monthly.columns:
        print(" -> スキップ: グラフ3の対象データがありません。")
        return None
    df_plot_line = df_monthly.copy()
    df_plot_line['売れ残り率_数値'] = df_plot_line['売れ残り率'].str.replace('%', '').astype(float)
    rate_by_month = df_plot_line.groupby('完成時期')['売れ残り率_数値'].mean()
    rate_by_month.index = rate_by_month.index.astype(str)
    all_months_set = set(df_for_plots['完成時期'].unique()) | set(rate_by_month.index)
    all_months = sorted(list(all_months_set))
    rate_by_month = rate_by_month.reindex(all_months, fill_value=np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
    fig.suptitle('3. 完成時期別の価格・供給戸数と売れ残り率の分析', fontsize=20, y=0.95)

    # --- 上段グラフ ---
    ax1_top = axes[0]
    ax2_top = ax1_top.twinx()
    df_plot_box = df_for_plots.copy()
    df_plot_box['完成時期'] = pd.Categorical(df_plot_box['完成時期'], categories=all_months, ordered=True)
    color_map_box = {'将来': 'tomato', '新築': 'mediumseagreen', '中古': 'cornflowerblue', '不明': 'grey'}
    sns.boxplot(data=df_plot_box, x='完成時期', y='価格（平均）', hue='属性', palette=color_map_box, ax=ax1_top)
    ax1_top.set_xlabel('')
    ax1_top.set_ylabel('価格（平均） (万円)', fontsize=12)
    ax1_top.grid(True, linestyle='--', alpha=0.6)
    ax1_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax2_top.plot(range(len(all_months)), rate_by_month.values, color='gold', marker='o', linestyle='--', label='売れ残り率 (実績)', zorder=10)
    x_numeric = np.arange(len(all_months))
    y_values = rate_by_month.values
    valid_mask = ~np.isnan(y_values)
    if np.sum(valid_mask) > 1:
        sns.regplot(x=x_numeric[valid_mask], y=y_values[valid_mask], ax=ax2_top, scatter=False,
                    ci=None, color='orange', line_kws={'linestyle':'-', 'linewidth': 3, 'alpha': 0.4}, label='売れ残り率 (トレンド)')
    ax2_top.set_ylabel('売れ残り率 (%)', fontsize=12, color='black')
    ax2_top.grid(False)
    ax2_top.set_ylim(bottom=0)
    h1, l1 = ax1_top.get_legend_handles_labels()
    h2, l2 = ax2_top.get_legend_handles_labels()
    ax1_top.legend(h1, l1, loc='upper left', title='属性')
    if h2: ax2_top.legend(h2, l2, loc='upper right')

    # --- 下段グラフ ---
    ax1_bottom = axes[1]
    ax2_bottom = ax1_bottom.twinx()
    df_plot_bar = df_for_plots.copy()
    df_plot_bar['販売済み戸数'] = df_plot_bar['総戸数'] - df_plot_bar['販売戸数']
    df_melted = pd.melt(df_plot_bar, id_vars=['完成時期', '属性'], value_vars=['販売戸数', '販売済み戸数'], var_name='販売状況_raw', value_name='戸数')
    df_melted['販売状況'] = df_melted['販売状況_raw'].map({'販売戸数': '販売中', '販売済み戸数': '販売済み'})
    df_melted['plot_category'] = df_melted['属性'] + '-' + df_melted['販売状況']
    plot_df = df_melted.pivot_table(index='完成時期', columns='plot_category', values='戸数', aggfunc='sum').fillna(0)
    plot_df = plot_df.reindex(all_months, fill_value=0)
    attr_order = ['将来', '新築', '中古']; status_order = ['販売中', '販売済み']
    column_order_bottom = [f"{attr}-{status}" for attr in attr_order for status in status_order]
    for col in column_order_bottom:
        if col not in plot_df.columns: plot_df[col] = 0
    plot_df = plot_df[column_order_bottom]
    color_map_bar_bottom = {'将来-販売中': '#b22222', '将来-販売済み': '#ff6347', '新築-販売中': '#006400', '新築-販売済み': '#90ee90', '中古-販売中': '#00008b', '中古-販売済み': '#add8e6'}
    colors_bar_bottom = [color_map_bar_bottom.get(col, 'grey') for col in plot_df.columns]
    plot_df.plot(kind='bar', stacked=True, color=colors_bar_bottom, ax=ax1_bottom, legend=None)
    ax1_bottom.set_ylabel('物件戸数', fontsize=12)
    ax1_bottom.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax1_bottom.grid(True, axis='x', linestyle=':', color='gray', alpha=0.7)
    ax1_bottom.tick_params(axis='x', rotation=45, labelsize=11)
    ax1_bottom.set_xlabel('完成時期', fontsize=12)
    ax2_bottom.plot(range(len(all_months)), rate_by_month.values, color='gold', marker='o', linestyle='--', zorder=10)
    if np.sum(valid_mask) > 1:
        sns.regplot(x=x_numeric[valid_mask], y=y_values[valid_mask], ax=ax2_bottom, scatter=False,
                    ci=None, color='orange', line_kws={'linestyle':'-', 'linewidth': 3, 'alpha': 0.4})
    ax2_bottom.grid(False)
    ax2_bottom.set_ylim(bottom=0)
    ax2_bottom.set_ylabel('売れ残り率 (%)', fontsize=12, color='black')
    from matplotlib.patches import Patch # ここで明示的にインポート
    handles_bar_manual = [Patch(facecolor=color_map_bar_bottom[name], label=name) for name in column_order_bottom if name in color_map_bar_bottom]
    ax1_bottom.legend(handles=handles_bar_manual, loc='upper left', ncol=2)

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_path = os.path.join(save_path, f'analysis_03_monthly_combined_chart_{date_str}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f" -> 一時保存完了: {os.path.basename(output_path)}")
    return output_path

def analyze_bubble_chart(df_detailed: pd.DataFrame, df_summary: pd.DataFrame, df_offices: pd.DataFrame, date_str: str, save_path: str) -> tuple:
    """4. 販売戸数、価格、売れ残り率、営業所数をバブルチャートで可視化・分析する。"""
    print(" - グラフ 4/4: バブルチャート分析を作成中...")
    sales = df_detailed.groupby('都道府県')['販売戸数'].sum()
    prices = df_detailed.groupby('都道府県')['価格（平均）'].mean()
    if '売れ残り率_数値' not in df_summary.columns:
        if '売れ残り率' in df_summary.columns:
            df_summary['売れ残り率_数値'] = df_summary['売れ残り率'].str.replace('%', '').astype(float)
        else:
            print(" -> スキップ: グラフ4に必須の'売れ残り率'データがありません。")
            return None, None
    rates = df_summary.groupby('都道府県')['売れ残り率_数値'].mean()
    bubble_df = pd.concat([sales, prices, rates], axis=1).dropna()
    bubble_df.columns = ['販売戸数', '価格（平均）', '売れ残り率']

    if df_offices is not None:
        office_counts = df_offices.set_index('都道府県')['営業所数']
        bubble_df = bubble_df.join(office_counts).fillna(0)
        bubble_df['営業所数'] = bubble_df['営業所数'].astype(int)
    else:
        bubble_df['営業所数'] = 0

    if bubble_df.empty:
        print(" -> スキップ: グラフ4の対象データがありません。")
        return None, None

    office_counts_unique = sorted(bubble_df['営業所数'].unique())
    palette_custom = {}
    positive_counts = [c for c in office_counts_unique if c > 0]
    if positive_counts:
        colors = plt.get_cmap('viridis', len(positive_counts) + 2)
        for i, count in enumerate(positive_counts):
            palette_custom[count] = colors(i + 1)
    if 0 in office_counts_unique:
        palette_custom[0] = 'lightgrey'


    fig, ax = plt.subplots(figsize=(16, 10))
    plot = sns.scatterplot(
        data=bubble_df,
        x='売れ残り率',
        y='価格（平均）',
        size='販売戸数',
        sizes=(50, 2000),
        hue='営業所数',
        palette=palette_custom,
        hue_order=office_counts_unique,
        alpha=0.8,
        edgecolor='dimgrey',
        linewidth=1,
        ax=ax
    )

    plt.title('4. 販売戸数、価格、売れ残り率、営業所数のバブルチャート分析', fontsize=20)
    plt.xlabel('売れ残り率 (%)'); plt.ylabel('平均価格 (万円)')
    plt.grid(True, linestyle='--', alpha=0.6)
    for i in range(bubble_df.shape[0]):
        plt.text(x=bubble_df['売れ残り率'].iloc[i] + 0.1, y=bubble_df['価格（平均）'].iloc[i] + 0.1, s=bubble_df.index[i], fontdict=dict(color='black',size=10))
    handles, labels = plot.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title="営業所数 / 販売戸数")
    output_path = os.path.join(save_path, f'analysis_04_bubble_chart_{date_str}.png')
    plt.savefig(output_path, bbox_inches='tight'); plt.close(fig)
    print(f" -> 一時保存完了: {os.path.basename(output_path)}")

    top3_sales = bubble_df.nlargest(3, '販売戸数'); bottom3_sales = bubble_df.nsmallest(3, '販売戸数')
    top3_rates = bubble_df.nlargest(3, '売れ残り率'); bottom3_rates = bubble_df.nsmallest(3, '売れ残り率')
    top3_prices = bubble_df.nlargest(3, '価格（平均）'); bottom3_prices = bubble_df.nsmallest(3, '価格（平均）')
    top3_offices = bubble_df.nlargest(3, '営業所数'); bottom3_offices = bubble_df[bubble_df['営業所数'] > 0].nsmallest(3, '営業所数')

    lines = ["・販売規模（円の大きさ）\n [Top 3]"]; [lines.append(f" {i}. {p} ({r['販売戸数']:,}戸)") for i, (p, r) in enumerate(top3_sales.iterrows(), 1)]
    lines.append(" [Bottom 3]"); [lines.append(f" {i}. {p} ({r['販売戸数']:,}戸)") for i, (p, r) in enumerate(bottom3_sales.iterrows(), 1)]
    lines.append("\n・売れ残り率（横軸）\n [Top 3 - 課題]"); [lines.append(f" {i}. {p} ({r['売れ残り率']:.1f}%)") for i, (p, r) in enumerate(top3_rates.iterrows(), 1)]
    lines.append(" [Bottom 3 - 優良]"); [lines.append(f" {i}. {p} ({r['売れ残り率']:.1f}%)") for i, (p, r) in enumerate(bottom3_rates.iterrows(), 1)]
    lines.append("\n・平均価格（縦軸）\n [Top 3]"); [lines.append(f" {i}. {p} ({r['価格（平均）']:,.0f}万円)") for i, (p, r) in enumerate(top3_prices.iterrows(), 1)]
    lines.append(" [Bottom 3]"); [lines.append(f" {i}. {p} ({r['価格（平均）']:,.0f}万円)") for i, (p, r) in enumerate(bottom3_prices.iterrows(), 1)]
    lines.append("\n・営業所数（色の濃淡）\n [Top 3]"); [lines.append(f" {i}. {p} ({r['営業所数']:,}箇所)") for i, (p, r) in enumerate(top3_offices.iterrows(), 1)]
    lines.append(" [Bottom 3 (1箇所以上)]"); [lines.append(f" {i}. {p} ({r['営業所数']:,}箇所)") for i, (p, r) in enumerate(bottom3_offices.iterrows(), 1)]

    return output_path, "\n".join(lines)

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    """スクリプト全体の処理を実行するメイン関数。"""
    # Google Driveパスではなく、GitHub Actionsのローカルパスを使用
    BASE_LOCAL_PATH = './TamaHome_CSV_Data' # 前のスクリプトと同じパス

    # os.makedirs(BASE_LOCAL_PATH, exist_ok=True) # ディレクトリはダウンロードステップで作成されるか、既存

    df_detailed, df_pref_summary, df_monthly_summary, df_offices, data_folder_path = load_data_from_local(BASE_LOCAL_PATH)
    if df_detailed is None:
        print("\n分析を中止します。")
        return

    # data_folder_pathがNoneになる可能性があるので、ここで改めて日付文字列を生成
    # 読み込みに成功したdetailed_fileから日付を取得するのが確実
    latest_detailed_file_list = glob.glob(os.path.join(BASE_LOCAL_PATH, '*/tamahome_detailed_list_*.csv'))
    if latest_detailed_file_list:
        # 最新のファイルパスから日付部分を抽出
        latest_file_name = os.path.basename(sorted(latest_detailed_file_list, reverse=True)[0])
        match = re.search(r'tamahome_detailed_list_(\d{8})\.csv', latest_file_name)
        if match:
            date_str = match.group(1)
        else:
            date_str = datetime.now().strftime('%Y%m%d') # フォールバック
            print(f"警告: 最新ファイル名から日付を抽出できませんでした。現在のタイムスタンプ '{date_str}' を使用します。")
    else:
        date_str = datetime.now().strftime('%Y%m%d') # フォールバック
        print(f"警告: 詳細リストファイルが見つかりません。現在のタイムスタンプ '{date_str}' を使用します。")

    # グラフの保存先は、日付フォルダ内
    # load_data_from_localから返されるlatest_folderを使用
    # data_folder_pathがNoneの場合はBASE_LOCAL_PATH直下を指す
    if not data_folder_path: # load_data_from_localがエラーでdata_folder_pathをNoneで返した場合
        # このシナリオでは分析は中止されるはずだが、念のためグラフ保存パスを設定
        data_folder_path = os.path.join(BASE_LOCAL_PATH, date_str)
        os.makedirs(data_folder_path, exist_ok=True) # フォルダが存在しない場合のみ作成

    print(f"\n--- 分析グラフの生成を開始します ---")
    image_paths = []
    bubble_analysis_text = ""

    # 各グラフ作成関数はNoneを返す場合があるので、それを考慮
    path1 = analyze_attribute_by_month(df_detailed.copy(), date_str, data_folder_path)
    if path1: image_paths.append(path1)

    path2 = analyze_combined_prefecture_view_detailed(df_detailed.copy(), df_pref_summary.copy(), df_offices.copy() if df_offices is not None else None, date_str, data_folder_path)
    if path2: image_paths.append(path2)

    path3 = analyze_combined_5_and_6(df_detailed.copy(), df_monthly_summary.copy(), date_str, data_folder_path)
    if path3: image_paths.append(path3)

    bubble_path, bubble_text = analyze_bubble_chart(df_detailed.copy(), df_pref_summary.copy(), df_offices.copy() if df_offices is not None else None, date_str, data_folder_path)
    if bubble_path:
        image_paths.append(bubble_path)
        if bubble_text:
            bubble_analysis_text = bubble_text

    # image_paths = [path for path in image_paths if path is not None] # 上記でNoneチェック済み

    print("\n--- すべての画像の一時保存が完了しました ---")

    pdf_filename = f"analysis_summary_{date_str}.pdf"
    pdf_path = os.path.join(data_folder_path, pdf_filename)
    if image_paths:
        print(f"\n--- 全グラフをPDFにまとめて保存します ---")
        try:
            images_for_pdf = [PILImage.open(p).convert("RGB") for p in image_paths]
            if images_for_pdf:
                # 最初の画像でPDFを作成し、残りを追記
                images_for_pdf[0].save(pdf_path, save_all=True, append_images=images_for_pdf[1:])
                print(f"-> PDFを '{pdf_path}' に保存しました。")
        except Exception as e:
            print(f"エラー: PDFの作成に失敗しました - {e}")
    else:
        print("生成されたグラフがないため、PDFは作成されません。")

    # GitHub Actionsのログにはテキストでバブルチャート分析結果を表示
    if bubble_analysis_text:
        print("\n--- バブルチャート分析結果 ---")
        print(bubble_analysis_text)

    # 一時保存した画像ファイルはアーティファクトとして保存されるため、削除しない
    # GitHub Actionsのランナーはジョブ完了後に自動的にクリーンアップされる
    # print("\n--- 一時保存した画像ファイルを削除します ---")
    # deleted_count = 0
    # for path in image_paths:
    #     try:
    #         os.remove(path)
    #         deleted_count += 1
    #     except OSError as e:
    #         print(f"エラー: ファイル '{os.path.basename(path)}' の削除に失敗しました - {e}")
    # print(f"-> {deleted_count}個の画像ファイルの削除が完了しました。")

# ==============================================================================
# スクリプトの実行
# ==============================================================================
if __name__ == '__main__':
    main()