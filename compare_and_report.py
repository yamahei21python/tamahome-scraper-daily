# -*- coding: utf-8 -*-
"""
【GitHub Actions対応版】
ローカルフォルダに保存されたタマホームの物件情報CSVを比較し、
詳細な分析レポートを生成・保存し、さらにアウトプットにも表示するスクリプト。
"""

import pandas as pd
import numpy as np
import glob
import os
# from IPython.display import display, HTML # GitHub Actionsでは不要なので削除
from datetime import datetime
from zoneinfo import ZoneInfo # Python 3.9以降で利用可能
import re
from typing import List, Dict, Any, Optional

# ==============================================================================
# データ読み込み関数 (パスをローカル向けに修正)
# ==============================================================================
def find_comparison_files_from_local(base_local_path: str) -> tuple:
    print(f"--- ローカルのベースパス '{base_local_path}' から日別フォルダを検索します ---")

    # 指定されたパスが存在するか確認
    if not os.path.exists(base_local_path):
        print(f"エラー: 指定されたベースパス '{base_local_path}' が存在しません。")
        return None, None, None

    all_subdirs = glob.glob(os.path.join(base_local_path, '*/'))
    date_folders = [d for d in all_subdirs if os.path.isdir(d) and os.path.basename(os.path.normpath(d)).isdigit() and len(os.path.basename(os.path.normpath(d))) == 8]

    if len(date_folders) < 2:
        print("エラー: 比較対象となる日別フォルダが2つ以上見つかりません。")
        print(f"指定されたベースパス '{base_local_path}' に、日付名のフォルダが2つ以上あるか確認してください。")
        return None, None, None

    sorted_folders = sorted(date_folders, reverse=True)

    latest_folder_path = sorted_folders[0]
    previous_folder_path = sorted_folders[1]

    latest_file_list = glob.glob(os.path.join(latest_folder_path, 'tamahome_detailed_list_*.csv'))
    previous_file_list = glob.glob(os.path.join(previous_folder_path, 'tamahome_detailed_list_*.csv'))

    if not latest_file_list or not previous_file_list:
        print("エラー: 最新または前回のフォルダ内に 'tamahome_detailed_list_*.csv' が見つかりません。")
        return None, None, None

    latest_file = latest_file_list[0]
    previous_file = previous_file_list[0]

    print(f"\n本日分のデータ (最新): {os.path.basename(latest_file)} (from {os.path.basename(os.path.normpath(latest_folder_path))})")
    print(f"前回分のデータ (比較対象): {os.path.basename(previous_file)} (from {os.path.basename(os.path.normpath(previous_folder_path))})")

    return previous_file, latest_file, latest_folder_path


# ==============================================================================
# 差分分析とレポート生成 (display関数の置き換え)
# ==============================================================================
def analyze_and_compare_lists(previous_path: str, latest_path: str, report_save_path: str, jst_now):
    """
    2つの物件リストを比較し、基本差分と詳細な分析レポートを生成・表示する。
    """
    try:
        df_previous = pd.read_csv(previous_path)
        df_latest = pd.read_csv(latest_path)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
        return

    # --- 1. 基本的な差分（新規・終了・更新）の検出 ---

    merge_keys = ['都道府県', '物件名']
    # URL列が両方のDFに存在する場合のみ、マージキーに追加する
    if 'URL' in df_previous.columns and 'URL' in df_latest.columns:
        merge_keys.append('URL')
        df_previous['複合キー'] = df_previous.apply(lambda row: '_'.join([str(row[k]) for k in merge_keys]), axis=1)
        df_latest['複合キー'] = df_latest.apply(lambda row: '_'.join([str(row[k]) for k in merge_keys]), axis=1)
    else:
        print("警告: 比較対象のCSVにURL列がありません。物件名のみで比較します。")
        df_previous['複合キー'] = df_previous[merge_keys[0]].astype(str) + '_' + df_previous[merge_keys[1]].astype(str)
        df_latest['複合キー'] = df_latest[merge_keys[0]].astype(str) + '_' + df_latest[merge_keys[1]].astype(str)


    merged_df = pd.merge(df_previous, df_latest, on='複合キー', how='outer', suffixes=('_前回', '_今回'), indicator=True)

    new_properties_df = merged_df[merged_df['_merge'] == 'right_only']
    removed_properties_df = merged_df[merged_df['_merge'] == 'left_only']
    both_df = merged_df[merged_df['_merge'] == 'both'].copy()

    # --- コンソールへの基本差分表示 ---

    # GitHub Actionsではハイパーリンク付きHTMLは表示されないため、URLを直接含める形で表示
    def format_output_row(row, name_col, url_col):
        url = row.get(url_col)
        if pd.isna(url) or url == '':
            return f"{row[name_col]}"
        else:
            return f"{row[name_col]} ({url})"

    print("\n" + "="*50 + "\n【1. 新規物件】: " + str(len(new_properties_df)) + "件\n" + "="*50)
    if not new_properties_df.empty:
        display_df = new_properties_df.copy()
        if 'URL_今回' in display_df.columns:
            display_df['物件名'] = display_df.apply(format_output_row, args=('物件名_今回', 'URL_今回'), axis=1)
            cols_to_display = ['都道府県_今回', '物件名', '属性_今回', '価格_今回', '販売戸数_今回']
        else:
            cols_to_display = ['都道府県_今回', '物件名_今回', '属性_今回', '価格_今回', '販売戸数_今回']

        display_df = display_df[cols_to_display].rename(columns=lambda c: c.replace('_今回', ''))
        print(display_df.to_string(index=False)) # display(HTML(...)) の代わりにto_string()を使用
    else:
        print("新規物件はありませんでした。")

    print("\n" + "="*50 + "\n【2. 掲載終了物件】: " + str(len(removed_properties_df)) + "件\n" + "="*50)
    if not removed_properties_df.empty:
        display_df = removed_properties_df.copy()
        if 'URL_前回' in display_df.columns:
            display_df['物件名'] = display_df.apply(format_output_row, args=('物件名_前回', 'URL_前回'), axis=1)
            cols_to_display = ['都道府県_前回', '物件名', '属性_前回', '価格_前回', '販売戸数_前回']
        else:
            cols_to_display = ['都道府県_前回', '物件名_前回', '属性_前回', '価格_前回', '販売戸数_前回']

        display_df = display_df[cols_to_display].rename(columns=lambda c: c.replace('_前回', ''))
        print(display_df.to_string(index=False)) # display(HTML(...)) の代わりにto_string()を使用
    else:
        print("掲載終了物件はありませんでした。")

    updated_records = []
    compare_cols = ['属性', '完成時期', '総戸数', '販売戸数', '価格（平均）', '価格']
    for _, row in both_df.iterrows():
        updates = {}
        for col in compare_cols:
            val_prev, val_latest = row[f'{col}_前回'], row[f'{col}_今回']
            is_diff = False
            if pd.isna(val_prev) and pd.isna(val_latest): is_diff = False
            elif pd.isna(val_prev) or pd.isna(val_latest): is_diff = True
            else:
                try:
                    # 数値型の場合の比較を強化
                    prev_is_numeric = isinstance(val_prev, (int, float)) or (isinstance(val_prev, str) and val_prev.replace(',', '').replace('.', '').isdigit())
                    latest_is_numeric = isinstance(val_latest, (int, float)) or (isinstance(val_latest, str) and val_latest.replace(',', '').replace('.', '').isdigit())

                    if prev_is_numeric and latest_is_numeric:
                        # 両方数値に変換可能なら数値として比較
                        if not np.isclose(float(str(val_prev).replace(',', '')), float(str(val_latest).replace(',', ''))): is_diff = True
                    else:
                        # それ以外は文字列として比較
                        if str(val_prev) != str(val_latest): is_diff = True
                except (ValueError, TypeError): # 変換できない場合も文字列比較にフォールバック
                    if str(val_prev) != str(val_latest): is_diff = True
            if is_diff:
                updates[col] = f"'{val_prev}' → '{val_latest}'"
        if updates:
            record_data = {'都道府県': row['都道府県_今回'], '物件名': row['物件名_今回'], '変更内容': updates}
            if 'URL_今回' in row: record_data['URL'] = row['URL_今回']
            updated_records.append(record_data)

    print("\n" + "="*50 + "\n【3. 情報更新物件】: " + str(len(updated_records)) + "件\n" + "="*50)
    if updated_records:
        # display(HTML(...)) の代わりにprint()と整形済み文字列を使用
        for record in updated_records:
            url = record.get('URL', '')
            url_display = f" ({url})" if url else ""
            print(f"■ {record['都道府県']} / {record['物件名']}{url_display}")
            for field, change in record['変更内容'].items():
                print(f"  - {field}: {change}")
        print("\n")
    else:
        print("情報が更新された物件はありませんでした。")

    # --- 2. 詳細な分析レポートの生成 ---
    report_lines = []
    now_str = jst_now.strftime('%Y-%m-%d %H:%M:%S JST')
    report_lines.append("="*60)
    report_lines.append(f"タマホーム物件情報 比較分析レポート ({now_str})")
    report_lines.append("="*60)
    report_lines.append(f"比較対象: {os.path.basename(previous_path)} vs {os.path.basename(latest_path)}")

    report_lines.append("\n\n--- A. 総合サマリー ---")
    summary_data = {'指標': ['総物件数', '総販売戸数', '価格（平均）'],
        '前回': [len(df_previous), df_previous['販売戸数'].sum(), f"{df_previous['価格（平均）'].mean():.0f} 万円"],
        '今回': [len(df_latest), df_latest['販売戸数'].sum(), f"{df_latest['価格（平均）'].mean():.0f} 万円"]}
    report_lines.append(pd.DataFrame(summary_data).to_string(index=False))

    report_lines.append("\n\n--- B. 都道府県別の変動サマリー ---")
    new_counts = new_properties_df['都道府県_今回'].value_counts()
    removed_counts = removed_properties_df['都道府県_前回'].value_counts()
    updated_counts = pd.Series([r['都道府県'] for r in updated_records]).value_counts()
    geo_summary_df = pd.concat([new_counts, removed_counts, updated_counts], axis=1).fillna(0).astype(int)
    geo_summary_df.columns = ['新規物件数', '掲載終了数', '情報更新数']
    geo_summary_df.index.name = '都道府県'
    if geo_summary_df.empty:
        report_lines.append("  (前回データとの間に、新規・掲載終了・更新された物件はありませんでした。)")
    else: report_lines.append(geo_summary_df.to_string())

    report_lines.append("\n\n--- C. 価格変動があった物件 ---")
    price_decreased, price_increased = [], []
    for r in updated_records:
        if '価格（平均）' in r['変更内容']:
            url_text = f" ( {r['URL']} )" if 'URL' in r else ""
            try:
                # 数値部分のみを抽出し、カンマを除去してfloatに変換
                prev_price_str_match = re.search(r'\'([\d,.]+)\'', r['変更内容']['価格（平均）'])
                latest_price_str_match = re.search(r'→ \'([\d,.]+)\'', r['変更内容']['価格（平均）'])

                if prev_price_str_match and latest_price_str_match:
                    prev_price = float(prev_price_str_match.group(1).replace(',', ''))
                    latest_price = float(latest_price_str_match.group(1).replace(',', ''))

                    line = f"- {r['都道府県']} / {r['物件名']}{url_text}: {r['変更内容']['価格（平均）']}"
                    if latest_price < prev_price: price_decreased.append(line)
                    elif latest_price > prev_price: price_increased.append(line)
                else:
                    # 価格文字列の解析に失敗した場合、そのまま出力
                    line = f"- {r['都道府県']} / {r['物件名']}{url_text}: {r['変更内容']['価格（平均）']} (価格解析エラー)"
                    # どちらかのリストに追加するのではなく、ログに出力
                    print(f"Warning: Could not parse prices for {r['物件名']}: {r['変更内容']['価格（平均）']}")

            except Exception as ex: # その他の解析エラーをキャッチ
                print(f"Warning: Unexpected error parsing prices for {r['物件名']}: {ex}")
                line = f"- {r['都道府県']} / {r['物件名']}{url_text}: {r['変更内容']['価格（平均）']} (予期せぬエラー)"
            # 価格解析に失敗した行も何らかの形でレポートに含める場合はここで追加
            # 例: report_lines.append(line)

    report_lines.append("\n[価格が下落した物件]"); report_lines.extend(price_decreased if price_decreased else ["  (なし)"])
    report_lines.append("\n[価格が上昇した物件]"); report_lines.extend(price_increased if price_increased else ["  (なし)"])


    report_lines.append("\n\n--- D. 販売戸数に変動があった物件 ---")
    sales_units_changed = []
    for r in updated_records:
        if '販売戸数' in r['変更内容']:
            url_text = f" ( {r['URL']} )" if 'URL' in r else ""
            sales_units_changed.append(f"- {r['都道府県']} / {r['物件名']}{url_text}: {r['変更内容']['販売戸数']}")
    if sales_units_changed: report_lines.extend(sales_units_changed)
    else: report_lines.append("  (なし)")

    report_content = "\n".join(report_lines)
    date_str = jst_now.strftime('%Y%m%d')
    report_filename = f"comparison_report_{date_str}.txt"
    report_path = os.path.join(report_save_path, report_filename)

    with open(report_path, 'w', encoding='utf-8') as f: f.write(report_content)

    print("\n" + "="*50 + "\n【4. 詳細分析レポート】\n" + "="*50)
    print(report_content)

    print("\n" + "-"*60)
    print(f"上記の内容は、以下のファイルにも保存されています:\n{report_path}")
    return report_content

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    """スクリプト全体の処理を実行するメイン関数。"""

    # 保存パスをGitHub Actionsの実行環境内の相対パスに変更
    BASE_LOCAL_PATH = './TamaHome_CSV_Data'
    jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))

    pd.set_option('display.max_rows', 100)
    # GitHub ActionsのログではHTML表示は関係ないので、max_colwidthは不要
    # pd.set_option('display.max_colwidth', None)

    previous_file, latest_file, report_save_folder = find_comparison_files_from_local(BASE_LOCAL_PATH)

    if latest_file and previous_file:
        comparison_text = analyze_and_compare_lists(previous_file, latest_file, report_save_folder, jst_now)
        print("\n--- 比較処理が完了しました ---")
    else:
        print("比較に必要なファイルが見つからないため、比較処理はスキップされました。")

# スクリプトを実行
if __name__ == '__main__':
    main()