# app.py
import streamlit as st
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime

# --- 定数設定 ---
# これらはあなたのリポジトリ情報に合わせて変更してください
GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "yamahei21python" # あなたのGitHubユーザー名
REPO_NAME = "tamahome-scraper-daily" # あなたのリポジトリ名
WORKFLOW_NAME = "scheduled-run.yml" # レポートを生成するワークフローファイル名
ARTIFACT_NAME = "daily-analysis-report" # アップロードされるアーティファクト名

# GitHub Personal Access Token (PAT) をStreamlitのシークレットから取得
# ローカルでテストする場合は、環境変数に GITHUB_TOKEN を設定してください
try:
    # Streamlit Community Cloudのシークレットから読み込み
    GITHUB_TOKEN = st.secrets["github"]["token"]
except (KeyError, FileNotFoundError):
    # ローカルテスト用のフォールバック
    import os
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# --- GitHub API と通信するためのヘッダー ---
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}",
}

# --- Streamlit アプリのUI設定 ---
st.set_page_config(page_title="タマホーム分析レポート", layout="wide")
st.title("📊 タマホーム 日次分析レポートビューア")
st.markdown(f"リポジトリ: [{REPO_OWNER}/{REPO_NAME}](https://github.com/{REPO_OWNER}/{REPO_NAME})")

# --- 関数定義 ---

@st.cache_data(ttl=3600) # 1時間結果をキャッシュ
def get_workflow_runs():
    """ワークフローの実行履歴を取得する"""
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_NAME}/runs"
    params = {"status": "success"} # 成功した実行のみを取得
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()["workflow_runs"]

@st.cache_data(ttl=3600) # 1時間結果をキャッシュ
def get_artifacts_for_run(run_id):
    """特定の実行IDに紐づくアーティファクトの情報を取得する"""
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}/artifacts"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()["artifacts"]

@st.cache_data(ttl=86400) # 1日結果をキャッシュ
def download_and_extract_pdf(artifact_url):
    """アーティファクトをダウンロードし、中のPDFをメモリ上で展開する"""
    response = requests.get(artifact_url, headers=HEADERS, stream=True)
    response.raise_for_status()

    # ダウンロードしたzipファイルをメモリ上で開く
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # zipファイル内の最初のPDFファイルを探す
        for filename in z.namelist():
            if filename.lower().endswith(".pdf"):
                # PDFファイルの中身をバイトデータとして返す
                return z.read(filename)
    return None

# --- メインロジック ---

if not GITHUB_TOKEN:
    st.error("GitHubのアクセストークンが設定されていません。Streamlitのシークრitに `GITHUB_TOKEN` を設定してください。")
else:
    try:
        # 1. ワークフローの実行履歴を取得
        runs = get_workflow_runs()
        
        if not runs:
            st.warning("成功したワークフローの実行が見つかりませんでした。")
        else:
            # 2. 日付選択のためのデータフレームを作成
            run_data = []
            for run in runs:
                # アーティファクト名が一致するものを探す
                run_artifacts = get_artifacts_for_run(run["id"])
                for artifact in run_artifacts:
                    if artifact["name"] == ARTIFACT_NAME:
                        run_data.append({
                            "display_name": f"{datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')).strftime('%Y年%m月%d日 %H:%M')} (ID: {run['id']})",
                            "run_id": run["id"],
                            "artifact_url": artifact["archive_download_url"]
                        })
                        break # 一致するアーティファクトが見つかったら次のrunへ

            if not run_data:
                st.warning(f"'{ARTIFACT_NAME}' という名前のアーティファクトが見つかりませんでした。")
            else:
                df_runs = pd.DataFrame(run_data)

                # 3. セレクトボックスで表示するレポートを選択
                selected_run_display_name = st.selectbox(
                    "表示したいレポートの日付を選択してください:",
                    df_runs["display_name"]
                )
                
                if selected_run_display_name:
                    # 4. 選択されたレポートのPDFをダウンロードして表示
                    selected_artifact_url = df_runs[df_runs["display_name"] == selected_run_display_name].iloc[0]["artifact_url"]
                    
                    with st.spinner("PDFレポートをダウンロード中..."):
                        pdf_bytes = download_and_extract_pdf(selected_artifact_url)
                    
                    if pdf_bytes:
                        # Streamlitの機能でPDFを埋め込み表示
                        st.pdf(pdf_bytes, height=1000)
                    else:
                        st.error("PDFファイルの取得または展開に失敗しました。")

    except requests.exceptions.RequestException as e:
        st.error(f"GitHub APIへのアクセス中にエラーが発生しました: {e}")
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")
