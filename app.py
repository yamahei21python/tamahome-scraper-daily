# app.py
import streamlit as st
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime

# ★★★★★ デバッグ用コードを追加 ★★★★★
st.write(f"Streamlit version: {st.__version__}")
# ★★★★★★★★★★★★★★★★★★★★★

# --- 定数設定 ---
GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "yamahei21python" # あなたのGitHubユーザー名
REPO_NAME = "tamahome-scraper-daily" # あなたのリポジトリ名

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここを修正: 正しいワークフローファイル名に変更 ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
WORKFLOW_FILENAME = "scheduled-scraper.yml" # 実際のファイル名に合わせる
ARTIFACT_NAME = "daily-analysis-report" # アップロードされるアーティファクト名

# GitHub Personal Access Token (PAT) をStreamlitのシークレットから取得
try:
    GITHUB_TOKEN = st.secrets["github"]["token"]
except (KeyError, FileNotFoundError):
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

@st.cache_data(ttl=86400) # 1日キャッシュ
def get_workflow_id_by_filename(filename: str) -> int:
    """ワークフローのファイル名からワークフローIDを取得する"""
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    workflows = response.json()["workflows"]
    for wf in workflows:
        if wf["path"].endswith(filename):
            return wf["id"]
    raise ValueError(f"Workflow with filename '{filename}' not found.")


@st.cache_data(ttl=3600) # 1時間結果をキャッシュ
def get_workflow_runs(workflow_id: int):
    """ワークフローの実行履歴を取得する"""
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{workflow_id}/runs"
    params = {"status": "success", "per_page": 30} # 成功した実行を30件まで取得
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

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for filename in z.namelist():
            if filename.lower().endswith(".pdf"):
                return z.read(filename)
    return None

# --- メインロジック ---

if not GITHUB_TOKEN:
    st.error("GitHubのアクセストークンが設定されていません。Streamlitのシークレットに `github.token` を設定してください。")
else:
    try:
        # 0. ファイル名からワークフローIDを取得
        workflow_id = get_workflow_id_by_filename(WORKFLOW_FILENAME)

        # 1. ワークフローの実行履歴を取得
        runs = get_workflow_runs(workflow_id)
        
        if not runs:
            st.warning("成功したワークフローの実行が見つかりませんでした。")
        else:
            # 2. 日付選択のためのデータフレームを作成
            run_data = []
            for run in runs:
                run_artifacts = get_artifacts_for_run(run["id"])
                for artifact in run_artifacts:
                    if artifact["name"] == ARTIFACT_NAME and not artifact["expired"]:
                        run_data.append({
                            "display_name": f"{datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')).strftime('%Y年%m月%d日 %H:%M')} (ID: {run['id']})",
                            "run_id": run["id"],
                            "artifact_url": artifact["archive_download_url"]
                        })
                        break 
            
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
                        st.pdf(pdf_bytes, height=1000)
                    else:
                        st.error("PDFファイルの取得または展開に失敗しました。")

    except requests.exceptions.RequestException as e:
        st.error(f"GitHub APIへのアクセス中にエラーが発生しました: {e}")
        st.error(f"URL: {e.request.url}") # デバッグ用にURLを表示
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")
