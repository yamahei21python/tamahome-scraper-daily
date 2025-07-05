# app.py
import streamlit as st
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime
from PIL import Image

# --- 定数設定 ---
GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "yamahei21python" 
REPO_NAME = "tamahome-scraper-daily"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここを修正: 実際に使用しているファイル名に合わせる ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
WORKFLOW_FILENAME = "scheduled-scraper.yml" 
ARTIFACT_NAME = "daily-analysis-report" 

try:
    GITHUB_TOKEN = st.secrets["github"]["token"]
except (KeyError, FileNotFoundError):
    import os
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}",
}

st.set_page_config(page_title="タマホーム分析レポート", layout="wide")
st.title("📊 タマホーム 日次分析レポートビューア")

# --- 関数定義 ---
@st.cache_data(ttl=86400)
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

@st.cache_data(ttl=3600)
def get_workflow_runs(workflow_id: int):
    """ワークフローの実行履歴を取得する"""
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{workflow_id}/runs"
    params = {"status": "success", "per_page": 30}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()["workflow_runs"]

@st.cache_data(ttl=3600)
def get_artifacts_for_run(run_id):
    """特定の実行IDに紐づくアーティファクトの情報を取得する"""
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}/artifacts"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()["artifacts"]

@st.cache_data(ttl=86400)
def download_and_extract_images(artifact_url):
    """アーティファクトをダウンロードし、画像とテキストを抽出する"""
    response = requests.get(artifact_url, headers=HEADERS, stream=True)
    response.raise_for_status()
    images = {}
    analysis_text = ""
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for filename in sorted(z.namelist()):
            if filename.lower().endswith(".png"):
                img_bytes = z.read(filename)
                images[filename] = Image.open(io.BytesIO(img_bytes))
            elif filename.lower().endswith(".txt"):
                analysis_text = z.read(filename).decode('utf-8')
    return images, analysis_text

# --- メインロジック ---
if not GITHUB_TOKEN:
    st.error("GitHubのアクセストークンが設定されていません。")
else:
    try:
        workflow_id = get_workflow_id_by_filename(WORKFLOW_FILENAME)
        runs = get_workflow_runs(workflow_id)
        
        if not runs:
            st.warning("成功したワークフローの実行が見つかりませんでした。")
        else:
            run_data = []
            for run in runs:
                run_artifacts = get_artifacts_for_run(run["id"])
                for artifact in run_artifacts:
                    if artifact["name"] == ARTIFACT_NAME and not artifact["expired"]:
                        run_data.append({
                            "display_name": f"{datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')).strftime('%Y年%m月%d日 %H:%M')} (ID: {run['id']})",
                            "artifact_url": artifact["archive_download_url"]
                        })
                        break 
            
            if not run_data:
                st.warning(f"'{ARTIFACT_NAME}' という名前のアーティファクトが見つかりませんでした。")
            else:
                df_runs = pd.DataFrame(run_data)
                selected_run_display_name = st.selectbox(
                    "表示したいレポートの日付を選択してください:",
                    df_runs["display_name"]
                )
                
                if selected_run_display_name:
                    selected_artifact_url = df_runs[df_runs["display_name"] == selected_run_display_name].iloc[0]["artifact_url"]
                    
                    with st.spinner("レポートをダウンロード中..."):
                        images, analysis_text = download_and_extract_images(selected_artifact_url)
                    
                    if images:
                        st.header("分析グラフ")
                        
                        # ★★★★★★★★★★ ここからが修正箇所 ★★★★★★★★★★
                        for filename, img in images.items():
                            # ファイル名で判定して表示方法を切り替える
                            if filename.startswith("01_attribute_pie"):
                                # 円グラフの場合：3列レイアウトで中央揃え＆サイズ調整
                                col1, col2, col3 = st.columns([1, 2, 1]) # 真ん中の列を広く取る
                                with col2:
                                    st.image(img, caption=filename, use_column_width=True) # 列の幅に合わせて表示
                            else:
                                # その他のグラフの場合：通常通り全幅で表示
                                st.image(img, caption=filename, use_column_width=True)
                        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                        
                        if analysis_text:
                            st.header("バブルチャート分析結果")
                            st.text(analysis_text)
                    else:
                        st.error("レポート画像の取得または展開に失敗しました。")

    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")
