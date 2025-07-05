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
st.markdown(f"リポジトリ: [{REPO_OWNER}/{REPO_NAME}](https://github.com/{REPO_OWNER}/{REPO_NAME})")

# --- 関数定義 ---
# ... (get_workflow_id_by_filename, get_workflow_runs, get_artifacts_for_run は変更なし) ...
@st.cache_data(ttl=86400)
def get_workflow_id_by_filename(filename: str):
    # ...
    pass

@st.cache_data(ttl=3600)
def get_workflow_runs(workflow_id: int):
    # ...
    pass

@st.cache_data(ttl=3600)
def get_artifacts_for_run(run_id):
    # ...
    pass

@st.cache_data(ttl=86400)
def download_and_extract_images(artifact_url):
    """アーティファクトをダウンロードし、画像とテキストを抽出する"""
    response = requests.get(artifact_url, headers=HEADERS, stream=True)
    response.raise_for_status()

    images = {}
    analysis_text = ""
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # zipファイル内のファイルをループ
        for filename in sorted(z.namelist()): # ファイル名でソートして順番を担保
            if filename.lower().endswith(".png"):
                # 画像ファイルの場合
                img_bytes = z.read(filename)
                images[filename] = Image.open(io.BytesIO(img_bytes))
            elif filename.lower().endswith(".txt"):
                # テキストファイルの場合
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
                        # ★★★★★★★★★★ ここからが修正箇所 ★★★★★★★★★★
                        images, analysis_text = download_and_extract_images(selected_artifact_url)
                    
                    if images:
                        st.header("分析グラフ")
                        for filename, img in images.items():
                            st.image(img, caption=filename, use_column_width=True)
                        
                        if analysis_text:
                            st.header("バブルチャート分析結果")
                            st.text(analysis_text)
                    else:
                        st.error("レポート画像の取得または展開に失敗しました。")

    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")
