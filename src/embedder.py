import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

# モデルとDBの初期化
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "./chroma_db"

def load_data():
    batters = pd.read_csv("data/batters_2024.csv", encoding="utf-8-sig")
    pitchers = pd.read_csv("data/pitchers_2024.csv", encoding="utf-8-sig")
    return batters, pitchers

def batter_to_text(row):
    return (
        f"{row['選手']}（{row['チーム']}・{row['リーグ']}）の2024年成績："
        f"打率{row['打率']}、{row['試合']}試合出場、"
        f"本塁打{row['本塁打']}本、打点{row['打点']}、"
        f"安打{row['安打']}本、長打率{row['長打率']}、出塁率{row['出塁率']}。"
    )

def pitcher_to_text(row):
    return (
        f"{row['投手']}（{row['チーム']}・{row['リーグ']}）の2024年成績："
        f"防御率{row['防御率']}、{row['勝利']}勝{row['敗北']}敗、"
        f"登板{row['登板']}試合、三振{row['三振']}個、"
        f"投球回{row['投球回']}回。"
    )

def register_to_db(batters, pitchers):
    print("Embeddingモデルを読み込み中...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_PATH)

    # 既存コレクションをリセット
    for name in ["batters", "pitchers"]:
        try:
            client.delete_collection(name)
        except:
            pass

    # 打者データを登録
    print("打者データをベクトル化して登録中...")
    batter_texts = [batter_to_text(row) for _, row in batters.iterrows()]
    batter_vectors = model.encode(batter_texts).tolist()
    batter_col = client.create_collection("batters")
    batter_col.add(
        documents=batter_texts,
        embeddings=batter_vectors,
        ids=[f"batter_{i}" for i in range(len(batter_texts))]
    )
    print(f"打者データ登録完了: {len(batter_texts)}件")

    # 投手データを登録
    print("投手データをベクトル化して登録中...")
    pitcher_texts = [pitcher_to_text(row) for _, row in pitchers.iterrows()]
    pitcher_vectors = model.encode(pitcher_texts).tolist()
    pitcher_col = client.create_collection("pitchers")
    pitcher_col.add(
        documents=pitcher_texts,
        embeddings=pitcher_vectors,
        ids=[f"pitcher_{i}" for i in range(len(pitcher_texts))]
    )
    print(f"投手データ登録完了: {len(pitcher_texts)}件")

    # 動作確認
    print("\n=== ベクトル検索テスト ===")
    query = "本塁打が多い打者"
    query_vec = model.encode([query]).tolist()
    results = batter_col.query(query_embeddings=query_vec, n_results=3)
    print(f"クエリ: '{query}'")
    for doc in results["documents"][0]:
        print(f"  → {doc}")

if __name__ == "__main__":
    batters, pitchers = load_data()
    register_to_db(batters, pitchers)
    print("\nDB登録完了！rag_engine.pyに進んでください。")
