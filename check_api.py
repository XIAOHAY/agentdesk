"""快速自检：读 .env，直接测真实大模型这条路（embedding + chat）。
用法：在 agentdesk 目录下  python check_api.py
不会打印你的 key。"""
from __future__ import annotations
import sys
from app.config import settings

print("=" * 48)
print("base_url      :", settings.openai_base_url)
print("chat_model    :", settings.chat_model)
print("embedding     :", settings.embedding_model)
print("模式          :", "真实大模型" if settings.use_llm else "离线 fallback（没读到 key）")
print("=" * 48)

if not settings.use_llm:
    print("⚠️  没读到 OPENAI_API_KEY，请确认 .env 在 agentdesk 目录下且 key 已填。")
    sys.exit(1)

from app.llm import embed_query, chat

try:
    v = embed_query("维度探测")
    print(f"✅ embedding 成功，维度 = {len(v)}  (bge-m3 应为 1024)")
except Exception as e:
    print("❌ embedding 失败：", type(e).__name__, str(e)[:200])
    print("   常见原因：key 错 / 余额不足 / 模型名不对 / 网络不通。")
    sys.exit(1)

try:
    r = chat("你是测试助手，只回一句话。", "用一句话打个招呼。")
    print("✅ chat 成功：", r[:80])
except Exception as e:
    print("❌ chat 失败：", type(e).__name__, str(e)[:200])
    sys.exit(1)

print("\n🎉 真实大模型这条路通了。现在可以 streamlit run streamlit_app.py，")
print("   首次启动会自动用 1024 维重建索引，侧边栏会显示 🟢 真实大模型。")
