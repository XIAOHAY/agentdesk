# 系统架构说明（虚构，用于测试）

## 向量数据库
系统使用 Qdrant 作为向量数据库，存储文档 embedding，支持余弦相似度检索。
也可切换为 PGVector 复用 Postgres。

## 缓存
使用 Redis 缓存高频查询结果与 embedding，默认 TTL 为 3600 秒，降低重复检索延迟。

## 部署
后端基于 FastAPI，通过 Docker Compose 编排 API、向量库与 Redis 三个服务。
