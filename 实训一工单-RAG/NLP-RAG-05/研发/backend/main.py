# 工单编号：人工智能 NLP-RAG-图像内容解析及检索优化

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.rag import router as rag_router


app = FastAPI(
    title="PDF Prospectus RAG QA",
    description="RAG QA system for 招股说明书1.pdf",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(rag_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "PDF RAG service is running."}
