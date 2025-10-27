# app.py

import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import (
    load_docs,
    split_docs,
    build_vectorstore,
    load_vectorstore,
    make_qa_chain
)

load_dotenv()

st.set_page_config(
    page_title="나만의 RAG 챗봇",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 나만의 RAG 챗봇")
st.write("내가 넣은 문서(docs 폴더)를 기반으로 답해주는 검색+생성(QA) 봇입니다.")

# --- 사이드바: 초기 설정 영역 ---
st.sidebar.header("세팅 / 인덱스 관리")

if st.sidebar.button("🔄 벡터DB (재)생성하기"):
    with st.spinner("문서를 불러오고, 쪼개고, 임베딩 중입니다..."):
        docs = load_docs("docs")
        chunks = split_docs(docs)
        build_vectorstore(chunks, save_path="vectorstore")
    st.sidebar.success("벡터스토어 준비 완료!")

# 벡터스토어 로딩 시도
try:
    vectordb = load_vectorstore("vectorstore")
    qa_chain = make_qa_chain(vectordb)
    ready = True
except Exception as e:
    ready = False
    st.sidebar.warning("❗ 아직 벡터스토어가 없거나 로드 실패했어요. 먼저 '벡터DB (재)생성하기'를 눌러주세요.")
    st.sidebar.code(str(e))

# --- 메인 영역: 챗 인터페이스 ---
if "history" not in st.session_state:
    st.session_state["history"] = []

user_question = st.text_input("질문을 입력하세요:")

ask_btn = st.button("질문하기", disabled=not ready)

if ask_btn:
    if user_question.strip() == "":
        st.error("질문을 입력해 주세요.")
    else:
        with st.spinner("답변 생성 중..."):
            result = qa_chain({"query": user_question})

            answer = result["result"]
            sources = result["source_documents"]

        # 기록 저장
        st.session_state["history"].append({
            "question": user_question,
            "answer": answer,
            "sources": [s.metadata.get("source", "") for s in sources]
        })

# --- 대화 기록 렌더링 ---
st.subheader("💬 대화 기록")

for turn in reversed(st.session_state["history"]):
    st.markdown(f"**🙋 질문:** {turn['question']}")
    st.markdown(f"**🤖 답변:** {turn['answer']}")
    with st.expander("참고한 문서 조각 보기"):
        for src in turn["sources"]:
            st.write(f"- {src}")
    st.markdown("---")
