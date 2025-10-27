import os
import random
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# rag_pipeline에서 함수들 가져오기
from rag_pipeline import (
    load_docs,
    split_docs,
    build_vectorstore,
    load_vectorstore,
    make_qa_chain,
)

# ---------- 페이지 설정 ----------
st.set_page_config(
    page_title="나만의 회계 튜터",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .question-box textarea {
        font-size: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 나만의 회계 RAG + 퀴즈 챗봇")
st.write("내가 넣은 강의자료(docs 폴더)와 문제은행(data/accounting_bank_full.csv)을 바탕으로 공부 도와주는 개인 튜터예요.")


# =====================================
# 1. 벡터스토어 관리 (RAG용 지식 인덱스)
# =====================================

VSTORE_PATH = "vectorstore"
DOCS_PATH = "docs"

# 전역 상태에 vectordb와 qa_fn을 저장해서 재사용하자
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None
if "qa_fn" not in st.session_state:
    st.session_state["qa_fn"] = None
if "build_error" not in st.session_state:
    st.session_state["build_error"] = None


def try_load_vectorstore():
    """이미 있는 vectorstore를 불러오거나 없으면 None."""
    try:
        vectordb = load_vectorstore(VSTORE_PATH)
        return vectordb
    except Exception as e:
        return None


def rebuild_vectorstore():
    """
    docs 폴더에서 pdf/txt를 읽고
    -> chunk로 쪼개고
    -> 새 vectorstore를 만든다.
    실패하면 에러 메시지 저장.
    """
    try:
        docs = load_docs(DOCS_PATH)  # pdf / txt 읽기
        chunks = split_docs(docs)    # 조각내기

        if len(chunks) == 0:
            raise ValueError(
                "docs 폴더에서 불러온 문서가 0개예요.\n"
                "docs 폴더 안에 회계 강의 pdf/txt 파일을 넣어주세요."
            )

        vectordb = build_vectorstore(chunks, save_path=VSTORE_PATH)
        st.session_state["vectordb"] = vectordb
        st.session_state["qa_fn"] = make_qa_chain(vectordb)
        st.session_state["build_error"] = None
        st.success("✅ 벡터스토어 생성 완료!")
    except Exception as e:
        st.session_state["build_error"] = str(e)
        st.session_state["vectordb"] = None
        st.session_state["qa_fn"] = None
        st.error(f"벡터스토어 생성 실패: {e}")


# 앱이 처음 켜질 때 시도: 이미 있는 vectorstore 불러오기
if st.session_state["vectordb"] is None:
    st.session_state["vectordb"] = try_load_vectorstore()
    if st.session_state["vectordb"] is not None:
        st.session_state["qa_fn"] = make_qa_chain(st.session_state["vectordb"])


# ----------------- 사이드바 -----------------
st.sidebar.markdown("### ⚙️ 세팅 / 인덱스 관리")

if st.sidebar.button("📂 벡터DB (재)생성하기"):
    rebuild_vectorstore()

if st.session_state["vectordb"] is None:
    st.sidebar.warning(
        "❗ 아직 벡터스토어가 없거나 로드 실패했어요.\n"
        "먼저 '벡터DB (재)생성하기'를 눌러 주세요."
    )
else:
    st.sidebar.success("✅ 벡터스토어 준비됨!")

if st.session_state["build_error"]:
    st.sidebar.code(st.session_state["build_error"], language="text")


# =====================================
# 2. 질문 → 답변 (RAG 질의응답)
# =====================================

st.markdown("## 💬 회계 질문해 보세요")

user_q = st.text_input(
    "예: '자산이 뭐예요?', '발생주의 회계 쉽게 설명해줘', '선급비용은 왜 자산이에요?' 등",
    key="question_input",
)

ask_button = st.button("질문하기")

if ask_button:
    if not user_q.strip():
        st.warning("질문을 입력해 주세요.")
    elif st.session_state["qa_fn"] is None:
        st.error("아직 QA 엔진이 준비가 안 됐어요. 왼쪽에서 벡터DB 먼저 만들거나 로드해 주세요.")
    else:
        with st.spinner("답변 생성 중..."):
            answer_text = st.session_state["qa_fn"](user_q)

        st.markdown("#### 📌 답변")
        st.write(answer_text)

        # 대화 히스토리 저장
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({"role": "user", "content": user_q})
        st.session_state["history"].append({"role": "assistant", "content": answer_text})


# =====================================
# 3. 회계 퀴즈 모드
# =====================================

st.markdown("---")
st.markdown("## 📝 회계원리 퀴즈")

@st.cache_data
def load_question_bank(csv_path="data/accounting_bank_full.csv"):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        return None

bank_df = load_question_bank()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🔎 랜덤 문제 받기")
    quiz_btn = st.button("문제 출제")

with col_right:
    difficulty_choice = st.selectbox(
        "난이도 선택(선택 시 난이도에 맞춰 랜덤):",
        ["전체", "easy", "medium", "hard"],
        index=0,
    )

if bank_df is None:
    st.error("❌ data/accounting_bank_full.csv 파일을 찾을 수 없어요. GitHub /data 에 있는 csv를 로컬 data 폴더에도 넣어주세요.")
else:
    if quiz_btn:
        # 난이도 필터
        if difficulty_choice == "전체":
            pool_df = bank_df
        else:
            pool_df = bank_df[bank_df["difficulty"] == difficulty_choice]

        if len(pool_df) == 0:
            st.warning(f"{difficulty_choice} 난이도 문제가 없습니다.")
        else:
            row = pool_df.sample(1).iloc[0]

            week = row.get("week", "N/A")
            topic = row.get("topic", "")
            question = row.get("question", "")
            choices_raw = str(row.get("choices", ""))
            answer = row.get("answer", "")
            explanation = row.get("explanation", "")

            st.markdown(f"**📚 주차:** {week}주차  /  **주제:** {topic}")
            st.markdown("**❓ 문제**")
            st.write(question)

            if isinstance(choices_raw, str) and choices_raw.strip() != "" and choices_raw.lower() != "nan":
                st.markdown("**보기**")
                for choice in choices_raw.split("|"):
                    st.write("- " + choice.strip())

            with st.expander("✅ 정답 보기 / 해설 보기"):
                st.markdown("**정답:**")
                st.write(answer)
                st.markdown("**해설:**")
                st.write(explanation)


# =====================================
# 4. 대화 기록
# =====================================

st.markdown("---")
st.markdown("## 💬 대화 기록")

if "history" in st.session_state:
    for turn in st.session_state["history"]:
        if turn["role"] == "user":
            st.markdown(f"**🙋 사용자:** {turn['content']}")
        else:
            st.markdown(f"**🤖 챗봇:** {turn['content']}")
else:
    st.write("아직 대화가 없어요 🙇")

