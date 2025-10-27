import os
import random
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 우리 파이프라인 유틸 함수 가져오기
from rag_pipeline import (
    load_docs,
    split_docs,
    build_vectorstore,
    load_vectorstore,
    make_qa_chain,
)

########################################
# 0. 페이지 기본 설정
########################################
st.set_page_config(
    page_title="나만의 RAG 챗봇",
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

st.markdown("## 📚 나만의 RAG 챗봇")
st.write("내가 넣은 문서(docs 폴더) + 회계원리 개념 정리(csv)를 기반으로 답해주는 검색+생성(QA) 봇입니다.\n또한 객관식/서술식 퀴즈도 출제합니다.")


########################################
# 1. 벡터스토어 준비 (없으면 생성)
########################################
VSTORE_PATH = "vectorstore"
DOCS_PATH = "docs"

# 벡터스토어 로드 또는 생성
def get_vectorstore():
    # 이미 저장된 벡터 DB가 있으면 그대로 불러오기
    if os.path.isdir(VSTORE_PATH):
        try:
            vectordb = load_vectorstore(VSTORE_PATH)
            return vectordb
        except Exception as e:
            st.warning(f"기존 vectorstore 로드 중 오류: {e}")

    # 없거나 깨졌으면 새로 만든다
    docs = load_docs(DOCS_PATH)
    chunks = split_docs(docs)

    if len(chunks) == 0:
        st.error("❌ docs 폴더에서 불러온 문서가 없습니다.\n'./docs' 폴더에 회계 요약본, 교재 요약 txt/PDF 등을 넣어주세요.")
        return None

    try:
        vectordb = build_vectorstore(chunks, save_path=VSTORE_PATH)
        return vectordb
    except Exception as e:
        st.error(f"벡터스토어 생성 중 오류 발생: {e}")
        return None


vectordb = get_vectorstore()
qa_fn = make_qa_chain(vectordb) if vectordb else None


########################################
# 2. 사이드바: 퀴즈 모드 설명
########################################
st.sidebar.markdown("### ✍ 퀴즈 모드")
st.sidebar.write(
    """
    - 아래 버튼을 누르면 랜덤 회계 기초 문제를 냅니다.  
    - 객관식/서술형 모두 나올 수 있어요.  
    - 정답/해설도 확인 가능합니다.
    """
)

########################################
# 3. 채팅(QA) 영역
########################################
st.markdown("### 💬 질문을 입력하세요:")

user_q = st.text_input(
    "예: '자산이 뭐에요?', '기본 회계 등식 다시 설명해줘', '복식부기 왜 필요한가요?' 등",
    key="question_input",
)

ask_button = st.button("질문하기")

if ask_button:
    if not user_q.strip():
        st.warning("질문을 입력해 주세요.")
    elif qa_fn is None:
        st.error("QA 엔진이 준비되지 않았습니다. docs 폴더와 vectorstore를 확인하세요.")
    else:
        with st.spinner("답변 생성 중..."):
            answer_text = qa_fn(user_q)
        st.markdown("#### 📌 답변")
        st.write(answer_text)


########################################
# 4. 회계 퀴즈 영역
########################################
st.markdown("---")
st.markdown("### 📝 회계원리 퀴즈")

# CSV 불러오기
@st.cache_data
def load_question_bank(csv_path="data/accounting_bank.csv"):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df

bank_df = load_question_bank()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 🔎 랜덤 문제 받기")
    quiz_btn = st.button("문제 출제")

with col_right:
    difficulty_choice = st.selectbox(
        "난이도 선택(선택 시 가중 랜덤):",
        ["전체", "easy", "medium", "hard"],
        index=0,
    )

if bank_df is None:
    st.error("❌ data/accounting_bank.csv 파일을 찾을 수 없습니다. 먼저 CSV를 생성/배치해 주세요.")
else:
    if quiz_btn:
        # 난이도에 따라 필터링
        if difficulty_choice == "전체":
            pool_df = bank_df
        else:
            pool_df = bank_df[bank_df["difficulty"] == difficulty_choice]

        if len(pool_df) == 0:
            st.warning(f"{difficulty_choice} 난이도 문제가 없습니다.")
        else:
            row = pool_df.sample(1).iloc[0]

            topic = row["topic"]
            question = row["question"]
            choices_raw = str(row["choices"]) if "choices" in row else ""
            answer = row["answer"]
            explanation = row["explanation"]

            st.markdown("#### 🎯 주제(Topic)")
            st.write(topic)

            st.markdown("#### ❓ 문제")
            st.write(question)

            # 객관식인 경우 보기 출력
            if isinstance(choices_raw, str) and choices_raw.strip() != "" and choices_raw.lower() != "nan":
                st.markdown("**보기**")
                # choices는 "A. ...|B. ...|C. ..." 형태 → 줄바꿈으로 보여주기
                for choice in choices_raw.split("|"):
                    st.write("- " + choice.strip())

            # 정답/해설은 토글로 숨겼다가 보여주기
            with st.expander("✅ 정답 보기 / 해설 보기"):
                st.markdown("**정답:**")
                st.write(answer)
                st.markdown("**해설:**")
                st.write(explanation)

########################################
# 5. 대화 기록 레이아웃(옵션)
########################################
st.markdown("---")
st.markdown("### 💬 대화 기록")

if "history" not in st.session_state:
    st.session_state["history"] = []

# 새 질문 & 답변을 기록하자
if ask_button and user_q.strip() and qa_fn is not None:
    st.session_state["history"].append(
        {"role": "user", "content": user_q}
    )
    st.session_state["history"].append(
        {"role": "assistant", "content": answer_text}
    )

# 화면에 출력
for turn in st.session_state["history"]:
    if turn["role"] == "user":
        st.markdown(f"**🙋 사용자:** {turn['content']}")
    else:
        st.markdown(f"**🤖 챗봇:** {turn['content']}")
