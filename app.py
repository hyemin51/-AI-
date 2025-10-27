import os
import random
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

#####################################################
# 0. 환경 변수 불러오기 (.env에서 OPENAI_API_KEY 등)
#####################################################
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#####################################################
# 0.1 CSV 절대 경로 설정 (네 PC 경로에 맞게!)
#
# ❗❗ 이 경로는 꼭 실제 경로로 맞춰줘야 해요.
# 아래 값은 지금까지 상황 기준으로 넣은 기본값이에요.
# 만약 다르면 바꿔주세요.
#####################################################
CSV_ABS_PATH = r"C:\Users\82105\OneDrive\바탕 화면\AI\data\accounting_bank_full.csv"

#####################################################
# 0.2 RAG 파이프라인 유틸 불러오기
#####################################################
from rag_pipeline import (
    load_docs,
    split_docs,
    build_vectorstore,
    load_vectorstore,
    make_qa_chain,
)

#####################################################
# 1. Streamlit 페이지 기본 설정
#####################################################
st.set_page_config(
    page_title="나만의 회계 튜터",
    page_icon="📚",
    layout="wide",
)

# 약간의 CSS로 입력 폰트 크기 키우기
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

#####################################################
# 2. 상단 헤더
#####################################################
st.title("📚 나만의 회계 RAG + 퀴즈 챗봇")
st.write("내가 넣은 강의자료(docs 폴더)와 문제은행(data/accounting_bank_full.csv)을 바탕으로 공부 도와주는 개인 튜터예요.")

# 현재 작업 디렉터리(디버그용) - 화면에 보여주면 경로 확인 가능
st.caption(f"현재 작업 디렉터리: {os.getcwd()}")
st.caption(f"퀴즈 CSV를 여기서 찾으려고 해요: {CSV_ABS_PATH}")

#####################################################
# 3. 세션 상태 초기화 (메모리처럼 계속 유지할 값들)
#####################################################
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None
if "qa_fn" not in st.session_state:
    st.session_state["qa_fn"] = None
if "build_error" not in st.session_state:
    st.session_state["build_error"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []  # Q/A 기록 담는 곳


#####################################################
# 4. 벡터스토어 / RAG 관련 함수 정의
#####################################################

VSTORE_PATH = "vectorstore"   # 벡터DB 저장(폴더)
DOCS_PATH = "docs"            # 수업자료 pdf/txt 넣는 폴더

def try_load_vectorstore():
    """
    이미 생성된 FAISS vectorstore를 로드 시도.
    실패하면 None 반환.
    """
    try:
        vectordb = load_vectorstore(VSTORE_PATH)
        return vectordb
    except Exception:
        return None


def rebuild_vectorstore():
    """
    docs 폴더의 pdf/txt → 문서 로드 → 청크 분할 → 임베딩 → FAISS 저장
    이걸 다시 실행해서 새로운 vectorstore를 만든다.
    에러가 나면 build_error에 저장.
    """
    try:
        docs = load_docs(DOCS_PATH)
        chunks = split_docs(docs)

        if len(chunks) == 0:
            raise ValueError(
                "docs 폴더에서 불러온 문서가 0개예요.\n"
                "docs 폴더 안에 회계 강의 pdf/txt 파일을 넣어주세요."
            )

        vectordb = build_vectorstore(chunks, save_path=VSTORE_PATH)

        # 세션에 저장해서 바로 질문 가능하게
        st.session_state["vectordb"] = vectordb
        st.session_state["qa_fn"] = make_qa_chain(vectordb)
        st.session_state["build_error"] = None

        st.success("✅ 벡터스토어 생성 완료!")

    except Exception as e:
        st.session_state["build_error"] = str(e)
        st.session_state["vectordb"] = None
        st.session_state["qa_fn"] = None
        st.error(f"벡터스토어 생성 실패: {e}")


# 앱 첫 로드시: vectorstore 자동 로드 시도
if st.session_state["vectordb"] is None:
    maybe_vs = try_load_vectorstore()
    if maybe_vs is not None:
        st.session_state["vectordb"] = maybe_vs
        st.session_state["qa_fn"] = make_qa_chain(maybe_vs)


#####################################################
# 5. 사이드바 (벡터DB 컨트롤)
#####################################################
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


#####################################################
# 6. 질의응답 영역 (RAG 질문 → 답변)
#####################################################
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

        # 답변 출력
        st.markdown("#### 📌 답변")
        st.write(answer_text)

        # 기록 저장
        st.session_state["history"].append({"role": "user", "content": user_q})
        st.session_state["history"].append({"role": "assistant", "content": answer_text})


#####################################################
# 7. 회계 퀴즈 영역
#####################################################
st.markdown("---")
st.markdown("## 📝 회계원리 퀴즈")

@st.cache_data
def load_question_bank():
    """
    우리가 만든 문제은행 CSV를 읽어온다.
    CSV_ABS_PATH에서 직접 읽기 때문에
    현재 작업 디렉터리가 어디든 상관없이 동작한다.
    """
    csv_path = CSV_ABS_PATH

    # 디버그: 실제 경로를 보여줌
    # (이건 사용자 화면에도 떠서 어디를 보고 있는지 알 수 있게 도와줌)
    st.caption(f"[DEBUG] 퀴즈 CSV 경로: {csv_path}")

    if not os.path.exists(csv_path):
        # 파일이 없으면 None
        return None

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"CSV 읽는 중 오류가 났어요: {e}")
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
    st.error(
        "❌ 회계 퀴즈 CSV를 아직 못 불렀어요.\n"
        "CSV 경로가 맞는지 확인해 주세요."
    )
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

            st.markdown(f"**📚 주차:** {week}주차 / **주제:** {topic}")
            st.markdown("**❓ 문제**")
            st.write(question)

            # 객관식 보기 있는 경우만 출력
            if isinstance(choices_raw, str) and choices_raw.strip() != "" and choices_raw.lower() != "nan":
                st.markdown("**보기**")
                for choice in choices_raw.split("|"):
                    st.write("- " + choice.strip())

            with st.expander("✅ 정답 보기 / 해설 보기"):
                st.markdown("**정답:**")
                st.write(answer)
                st.markdown("**해설:**")
                st.write(explanation)


#####################################################
# 8. 대화 기록 영역
#####################################################
st.markdown("---")
st.markdown("## 💬 대화 기록")

if len(st.session_state["history"]) == 0:
    st.write("아직 대화가 없어요 🙇")
else:
    for turn in st.session_state["history"]:
        if turn["role"] == "user":
            st.markdown(f"**🙋 사용자:** {turn['content']}")
        else:
            st.markdown(f"**🤖 챗봇:** {turn['content']}")
