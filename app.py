import os
import tempfile
import streamlit as st

from rag_pipeline import (
    load_docs,
    split_docs,
    build_vectorstore,
    make_qa_chain,
)

# ----------------------------------------
# 기본 세팅 UI
# ----------------------------------------
st.set_page_config(
    page_title="회계원리 RAG 튜터",
    page_icon="💼",
    layout="wide",
)

st.title("💼 회계원리 RAG 튜터")
st.write(
    """
    이 챗봇은 *내가 업로드한 회계 자료* (강의노트, 요약본, 연습문제 등)를 기반으로
    질문에 답해주는 RAG 기반 학습 도우미입니다.  
    - 회계 기초 개념 (자산/부채/자본 등)
    - 분개 / 차변 / 대변
    - 재무제표 구조  
    등을 쉽게 설명해 줘요.
    """
)

# ----------------------------------------
# 사이드바: 자료 업로드 + 벡터스토어 재구성
# ----------------------------------------
st.sidebar.header("📂 자료 업로드")

uploaded_files = st.sidebar.file_uploader(
    "회계 수업 자료 업로드 (PDF 또는 TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="여기에 올린 파일은 임시 폴더에 저장 후, RAG 지식베이스로 활용돼요.",
)

st.sidebar.write("---")
rebuild_clicked = st.sidebar.button("📚 업로드한 자료로 지식베이스 만들기 / 새로고침")


# ----------------------------------------
# 벡터스토어 준비 함수 (캐시 사용)
# ----------------------------------------
@st.cache_resource(show_spinner=True)
def build_pipeline_from_folder(folder_path: str):
    """
    folder_path 안의 파일들로부터
    - 문서 로드
    - 청크 분할
    - 벡터스토어 구성
    - QA 체인 생성
    을 한 번에 수행해서 qa_chain을 리턴
    """
    docs = load_docs(folder_path)          # rag_pipeline.load_docs
    chunks = split_docs(docs)              # rag_pipeline.split_docs
    vectordb = build_vectorstore(chunks)   # rag_pipeline.build_vectorstore
    qa_chain = make_qa_chain(vectordb)     # rag_pipeline.make_qa_chain
    return qa_chain


# ----------------------------------------
# 1) 기본 데이터 폴더(data/)를 우선 지식베이스로 사용
#    2) 업로드하면 업로드 파일까지 포함한 임시폴더 기반으로 다시 빌드
# ----------------------------------------

DEFAULT_DATA_DIR = "data"  # <- 깃허브에 함께 올린 기본 수업자료 폴더라고 가정

# 상태 변수들 (Streamlit 세션 상태)
if "qa_chain" not in st.session_state:
    # 앱 처음 켤 때: 기본 data/ 폴더로 파이프라인 구성 시도
    try:
        st.session_state.qa_chain = build_pipeline_from_folder(DEFAULT_DATA_DIR)
        st.session_state.info_msg = "기본 data/ 자료로 학습 중입니다."
    except Exception as e:
        st.session_state.qa_chain = None
        st.session_state.info_msg = f"기본 data/ 로 파이프라인을 만들지 못했어요: {e}"

if rebuild_clicked:
    # 사용자가 새로 업로드한 파일들로 새로운 벡터스토어 구성
    if uploaded_files and len(uploaded_files) > 0:
        # 임시 폴더 하나 만듦
        tmpdir = tempfile.mkdtemp()

        # 업로드된 파일들을 임시 폴더 안에 저장
        for uf in uploaded_files:
            save_path = os.path.join(tmpdir, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())

        # 그 임시 폴더로 다시 파이프라인 빌드
        try:
            st.session_state.qa_chain = build_pipeline_from_folder(tmpdir)
            st.session_state.info_msg = f"현재 지식베이스는 업로드한 파일({len(uploaded_files)}개) 기준입니다."
            st.sidebar.success("지식베이스가 새로 만들어졌어요!")
        except Exception as e:
            st.sidebar.error(f"업로드한 자료로 파이프라인을 만들지 못했어요: {e}")
    else:
        st.sidebar.warning("먼저 파일을 업로드해 주세요.")


# ----------------------------------------
# 메인 영역 상단 안내
# ----------------------------------------
st.markdown("### 현재 지식베이스 상태")
st.info(st.session_state.get("info_msg", "정보 없음"))


# ----------------------------------------
# Q&A 영역
# ----------------------------------------
st.markdown("## ✍ 질문해 보세요")

question = st.text_input(
    "예: '차변과 대변의 차이를 쉬운 말로 설명해줘', '자산이 뭐야?', '이 거래를 어떻게 분개해?'",
    value="",
)

if st.button("🤖 답변 받기"):
    if not question.strip():
        st.warning("질문을 먼저 입력해주세요.")
    else:
        if st.session_state.qa_chain is None:
            st.error("지식베이스(qa_chain)가 아직 준비되지 않았어요.")
        else:
            with st.spinner("답변 생성 중..."):
                try:
                    answer = st.session_state.qa_chain(question)
                    st.markdown("#### 📘 답변")
                    st.write(answer)
                except Exception as e:
                    st.error(f"답변 중 에러가 발생했어요: {e}")


# ----------------------------------------
# (선택) 회계 퀴즈 섹션
# 교수님이 '추가 구현요소 자유롭게'라고 했으니까
# 점수 + 차별화 요소로서 퀴즈 기능을 넣을 수 있음.
# ----------------------------------------

st.markdown("---")
st.markdown("## 🎯 회계 퀴즈 모드 (추가 기능)")

st.write(
    "아래 버튼을 누르면 회계 기초 개념/분개 연습 문제를 랜덤으로 낼 수 있어요. "
    "이 기능은 수업 복습용입니다."
)

# 이 부분은 rag_pipeline.py에 다음과 같은 함수가 있다고 가정:
# def get_random_quiz():
#     return {"question": "....", "answer": "...."}  # 둘 다 str
try:
    from rag_pipeline import get_random_quiz
    quiz_available = True
except Exception:
    quiz_available = False

if not quiz_available:
    st.caption("퀴즈 기능은 아직 준비 중입니다. (get_random_quiz() 추가하면 활성화돼요!)")
else:
    if st.button("🎲 퀴즈 내줘"):
        qa = get_random_quiz()
        st.session_state.quiz_q = qa["question"]
        st.session_state.quiz_a = qa["answer"]

    if "quiz_q" in st.session_state:
        st.subheader("문제")
        st.write(st.session_state.quiz_q)

        show_answer = st.checkbox("정답 보기")
        if show_answer:
            st.subheader("정답 / 해설")
            st.write(st.session_state.quiz_a)
