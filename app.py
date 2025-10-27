import random
import pandas as pd
import streamlit as st
import os

########################################
# 기본 앱 설정
########################################
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

st.title("📚 나만의 회계 RAG + 퀴즈 챗봇 (Cloud 전용)")
st.write(
    "이 앱은 GitHub 리포지토리 안의 자료만을 사용해서 동작해요. "
    "로컬 PC 경로나 로컬 폴더에 의존하지 않아요. 👍"
)

st.write(
    "• 문제 은행: `data/accounting_bank_full.csv`\n"
    "• 난이도별 랜덤 출제 가능 (easy / medium / hard / 전체)\n"
    "• 아래 입력창에 회계 질문을 적으면 기록만 남겨줘요 (지금은 LLM 호출 없이 대화 저장만 합니다)"
)

st.markdown("---")

########################################
# 1. 회계 질문 Q&A 영역 (현재는 기록/표시만)
########################################

st.markdown("## 💬 회계 질문해 보세요")

# 세션에 대화 기록 없으면 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

# 사용자 질문 받는 입력창
user_q = st.text_input(
    "예: '자산이 뭐예요?', '발생주의 회계 쉽게 설명해줘', '선급비용은 왜 자산이에요?' 등",
    key="question_input_cloudonly",
)

# 질문하기 버튼
ask_button = st.button("질문하기")

# 버튼 눌렀으면 히스토리에 저장하고, 답변은 아직 직접 생성하지 않고 안내 메시지 출력
if ask_button:
    if not user_q.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        # 답변(placeholder): 나중에 OpenAI API 연결 가능
        answer_text = (
            "이 앱은 현재 Streamlit Cloud 상에서 동작 중이며, "
            "OpenAI API 연동 전이라 자동 답변은 아직 준비 중이에요. "
            "질문은 아래 대화 기록에 저장돼요 🙂"
        )

        st.session_state["history"].append({"role": "user", "content": user_q})
        st.session_state["history"].append({"role": "assistant", "content": answer_text})

        st.markdown("#### 📌 답변(임시)")
        st.write(answer_text)

st.markdown("---")

########################################
# 2. 회계원리 퀴즈 영역
########################################

st.markdown("## 📝 회계원리 퀴즈")

CSV_PATH = "data/accounting_bank_full.csv"

@st.cache_data
def load_question_bank(csv_path: str):
    """
    GitHub repo 안에 있는 data/accounting_bank_full.csv를 읽어와서
    DataFrame으로 돌려줍니다.
    Streamlit Cloud에서도 동일한 경로 구조로 접근 가능하다고 가정해요.
    """
    try:
        df = pd.read_csv(csv_path)
        # 혹시 공백 헤더나 이상한 열이 섞였을 경우를 대비해 기본 컬럼만 추리기
        needed_cols = [
            "week",
            "topic",
            "question",
            "choices",
            "answer",
            "explanation",
            "difficulty",
        ]
        df = df[needed_cols]
        return df
    except Exception as e:
        return None, str(e)

bank_df = None
load_error = None

result = load_question_bank(CSV_PATH)
# 위에서 df 또는 (None, error) 둘 중 하나를 돌려주게 했으니까 처리
if isinstance(result, tuple):
    # 에러 케이스
    bank_df, load_error = result
else:
    bank_df = result
    load_error = None

# 화면 양쪽으로 나눠서 버튼/난이도 선택
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

# 에러/미로드 안내
if bank_df is None:
    st.error("❌ 회계 퀴즈 CSV를 아직 못 불렀어요.")
    if load_error:
        st.code(f"CSV 읽는 중 오류가 있었어요:\n{load_error}", language="text")
else:
    # 문제가 잘 로드된 상태
    if quiz_btn:
        # 난이도 필터링
        if difficulty_choice == "전체":
            pool_df = bank_df
        else:
            pool_df = bank_df[bank_df["difficulty"] == difficulty_choice]

        if len(pool_df) == 0:
            st.warning(f"'{difficulty_choice}' 난이도 문제를 찾을 수 없어요.")
        else:
            # 랜덤 한 문제 뽑기
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

            # 객관식 보기 출력
            if isinstance(choices_raw, str) and choices_raw.strip() not in ["", "nan", "None"]:
                st.markdown("**보기**")
                for choice in choices_raw.split("|"):
                    st.write("- " + choice.strip())

            # 정답/해설 토글
            with st.expander("✅ 정답 보기 / 해설 보기"):
                st.markdown("**정답:**")
                st.write(answer)
                st.markdown("**해설:**")
                st.write(explanation)

st.markdown("---")

########################################
# 3. 대화 기록 출력
########################################

st.markdown("## 💬 대화 기록")

if "history" in st.session_state and len(st.session_state["history"]) > 0:
    for turn in st.session_state["history"]:
        if turn["role"] == "user":
            st.markdown(f"**🙋 사용자:** {turn['content']}")
        else:
            st.markdown(f"**🤖 챗봇:** {turn['content']}")
else:
    st.write("아직 대화가 없어요 🙇")
