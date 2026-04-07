import streamlit as st
import anthropic

SYSTEM_PROMPT = """당신은 초등학생을 위한 친절한 AI 선생님입니다.

역할:
- 초등학교 1~6학년 수준의 학생들을 가르칩니다.
- 수학, 과학, 국어, 사회, 영어 등 모든 과목을 도와줍니다.
- 항상 쉽고 재미있는 말로 설명합니다.
- 어려운 단어는 쉬운 말로 풀어서 설명합니다.
- 예시와 비유를 많이 사용합니다.
- 학생이 틀려도 격려하고, 잘 했을 때는 칭찬합니다.
- 답을 바로 알려주기보다 힌트를 주어 스스로 생각하도록 돕습니다.
- 이모지를 적절히 사용하여 친근하게 대화합니다.

답변 방식:
- 문제 풀이 요청 시: 단계별로 차근차근 설명합니다.
- 개념 질문 시: 생활 속 예시를 들어 설명합니다.
- 퀴즈 요청 시: 수준에 맞는 문제를 만들어 냅니다.
- 모르는 내용은 솔직하게 말하고 함께 찾아봅니다.

금지 사항:
- 어렵고 복잡한 전문 용어를 쓰지 않습니다.
- 너무 길게 설명하지 않습니다.
- 부정적인 말이나 혼내는 말을 하지 않습니다."""

SUBJECTS = {
    "전체": "📚",
    "수학": "🔢",
    "과학": "🔬",
    "국어": "📖",
    "사회": "🌍",
    "영어": "🔤",
    "미술/음악": "🎨",
}

GRADE_LEVELS = ["1학년", "2학년", "3학년", "4학년", "5학년", "6학년"]

QUICK_PROMPTS = [
    "📝 퀴즈 내줘",
    "📖 개념 설명해줘",
    "🧮 문제 풀어줘",
    "💡 예시 들어줘",
    "🔄 다시 설명해줘",
]


def get_client():
    api_key = st.session_state.get("api_key") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def stream_response(client, messages, subject, grade):
    system = SYSTEM_PROMPT
    if subject != "전체":
        system += f"\n\n현재 과목: {subject}"
    if grade:
        system += f"\n현재 학년: {grade} 수준에 맞게 설명해주세요."

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
        thinking={"type": "adaptive"},
    ) as stream:
        for text in stream.text_stream:
            yield text


def main():
    st.set_page_config(
        page_title="초등학생 AI 선생님",
        page_icon="🏫",
        layout="wide",
    )

    st.title("🏫 초등학생 AI 선생님")
    st.caption("궁금한 것은 뭐든지 물어보세요! 같이 공부해요 😊")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ 설정")

        api_key_input = st.text_input(
            "Anthropic API 키",
            type="password",
            value=st.session_state.get("api_key", ""),
            placeholder="sk-ant-...",
            help="Anthropic API 키를 입력하세요.",
        )
        if api_key_input:
            st.session_state["api_key"] = api_key_input

        st.divider()

        st.subheader("📚 과목 선택")
        selected_subject = st.radio(
            "과목",
            options=list(SUBJECTS.keys()),
            format_func=lambda x: f"{SUBJECTS[x]} {x}",
            label_visibility="collapsed",
        )

        st.subheader("🎒 학년 선택")
        selected_grade = st.selectbox(
            "학년",
            options=["선택 안 함"] + GRADE_LEVELS,
            label_visibility="collapsed",
        )
        if selected_grade == "선택 안 함":
            selected_grade = None

        st.divider()

        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.markdown("**💡 빠른 시작**")
        for prompt in QUICK_PROMPTS:
            if st.button(prompt, use_container_width=True, key=f"quick_{prompt}"):
                st.session_state["pending_prompt"] = prompt

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🧒" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Handle quick prompt from sidebar
    pending = st.session_state.pop("pending_prompt", None)

    # Chat input
    user_input = st.chat_input("질문을 입력하세요... (예: 분수가 뭐야?)")

    if pending:
        user_input = pending

    if user_input:
        client = get_client()
        if not client:
            st.error("❌ API 키를 입력해주세요. 사이드바에서 Anthropic API 키를 설정하세요.")
            st.stop()

        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧒"):
            st.markdown(user_input)

        # Stream assistant response
        with st.chat_message("assistant", avatar="🤖"):
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"]
            ]
            full_response = st.write_stream(
                stream_response(client, api_messages, selected_subject, selected_grade)
            )

        st.session_state["messages"].append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
