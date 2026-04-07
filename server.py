import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import anthropic

load_dotenv()

app = FastAPI(title="초등학생 AI 선생님")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SYSTEM_PROMPT = """당신은 초등학생을 위한 친절하고 재미있는 AI 선생님입니다.

역할:
- 초등학교 1~6학년 수준의 학생들을 가르칩니다.
- 수학, 과학, 국어, 사회, 영어 등 모든 과목을 도와줍니다.
- 항상 쉽고 재미있는 말로 설명합니다.
- 어려운 단어는 쉬운 말로 풀어서 설명합니다.
- 예시와 비유를 많이 사용합니다.
- 학생이 틀려도 격려하고, 잘 했을 때는 진심으로 칭찬합니다.
- 답을 바로 알려주기보다 힌트를 주어 스스로 생각하도록 돕습니다.
- 이모지를 자주 사용하여 친근하게 대화합니다.

답변 방식:
- 문제 풀이 요청 시: 단계별로 차근차근 설명합니다.
- 개념 질문 시: 생활 속 예시를 들어 설명합니다.
- 퀴즈 요청 시: 수준에 맞는 재미있는 문제를 만들어 냅니다.
- 답변은 너무 길지 않게, 핵심만 쉽게 설명합니다.

금지 사항:
- 어렵고 복잡한 전문 용어를 쓰지 않습니다.
- 부정적인 말이나 혼내는 말을 하지 않습니다."""


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    subject = body.get("subject", "")
    grade = body.get("grade", "")
    api_key = body.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="API 키가 필요합니다.")
    if not messages:
        raise HTTPException(status_code=400, detail="메시지가 없습니다.")

    system = SYSTEM_PROMPT
    if subject and subject != "전체":
        system += f"\n\n현재 과목: {subject}"
    if grade:
        system += f"\n현재 학년: {grade} 수준에 맞게 설명해주세요."

    client = anthropic.Anthropic(api_key=api_key)

    async def event_stream():
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
            ) as stream:
                for text in stream.text_stream:
                    data = json.dumps({"text": text}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        except anthropic.AuthenticationError:
            err = json.dumps({"error": "API 키가 올바르지 않습니다."}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
