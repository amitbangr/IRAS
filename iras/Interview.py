import sys
import os

# Allow standalone execution
# streamlit run iras/Interview.py
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import streamlit as st
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import queue
import time
import json
import plotly.graph_objects as go
import pandas as pd

import asyncio
import edge_tts
from io import BytesIO
import base64
import html

from groq import Groq

from iras.interview_engine import generate_questions, evaluate
from iras.speech_to_text import SpeechToText


if 'stt_engine' not in st.session_state:
    st.session_state.stt_engine = SpeechToText()

stt_engine = st.session_state.stt_engine


# ---------- SESSION DEFAULTS ----------
def init_interview_state():

    defaults = {
        "recording": False,
        "audio_data": [],
        "interview_started": False,
        "questions": [],
        "current_q": 0,
        "results": [],
        "selected_role": "Data Scientist",
        "previous_role": "",
        "chat_messages": [],
        "chat_started": False,
        "mode": "idle",
        "q_index": 0,
        "last_spoken_q": -1,
        "live_text": "",
        "answer": "",
        "improvements": [],
        "strengths": [],
        "result_saved": False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_interview_state()




# ---------- SMART RECORD AUDIO ----------
def record_audio(duration=120, fs=16000):

    silence_threshold = 0.01
    silence_duration = 2.0
    chunk_duration = 0.5
    max_record_seconds = duration

    audio_queue = queue.Queue()
    audio_chunks = []

    speech_detected = False
    silent_time = 0
    start_time = time.time()

    status_box = st.empty()

    status_box.info("Listening... Speak naturally")

    def callback(indata, frames, time_info, status):

        if status:
            print(status)

        audio_queue.put(indata.copy())

    stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=int(chunk_duration * fs)
    )

    try:

        with stream:

            while True:

                if (time.time() - start_time) > max_record_seconds:
                    break

                chunk = audio_queue.get()

                rms = np.sqrt(np.mean(np.square(chunk)))

                if rms > silence_threshold:

                    speech_detected = True
                    silent_time = 0

                    audio_chunks.append(chunk)

                    status_box.success("Speech detected...")

                else:

                    if speech_detected:

                        silent_time += chunk_duration
                        audio_chunks.append(chunk)

                        status_box.info("Waiting for completion...")

                if speech_detected and silent_time >= silence_duration:
                    break

                if not speech_detected and (time.time() - start_time) >= 6:

                    status_box.warning("No speech detected")
                    return None

    except Exception as e:

        st.error(f"Recording failed: {str(e)}")
        return None

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0)

    audio = np.squeeze(audio)

    # Normalize audio
    max_val = np.max(np.abs(audio))

    if max_val > 0:
        audio = audio / max_val

    # Convert float32 → int16
    audio_int16 = np.int16(audio * 32767)

    temp_wav = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    temp_wav.close()

    wav.write(temp_wav.name, fs, audio_int16)

    status_box.success("Recording completed")

    return str(temp_wav.name)


# ---------- BROWSER REALTIME TTS ----------
def speak_live(text):

    safe_text = html.escape(text)
    unique_id = f"iras_tts_{int(time.time() * 1000)}"
    tts_html = f"""
    <div id="{unique_id}"></div>

    <script>

    (function() {{

        const text = `{safe_text}`;

        function stopExistingSpeech() {{
            window.speechSynthesis.cancel();
        }}

        function getBestVoice() {{
            const voices = window.speechSynthesis.getVoices();
            if (!voices || voices.length === 0) {{
                return null;
            }}
            console.log("[IRAS] Available voices:", voices.map(v => v.name));
            // Fixed preferred voice configuration:
            const preferredKeywords = ["guy", "david", "male"];
            for (const keyword of preferredKeywords) {{
                const matchedVoice = voices.find(v =>
                    v.name.toLowerCase().includes(keyword)
                );
                if (matchedVoice) {{
                    return matchedVoice;
                }}
            }}
            return voices[0];
        }}

        function speakNow() {{
            stopExistingSpeech();
            const utterance = new SpeechSynthesisUtterance(text);
            const selectedVoice = getBestVoice();
            if (selectedVoice) {{
                utterance.voice = selectedVoice;
                console.log("[IRAS TTS] Using voice:", selectedVoice.name);
            }}
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            utterance.onstart = () => {{
                console.log("[IRAS TTS] Speaking started");
            }};
            utterance.onend = () => {{
                console.log("[IRAS TTS] Speaking completed");
            }};
            utterance.onerror = (e) => {{
                console.error("[IRAS TTS ERROR]", e);
            }};
            window.speechSynthesis.speak(utterance);
        }}

        if (window.speechSynthesis.getVoices().length === 0) {{
            window.speechSynthesis.onvoiceschanged = () => {{
                setTimeout(speakNow, 300);
            }};
        }} else {{
            setTimeout(speakNow, 300);
        }}

    }})();

    </script>
    """

    st.components.v1.html(
        tts_html,
        height=0
    )


# ---------- CHAT INTERVIEW ----------
def render_chat_interview():
    # Safe session-state initialization
    if "chat_started" not in st.session_state:
        st.session_state.chat_started = False

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.markdown("## Chat with AI Interviewer")
    st.caption("Practice interviews through a conversational AI chat interface.")

    role = st.session_state.get("selected_role", "Data Scientist")

    st.info(f"Target Role: {role}")

    difficulty = st.selectbox(
        "Difficulty",
        ["Beginner", "Intermediate", "Advanced"],
        key="chat_difficulty"
    )

    interview_type = st.selectbox(
        "Interview Type",
        ["HR", "Technical", "Behavioral"],
        key="chat_type"
    )

    if not st.session_state.chat_started:

        if st.button("Start Chat Interview"):

            st.session_state.chat_started = True

            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": f"""
Hello! I’ll conduct a {difficulty.lower()} {interview_type.lower()} interview for the role of {role}.

Let’s begin.

Tell me about yourself.
"""
                }
            ]

            st.rerun()

    if st.session_state.chat_started:

        chat_container = st.container(height=450, border=True)

        with chat_container:

            for msg in st.session_state.chat_messages:

                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_input = st.chat_input("Type your answer...")

        if user_input:

            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            try:

                groq_client = Groq(
                    api_key=os.getenv("GROQ_API_KEY")
                )

                system_prompt = f"""
You are a friendly AI mock interviewer.

Conduct a realistic {interview_type} interview for the role of {role}.

Rules:
1. Keep responses short and natural.
2. Talk like a real interviewer.
3. Avoid textbook-style explanations.
4. Give only brief feedback.
5. Ask one interview question at a time.
6. Make the conversation feel human.
7. Keep responses under 80 words.
8. Do not sound robotic.
9. Ask practical industry-style questions.
10. Difficulty level is {difficulty}.
"""

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        *st.session_state.chat_messages
                    ],
                    temperature=0.5
                )

                ai_reply = response.choices[0].message.content

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": ai_reply
                    }
                )

                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Restart Interview"):

                st.session_state.chat_messages = []
                st.session_state.chat_started = False

                st.rerun()

        with col2:
            if st.button("Clear Chat"):

                st.session_state.chat_messages = []

                st.rerun()


# ---------- MAIN INTERVIEW PAGE ----------
def render_interview():

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0f172a,#111827);
        padding:2rem;
        border-radius:24px;
        margin-bottom:1.5rem;
        border:1px solid rgba(255,255,255,0.08);
    ">
        <h1 style="color:white;margin:0;">AI Interview Engine</h1>
        <p style="color:#94a3b8;margin-top:10px;">
            Practice AI-powered mock interviews with voice and chat interaction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Interview Setup")

    interview_roles = [
        "Data Scientist",
        "AI/ML Engineer",
        "Software Engineer",
        "Full Stack Developer",
        "Frontend Developer",
        "Backend Developer",
        "Cloud Engineer",
        "DevOps Engineer",
        "Cybersecurity Analyst",
        "Data Analyst",
        "Business Analyst",
        "Mobile App Developer",
        "UI/UX Designer",
        "Product Manager"
    ]

    setup_option = st.radio(
        "Choose interview setup method",
        [
            "Select Role Manually",
            "Upload Resume to Detect Role"
        ],
        horizontal=True
    )

    if setup_option == "Select Role Manually":

        selected_role = st.selectbox(
            "Select Your Target Role",
            interview_roles,
            key="main_interview_role"
        )

        # Reset interview only if role changed
        if st.session_state.get("previous_role") != selected_role:

            st.session_state.selected_role = selected_role
            st.session_state.previous_role = selected_role

            st.session_state.questions = []
            st.session_state.results = []
            st.session_state.q_index = 0
            st.session_state.mode = "idle"
            st.session_state.last_spoken_q = -1
            st.session_state.live_text = ""
            st.session_state.answer = ""

    else:

        uploaded_resume = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx"],
            key="interview_resume_upload"
        )

        if uploaded_resume:

            try:

                resume_text = ""

                if uploaded_resume.type == "application/pdf":
                    import fitz
                    pdf = fitz.open(stream=uploaded_resume.read(), filetype="pdf")
                    for page in pdf:
                        resume_text += page.get_text()

                else:
                    from docx import Document
                    doc = Document(uploaded_resume)
                    resume_text = "\n".join([
                        p.text for p in doc.paragraphs
                    ])

                lower_text = resume_text.lower()

                role_keywords = {

                    "Data Scientist": [
                        "machine learning",
                        "data science",
                        "tensorflow",
                        "pandas"
                    ],

                    "AI/ML Engineer": [
                        "deep learning",
                        "llm",
                        "neural network"
                    ],

                    "Software Engineer": [
                        "java",
                        "algorithms",
                        "software engineer"
                    ],

                    "Full Stack Developer": [
                        "react",
                        "node",
                        "mongodb",
                        "full stack"
                    ],

                    "Frontend Developer": [
                        "frontend",
                        "javascript",
                        "react",
                        "html",
                        "css"
                    ],

                    "Backend Developer": [
                        "backend",
                        "django",
                        "flask",
                        "api"
                    ],

                    "Cloud Engineer": [
                        "aws",
                        "azure",
                        "cloud",
                        "gcp"
                    ],

                    "DevOps Engineer": [
                        "docker",
                        "jenkins",
                        "ci/cd"
                    ],

                    "Cybersecurity Analyst": [
                        "cybersecurity",
                        "penetration testing",
                        "network security"
                    ],

                    "Data Analyst": [
                        "excel",
                        "sql",
                        "power bi",
                        "tableau"
                    ],

                    "Business Analyst": [
                        "business analysis",
                        "stakeholder"
                    ],

                    "Mobile App Developer": [
                        "flutter",
                        "android",
                        "ios"
                    ],

                    "UI/UX Designer": [
                        "figma",
                        "wireframe",
                        "prototype"
                    ],

                    "Product Manager": [
                        "product strategy",
                        "roadmap"
                    ]
                }

                best_match = None
                best_score = 0

                for role_name, keywords in role_keywords.items():

                    score = sum(
                        1 for keyword in keywords
                        if keyword in lower_text
                    )

                    if score > best_score:
                        best_score = score
                        best_match = role_name

                if best_match:

                    if st.session_state.get("previous_role") != best_match:

                        st.session_state.selected_role = best_match
                        st.session_state.previous_role = best_match

                        st.session_state.questions = []
                        st.session_state.results = []
                        st.session_state.q_index = 0
                        st.session_state.mode = "idle"
                        st.session_state.last_spoken_q = -1
                        st.session_state.live_text = ""
                        st.session_state.answer = ""

                    st.success(f"Detected Role: {best_match}")

                else:

                    st.warning(
                        "⚠️ Could not detect role confidently."
                    )

            except Exception as e:

                st.error(f"Resume analysis failed: {str(e)}")

    st.markdown("---")

    tab1, tab2 = st.tabs([
        "Voice Interview",
        "Chat Practice"
    ])

    # ---------- VOICE ----------
    with tab1:

        st.markdown("## Interview with AI")
        st.caption("Practice interviews through a conversational AI voice interface.")

        # Difficulty selection for Voice Interview
        difficulty = st.selectbox(
            "Interview Difficulty",
            ["Easy", "Medium", "Hard"],
            index=1,
            key="voice_interview_difficulty"
        )

        role = str(
            st.session_state.get(
                "selected_role",
                "Data Scientist"
            )
        ).strip()

        # Force-sync role from current UI selection
        if setup_option == "Select Role Manually":
            role = str(selected_role).strip()
            st.session_state.selected_role = role

        if st.session_state.mode == "idle":

            st.markdown(f"""
            <div style="
                background:#111827;
                padding:1rem;
                border-radius:16px;
                border:1px solid rgba(255,255,255,0.08);
                margin-bottom:1rem;
            ">
                <h4 style="color:white; margin:0;">Selected Target Role</h4>
                <p style="color:#60a5fa; font-size:1.1rem; margin-top:8px;">
                    {role}
                </p>
                <p style="color:#94a3b8; margin-top:4px;">
                    Difficulty: {difficulty}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Reset interview state if role changed
            if st.session_state.get("previous_role") != role:

                st.session_state.questions = []
                st.session_state.results = []
                st.session_state.q_index = 0
                st.session_state.mode = "idle"
                st.session_state.last_spoken_q = -1
                st.session_state.live_text = ""
                st.session_state.answer = ""
                st.session_state.previous_role = role

            if st.button("Start Interview"):

                try:

                    current_role = str(role).strip()
                    st.session_state.selected_role = current_role

                    print(f"[IRAS DEBUG] CURRENT ROLE => {current_role}")

                    generated_questions = generate_questions(
                        role=current_role,
                        difficulty=difficulty.lower()
                    )

                    if not generated_questions:
                        st.error("No interview questions generated.")
                        return

                    if isinstance(generated_questions, str):
                        generated_questions = [generated_questions]

                    st.session_state.questions = generated_questions
                    st.session_state.q_index = 0
                    st.session_state.results = []
                    st.session_state.mode = "ask"
                    st.session_state.last_spoken_q = -1
                    st.session_state.live_text = ""
                    st.session_state.answer = ""
                    st.session_state.result_saved = False

                    print(f"[IRAS] Generated questions for role: {current_role}")
                    print(f"[IRAS] Questions count: {len(generated_questions)}")

                    st.success(
                        f"Generated {len(generated_questions)} questions for {current_role}"
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"Question generation failed: {str(e)}")

        # Safety check
        if not st.session_state.questions:

            st.markdown("""
            <div style='
                background:#0f172a;
                padding:1rem;
                border-radius:14px;
                border:1px solid rgba(255,255,255,0.06);
                color:#94a3b8;
                margin-top:1rem;
            '>
                Click <b>Start Interview</b> to begin the voice interview.
                <br><br>
                Or switch to the <b>Chat Practice</b> tab for conversational practice.
            </div>
            """, unsafe_allow_html=True)

        else:
            if st.session_state.q_index >= len(st.session_state.questions):
                st.session_state.q_index = len(st.session_state.questions) - 1

            q = st.session_state.questions[
                st.session_state.q_index
            ]

            st.markdown(f"### Question {st.session_state.q_index + 1}")

            st.caption(f"Interview Role: {role}")
            st.info(q)


            if st.session_state.last_spoken_q != st.session_state.q_index:

                with st.spinner("AI interviewer speaking..."):
                    speak_live(q)

                st.session_state.last_spoken_q = st.session_state.q_index
                st.session_state.mode = "listen"

            if st.session_state.mode == "listen":

                st.info("Tip: Keep your answer between 1–2 minutes")

                if st.button("Record Answer"):

                    with st.spinner("Smart voice capture active..."):
                        audio_path = record_audio(duration=120)

                    try:
                        text = stt_engine.transcribe(audio_path)

                    except Exception as e:
                        st.error(f"STT Error: {str(e)}")
                        return

                    if text and len(text.strip()) > 2:

                        word_count = len(text.split())

                        if word_count < 20:
                            st.warning("Answer too short. Try to elaborate more.")

                        elif word_count > 400:
                            st.warning("Answer too long. Try to be concise.")

                        st.session_state.live_text = text
                        st.session_state.answer = text
                        st.session_state.result_saved = False
                        st.session_state.mode = "process"

                        st.rerun()

                    else:
                        st.warning("Speech unclear or too short. Try speaking closer to the microphone.")

            st.markdown("### You said:")
            st.success(st.session_state.get("live_text", "Waiting..."))

            if st.session_state.mode == "process":

                st.info("Evaluating...")

                result = evaluate(q, st.session_state.answer)

                score = 5
                feedback = result
                parsed = {}

                try:
                    if isinstance(result, dict):
                        parsed = result
                    else:
                        import re

                        cleaned_result = str(result).strip()

                        # Handle ```json ... ``` responses
                        cleaned_result = re.sub(
                            r'^```json\s*',
                            '',
                            cleaned_result,
                            flags=re.IGNORECASE
                        )
                        cleaned_result = re.sub(
                            r'^```',
                            '',
                            cleaned_result
                        )
                        cleaned_result = re.sub(
                            r'```$',
                            '',
                            cleaned_result
                        )

                        # Extract JSON object even if extra text exists
                        json_match = re.search(
                            r'\{.*\}',
                            cleaned_result,
                            re.DOTALL
                        )

                        if json_match:
                            parsed = json.loads(json_match.group(0))
                        else:
                            raise ValueError(
                                f'No valid JSON found in LLM response: {cleaned_result[:300]}'
                            )

                    raw_score = parsed.get("overall_score", 5)

                    try:
                        score = float(raw_score)
                    except Exception:
                        import re
                        match = re.search(r"\d+(?:\.\d+)?", str(raw_score))
                        score = float(match.group()) if match else 5.0
                    feedback = parsed

                except Exception as e:

                    print(f"Evaluation parse error: {e}")

                    import re

                    feedback = str(result)

                    score_match = re.search(r"score\s*:?\s*(\d+)", feedback.lower())

                    if score_match:
                        score = int(score_match.group(1))
                    else:
                        score = 5

                improvements = []
                strengths = []

                answer_text = st.session_state.answer.lower()

                if len(answer_text.split()) < 25:
                    improvements.append(
                        "Answer is too short — expand with explanation and examples"
                    )
                else:
                    strengths.append(
                        "Good answer length and effort"
                    )

                if "example" not in answer_text:
                    improvements.append(
                        "Add real-world examples to strengthen your answer"
                    )
                else:
                    strengths.append(
                        "Used examples effectively"
                    )

                if "i" not in answer_text:
                    improvements.append(
                        "Make answers more personal using real ownership"
                    )
                else:
                    strengths.append(
                        "Good ownership in response"
                    )

                if not st.session_state.get("result_saved", False):

                    current_question = st.session_state.q_index

                    while len(st.session_state.results) <= current_question:
                        st.session_state.results.append(None)

                    st.session_state.results[current_question] = {
                        "question": q,
                        "answer": st.session_state.answer,
                        "overall_score": float(score),
                        "relevance": float(parsed.get("relevance", score)) if parsed else float(score),
                        "technical_accuracy": float(parsed.get("technical_accuracy", score)) if parsed else float(score),
                        "depth": float(parsed.get("depth", score)) if parsed else float(score),
                        "communication": float(parsed.get("communication", score)) if parsed else float(score),
                        "practical_application": float(parsed.get("practical_application", score)) if parsed else float(score),
                        "feedback": feedback,
                        "improvements": improvements,
                        "strengths": strengths
                    }

                    st.session_state.result_saved = True

                # If this is the final question, skip showing per-question feedback
                # and go directly to the Executive Interview Dashboard.
                if st.session_state.q_index == len(st.session_state.questions) - 1:
                    st.session_state.q_index += 1

                # Only show Feedback, Strengths, and Improvements for non-final questions
                if st.session_state.q_index < len(st.session_state.questions) - 1:
                    st.write("### Feedback")

                    if parsed:

                        if not parsed.get("evaluable", True):
                            st.error(
                                parsed.get(
                                    "reason_if_not_evaluable",
                                    "Answer could not be evaluated."
                                )
                            )

                        if parsed.get("strengths"):
                            for item in parsed.get("strengths", []):
                                st.success(item)

                        if parsed.get("weaknesses"):
                            for item in parsed.get("weaknesses", []):
                                st.warning(item)

                        if parsed.get("suggested_improvement"):
                            st.info(parsed.get("suggested_improvement"))

                    else:
                        st.write(str(feedback))

                    if strengths:
                        st.write("### Strengths")

                        for s in strengths:
                            st.success(s)

                    if improvements:
                        st.write("### Improvements")

                        for i in improvements:
                            st.warning(i)

                    st.session_state.q_index += 1

                if st.session_state.q_index >= len(st.session_state.questions):

                    # Keep only scores for the current interview
                    max_questions = len(st.session_state.questions)
                    if len(st.session_state.results) > max_questions:
                        st.session_state.results = st.session_state.results[:max_questions]
                    st.success("Interview Completed")

                    import re

                    valid_results = [r for r in st.session_state.results if r is not None]

                    question_scores = []

                    for r in valid_results:
                        try:
                            question_scores.append(float(r.get("overall_score", 0)))
                        except Exception:
                            pass

                    avg_score = (
                        sum(question_scores) / len(question_scores)
                    ) if question_scores else 0

                    technical_score = np.mean([
                        r.get("technical_accuracy", 0)
                        for r in valid_results
                    ]) if valid_results else 0

                    communication_score = np.mean([
                        r.get("communication", 0)
                        for r in valid_results
                    ]) if valid_results else 0

                    problem_solving_score = np.mean([
                        r.get("practical_application", 0)
                        for r in valid_results
                    ]) if valid_results else 0

                    confidence_score = np.mean([
                        r.get("depth", 0)
                        for r in valid_results
                    ]) if valid_results else 0

                    hiring_probability = int(avg_score * 10)

                    # 1. Hero Card
                    st.markdown(
                        f"""
<div style="
    background: linear-gradient(135deg,#111827 0%,#374151 100%);
    border-radius: 32px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom:2.2rem;
    box-shadow:0 6px 36px 0 rgba(0,0,0,0.18);
    border:1.5px solid rgba(255,255,255,0.08);
    display:flex;
    flex-direction:column;
    align-items:flex-start;
">
    <div style="font-size:2.5rem;font-weight:800;color:#fff;letter-spacing:-1px;margin-bottom:0.6rem;">
        Executive Interview Report
    </div>
    <div style="font-size:1.25rem;color:#94a3b8;margin-bottom:1.3rem;">
        Target Role: <span style="color:#60a5fa;font-weight:600;">{role}</span>
    </div>
    <div style="display:flex;gap:2.5rem;align-items:center;">
        <div style="font-size:1.7rem;font-weight:700;color:#fff;">
            Overall Score: <span style="color:#fbbf24;">{avg_score:.1f}/10</span>
        </div>
        <div style="font-size:1.7rem;font-weight:700;color:#fff;">
            Hiring Probability: <span style="color:#34d399;">{hiring_probability}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

                    # 2. Metrics Row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Overall", f"{avg_score:.1f}/10")
                    c2.metric("Technical", f"{technical_score:.1f}/10")
                    c3.metric("Communication", f"{communication_score:.1f}/10")
                    c4.metric("Hiring %", f"{hiring_probability}%")

                    # 3. Hiring Recommendation
                    if hiring_probability >= 80:
                        st.success("Hiring Recommendation: Strong Hire")
                    elif hiring_probability >= 60:
                        st.warning("Hiring Recommendation: Consider Hire")
                    else:
                        st.error("Hiring Recommendation: Needs Improvement")

                    # 4. Top Strengths / Growth Areas
                    all_improvements = []
                    all_strengths = []
                    for r in st.session_state.results:
                        if not r:
                            continue
                        all_improvements.extend(r.get("improvements", []))
                        all_strengths.extend(r.get("strengths", []))
                    s_col, w_col = st.columns(2)
                    with s_col:
                        st.markdown("### Top Strengths")
                        for item in list(set(all_strengths))[:5]:
                            st.success(item)
                    with w_col:
                        st.markdown("### Growth Areas")
                        for item in list(set(all_improvements))[:5]:
                            st.warning(item)

                    # 5. Skill Assessment Radar
                    st.markdown("### Skill Assessment")
                    left, right = st.columns(2)
                    with left:
                        radar = go.Figure()
                        radar.add_trace(
                            go.Scatterpolar(
                                r=[
                                    technical_score,
                                    communication_score,
                                    problem_solving_score,
                                    confidence_score,
                                    avg_score
                                ],
                                theta=[
                                    "Technical",
                                    "Communication",
                                    "Problem Solving",
                                    "Confidence",
                                    "Overall"
                                ],
                                fill="toself"
                            )
                        )
                        radar.update_layout(
                            title="Skill Assessment Radar",
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 10]
                                )
                            ),
                            height=520,
                            showlegend=False
                        )
                        st.plotly_chart(radar, use_container_width=True)
                    # 6. Competency Breakdown
                    with right:
                        st.markdown("### Competency Breakdown")
                        score_bar = go.Figure(
                            go.Bar(
                                x=["Technical", "Communication", "Problem Solving", "Confidence"],
                                y=[
                                    technical_score,
                                    communication_score,
                                    problem_solving_score,
                                    confidence_score
                                ]
                            )
                        )
                        score_bar.update_layout(
                            title="Competency Breakdown",
                            yaxis=dict(range=[0, 10]),
                            height=520
                        )
                        st.plotly_chart(score_bar, use_container_width=True)

                    # 7. Question-by-Question Performance
                    max_questions = len(st.session_state.questions)
                    question_scores = []
                    for r in st.session_state.results[:max_questions]:
                        if not r:
                            continue
                        try:
                            question_scores.append(
                                float(r.get("overall_score", 0))
                            )
                        except Exception:
                            pass
                    st.markdown("### Question-by-Question Performance")
                    performance_fig = go.Figure(
                        go.Bar(
                            x=question_scores,
                            y=[f"Q{i}" for i in range(1, min(len(question_scores), max_questions) + 1)],
                            orientation="h",
                            text=[f"{float(s):.1f}" for s in question_scores],
                            textposition="outside"
                        )
                    )
                    performance_fig.update_layout(
                        title="Interview Question Scores",
                        xaxis_title="Score",
                        yaxis_title="Question",
                        xaxis=dict(range=[0, 10]),
                        height=500,
                        template="plotly_dark"
                    )
                    st.plotly_chart(performance_fig, use_container_width=True)

                    # 8. Performance Trend Across Questions
                    if question_scores:
                        st.markdown("### Performance Trend")
                        trend_fig = go.Figure()
                        trend_fig.add_trace(
                            go.Scatter(
                                x=list(range(1, len(question_scores) + 1)),
                                y=question_scores,
                                mode="lines+markers"
                            )
                        )
                        trend_fig.update_layout(
                            title="Performance Trend Across Questions",
                            xaxis_title="Question",
                            yaxis_title="Score",
                            height=450
                        )
                        st.plotly_chart(trend_fig, use_container_width=True)

                    # 9. AI Career Coach
                    st.markdown("### AI Career Coach")
                    try:
                        groq_client = Groq(
                            api_key=os.getenv("GROQ_API_KEY")
                        )
                        import json as _json
                        coach_prompt = (
                            "You are an expert AI interview coach. "
                            "Given the following JSON list of interview results, provide up to 5 bullet points (maximum 120 words) listing only actual mistakes or weaknesses from the candidate's answers. "
                            "Reference what the candidate said and suggest a better response if possible. "
                            "No headings, no emojis, no motivation, no career advice. "
                            "Do not repeat strengths. Only mention specific mistakes, poor examples, missing details, or unclear explanations. "
                            "Format as a plain bullet list."
                            "\n\nInterview Results:\n"
                            + _json.dumps(st.session_state.results, indent=2)
                        )
                        coach_response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {
                                    "role": "system",
                                    "content": coach_prompt
                                }
                            ],
                            temperature=0.5,
                            max_tokens=400
                        )
                        st.info(coach_response.choices[0].message.content)
                    except Exception as e:
                        st.info(
                            "Use STAR method, include project examples, quantify impact, explain technical decisions, and show problem-solving thinking."
                        )

                    # 10. Restart Interview
                    if st.button("Restart Full Interview"):
                        current_role = st.session_state.get(
                            "selected_role",
                            "Data Scientist"
                        )
                        st.session_state.mode = "idle"
                        st.session_state.questions = []
                        st.session_state.results = []
                        st.session_state.q_index = 0
                        st.session_state.last_spoken_q = -1
                        st.session_state.live_text = ""
                        st.session_state.answer = ""
                        st.session_state.previous_role = current_role
                        st.rerun()

                    return

                st.session_state.mode = "ask"
                st.rerun()

    # ---------- CHAT ----------
    with tab2:
        render_chat_interview()

# ---------- STANDALONE RUN ----------
if __name__ == "__main__":

    st.set_page_config(
        page_title="AI Interview Engine",
        page_icon="🎤",
        layout="wide"
    )

    render_interview()
    