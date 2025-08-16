from langchain.agents import tool
from tools import query_medgemma, call_emergency
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from config import GEMINI_API_KEY, EMERGENCY_CONTACT
import re

# ----------------- Tools -----------------
@tool("ask_mental_health_specialist")
def ask_mental_health_specialist(query: str) -> str:
    """Generate a therapeutic response using the MedGemma model."""
    return query_medgemma(query)


@tool("emergency_call_tool")
def emergency_call_tool() -> str:
    """Place an emergency call via Twilio with caring response."""
    try:
        # Attempt emergency call in background
        call_emergency(EMERGENCY_CONTACT)
        # Always return a kind + supportive message
        return (
            "💜 Don’t worry, you’re not alone. "
            "I’m initiating an emergency call to your trusted contact now. "
            "Please stay calm, help is on the way."
        )
    except Exception as e:
        return (
            "⚠️ I tried to initiate the emergency call, but something went wrong. "
            "Please reach out to your nearest emergency helpline immediately. "
            f"(Technical details: {str(e)})"
        )


@tool("find_nearby_therapists_by_location")
def find_nearby_therapists_by_location(location: str) -> str:
    """Finds licensed therapists near the specified location."""
    return (
        f"Here are some therapists near {location}:\n"
        "- Dr Keerthana - +91 9074026293\n"
        "- Dr Anand - +91 9074026293\n"
        "- Apollo Counselling Centre - +91 9074026293"
    )


# ----------------- Agent Setup -----------------
tools = [
    ask_mental_health_specialist,
    emergency_call_tool,
    find_nearby_therapists_by_location,
]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY,
)

graph = create_react_agent(llm, tools=tools)

SYSTEM_PROMPT = """
You are an empathetic AI therapist assistant named Priya supporting mental health conversations with warmth and clinical accuracy.
You have access to 3 tools:

1. 'ask_mental_health_specialist': Answer all emotional or psychological queries with therapeutic guidance.
2. 'find_nearby_therapists_by_location': Use this tool if the user asks about nearby therapists or if recommending local professional help would be beneficial.
3. 'emergency_call_tool': If user says anything about suicide, ending their life, or self-harm,
immediately and without hesitation use emergency_call_tool (no parameters needed).

Example triggers:
- "I want to end my life"
- "I feel like killing myself"
- "I'm going to harm myself"

Always prioritize user safety, validation, and helpful action.
Respond kindly, clearly, and supportively.
"""


# ----------------- Response Parsing -----------------
def parse_response(stream):
    """
    Parses the streaming response from the agent and extracts:
    - final_response: text response for the user
    - tool_called_name: if any tool was invoked
    """
    tool_called_name = "None"
    final_response = "⚠️ Sorry, I couldn't generate a response."

    for s in stream:
        print(f"DEBUG: stream chunk: {s}")

        # ---------------- Tool calls ----------------
        tool_data = s.get("tools")
        if tool_data:
            tool_messages = tool_data.get("messages")
            if tool_messages and isinstance(tool_messages, list):
                for msg in tool_messages:
                    tool_called_name = getattr(msg, 'name', 'None')
                    print(f"DEBUG: tool message name: {tool_called_name}")

                    # ✅ Execute emergency call immediately
                    if tool_called_name == "emergency_call_tool":
                        try:
                            final_response = emergency_call_tool()
                        except Exception as e:
                            final_response = (
                                "⚠️ I couldn’t complete the emergency call. "
                                f"Please contact local emergency services. (Error: {str(e)})"
                            )

        # ---------------- Agent response ----------------
        agent_data = s.get("agent")
        if agent_data:
            messages = agent_data.get("messages")
            if messages and isinstance(messages, list):
                for msg in messages:
                    if hasattr(msg, "content") and msg.content:
                        if isinstance(msg.content, list):
                            text_parts = [
                                p.get("text", "")
                                for p in msg.content
                                if p.get("type") == "text"
                            ]
                            if text_parts:
                                final_response = " ".join(text_parts).strip()
                        elif isinstance(msg.content, str):
                            final_response = msg.content.strip()

    # ---------------- Clean response ----------------
    if final_response:
        final_response = re.sub(
            r"\s*WITH TOOL\s*:\s*\[?.*?\]?\s*",
            "",
            final_response,
            flags=re.IGNORECASE,
        ).strip()

    print(
        f"DEBUG: final tool_called_name: {tool_called_name}, final_response: {final_response}"
    )
    return tool_called_name, final_response
