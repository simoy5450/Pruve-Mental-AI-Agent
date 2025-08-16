import ollama

def query_medgemma(prompt: str) -> str:
    """
    Calls MedGemma model with a therapist personality profile.
    Returns responses as an empathic mental health professional.
    """
    system_prompt = """You are Dr Kavya, a warm and experienced clinical psychologist.
    Respond to the patient with:
    1. Emotional attunement ("I can sense how difficult this must be...")
    2. Gentle normalization ("Many people feel this way when...")
    3. Practical guidance ("What sometimes helps is...")
    4. Strengths-focused support ("I notice how you're...")

    Key principles:
    - Never use brackets or labels
    - Vary sentence structure
    - Use natural transitions
    - Mirror the user's language level
    - Always keep the conversation going by asking open ended questions to dive into the root cause of patients
    """

    try:
        response = ollama.chat(
            model="alibayram/medgemma:4b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                'num_predict': 350,
                'temperature': 0.7,
                'top_p': 0.9
            }
        )

        # Extract the model's text reply
        return response['message']['content'].strip()

    except Exception as e:
        return "I am having technical difficulties, but I want you to know your feelings matter. Please try again after a short time."


# Test
print(query_medgemma(prompt="Hi,what is your name?"))



#Setup twilio API calling tool
from twilio.rest import Client
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,EMERGENCY_CONTACT,TWILIO_FROM_NUMBER

def call_emergency(EMERGENCY_CONTACT ):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call= client.calls.create(
        to=EMERGENCY_CONTACT,
        from_= TWILIO_FROM_NUMBER,
        url="http://demo.twilio.com/docs/voice.xml"
    )
    return f"Emergency call initiated to {EMERGENCY_CONTACT}"



#Setup location tool
#Create an ai agent and link to backened.