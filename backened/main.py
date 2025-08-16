from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from ai_agent import graph, SYSTEM_PROMPT, parse_response

# Step 1: Create app
app = FastAPI()

# Step 2: Add CORS (must be after app is created)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now, later you can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 3: Receive and validate request from frontend
class Query(BaseModel):
    message: str

@app.post("/ask")
async def ask(query: Query):
    inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", query.message)]}
    stream = graph.stream(inputs, stream_mode="updates")
    tool_called_name, final_response = parse_response(stream)

    return {
        "response": final_response,
        "tool_called": tool_called_name
    }

# Step 4: Run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
