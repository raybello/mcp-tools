from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel

import uvicorn

########################

app = FastAPI()

mcp = FastApiMCP(
    app,
    name="Multi-tool MCP Server",
    description="Current MCP server",
    describe_all_responses=True,
    describe_full_response_schema=True,
)

mcp.mount()


# Pydantic model for the POST request body
class TranscriptRequest(BaseModel):
    url: str
    
class SearchRequest(BaseModel):
    url: str


# This endpoint will not be registered as a tool, since it was added after the MCP instance was created
@app.get("/v1/sample/", operation_id="get_sample", response_model=dict[str, str])
async def get_sample():
    return {"message": "Hello, world!"}


@app.post(
    "/v1/transcript/", operation_id="get_transcript", response_model=dict[str, str]
)
async def get_transcript(request: TranscriptRequest):
    url = request.url

    ytt_api = YouTubeTranscriptApi()

    fetched_transcript = ytt_api.fetch(url.split("=")[1])
    transcript = " ".join([snippet.text for snippet in fetched_transcript])

    return {"transcript": f"{transcript}"}

# Endpoint that given a search topic, searches google and formats the content of the first 3 responses
# as a text file.
@app.post("/v1/search", operation_id="search_web", response_model=dict[str, str])
async def search_web(request: SearchRequest):
    # TODO: Implement this
    return {"search": "Function not yet implemented"}
    


# But if you re-run the setup, the new endpoints will now be exposed.
mcp.setup_server()


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8070)
