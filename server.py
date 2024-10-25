
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic models
class Question(BaseModel):
    question: str

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="mixtral-8x7b-32768",
    api_key=os.getenv("GROQ_API_KEY")
)
output_parser = StrOutputParser()

@app.get("/")
async def read_root():
    return {"message": "Welcome to LangChain API with Groq"}

# Note: Adding trailing slash to make it match both with and without slash
@app.post("/ask/")
async def get_answer(question: Question):
    try:
        # Create a simple prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Please provide clear and concise responses."),
            ("user", "{question}")
        ])
        
        # Create chain and invoke
        chain = prompt | llm | output_parser
        
        # Get response
        response = chain.invoke({"question": question.question})
        
        return {"answer": response}
        
    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
