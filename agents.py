from pathlib import Path

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


def _groq_api_key() -> str:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to .env "
            "(create a key at https://console.groq.com/keys — it starts with gsk_)."
        )
    if key.startswith("mstrl_"):
        raise ValueError(
            "GROQ_API_KEY looks like a Mistral key (mstrl_). "
            "Create a Groq key at https://console.groq.com/keys and set GROQ_API_KEY=gsk_..."
        )
    return key


# Groq retired llama-3.3-70b-versatile (Aug 2026); override via GROQ_MODEL in .env
_DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# model setup
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", _DEFAULT_GROQ_MODEL).strip() or _DEFAULT_GROQ_MODEL,
    temperature=0.7,
    groq_api_key=_groq_api_key(),
)


#1st agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search]
    )

#2nd agent 

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url]
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
