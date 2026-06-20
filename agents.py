from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

# Defensive LLM selection
llm = None
_mistral_available = False
try:
    # try the mistral integration package used in the original code
    from langchain_mistralai import ChatMistralAI  # type: ignore
    _mistral_available = True
except Exception:
    ChatMistralAI = None
    _mistral_available = False

if _mistral_available:
    # Ensure the API key is present; otherwise Mistral will return 401 at runtime
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Set it in your environment or .env file before running."
        )
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.7)
else:
    # Fallback to OpenAI Chat model if available
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(temperature=0.7)
    except Exception:
        llm = None

if llm is None:
    raise RuntimeError(
        "No LLM available. Install and configure Mistral (langchain-mistralai) with MISTRAL_API_KEY,\n"
        "or set OPENAI_API_KEY and install langchain-openai. See requirements.txt and .env for examples."
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
