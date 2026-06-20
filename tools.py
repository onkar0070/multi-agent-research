from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from rich import print

load_dotenv()


def _get_tavily_client():
    """Return a TavilyClient if the TAVILY_API_KEY is set, otherwise None."""
    try:
        from tavily import TavilyClient
    except Exception:
        return None

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs and snippets.

    If Tavily is not configured, returns an informative error message instead of raising.
    """
    tavily = _get_tavily_client()
    if not tavily:
        return (
            "Tavily client not available. Please install 'tavily-python' and set TAVILY_API_KEY in your environment."
        )

    try:
        results = tavily.search(query=query, max_results=5)
    except Exception as e:
        return f"Tavily search failed: {e}"

    out = []
    for r in results.get("results", []):
        out.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('content','')[:300]}\n")

    return "\n----\n".join(out)

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
