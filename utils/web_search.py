import os, requests

def web_search(query, num_results=5):
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return ["⚠️ No SerpAPI key configured."]
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": api_key}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet") or ""
            results.append(f"{title} — {snippet}\n{link}")
        return results
    except Exception as e:
        return [f"Error during web search: {e}"]
