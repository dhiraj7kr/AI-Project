import requests
from bs4 import BeautifulSoup
import urllib.parse

def search_reddit_web(query, max_results=5):
    # Encode query for URL
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.reddit.com/search/?q={encoded_query}"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("❌ Failed to fetch search results.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    posts = soup.find_all("a", href=True)

    results = []
    for link in posts:
        href = link['href']
        title = link.get_text().strip()

        # Only get valid post links and filter out duplicates/ads
        if "/r/" in href and "comments" in href and title and len(title) > 15:
            full_link = urllib.parse.urljoin("https://www.reddit.com", href)
            results.append((title, full_link))

            if len(results) >= max_results:
                break

    return results

# === Main Program ===
if __name__ == "__main__":
    print("🔴 Reddit Web Search")
    question = input("👉 Ask your question: ").strip()

    if question:
        results = search_reddit_web(question)

        if results:
            print("\n✅ Top Reddit posts:\n")
            for i, (title, link) in enumerate(results, 1):
                print(f"{i}. {title}\n   {link}")
        else:
            print("\n⚠️ No results found. Try different keywords.")
    else:
        print("❗ Please enter a valid question.")
