import re
import requests
import time
from bs4 import BeautifulSoup

URL = "https://www.creepypasta.se/poddmanuskript/"
CLEANR = re.compile("<.*?>")


def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def get_story(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html5lib")
    article = soup.find("article").get_text()
    return article


r = requests.get(URL)
soup = BeautifulSoup(
    r.content, "html5lib"
)  # If this line causes an error, run 'pip install html5lib' or install html5lib
links = soup.find("div", attrs={"class": "entry-content"})
urls = [l["href"] for l in links.findAll("a") if "poddmanuskript" in l["href"]]

from_menu = soup.findAll("li", attrs={"class": "page_item"})
menu_urls = [
    l.find("a")["href"] for l in from_menu if "poddmanuskript" in l.find("a")["href"]
]
unique_urls = set(menu_urls + urls)
print(len(unique_urls), "avsnitt hittade.")

stories = []
for i, url in enumerate(unique_urls):
    story_parts = (
        get_story(url)
        .split("Du har lyssnat på Creepypodden")[0]
        .split("Signaturmelodi")
    )
    if len(story_parts[-1]) < 1200:
        story_parts = story_parts[:-1]
    story = "***".join(story_parts)
    story = cleanhtml(story)
    story = (
        story.replace("\n\n\n\n", "\n")
        .replace("\n\n\n", "\n\n")
        .replace("\t\t\t", "\t")
        .replace("\t\t", "\t")
    )
    story = (
        story.replace("“", '"')
        .replace("”", '"')
        .replace("…", "...")
        .replace("—", "-")
        .replace("–", "-")
        .replace("−", "-")
        .replace("‑", "-")
    )
    words = len(story.split(" "))
    print(f"Avsnitt {i+1}, {words} ord")
    stories.append(story)
    time.sleep(2)

with open("datasets/corpus_train.txt", "w", encoding="utf-8") as f:
    f.write("\n<|endoftext|>".join(stories[5:]))
    
with open("datasets/corpus_validation.txt", "w", encoding="utf-8") as f:
    f.write("\n<|endoftext|>".join(stories[:5]))
