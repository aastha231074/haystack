import requests
from typing import List
from haystack import Document, Pipeline, component
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

@component
class HackernewsNewestFetcher: 
    def __init__(self): 
        fetcher=LinkContentFetcher()
        converter=HTMLToDocument()

        html_coversion_pipeline = Pipeline()
        html_coversion_pipeline.add_component("fetcher", fetcher)
        html_coversion_pipeline.add_component("converter", converter)

        html_coversion_pipeline.connect("fetcher", "converter")

        self.html_pipeline = html_coversion_pipeline

    @component.output_types(articles=List[Document])
    def run(self, top_k: int): 
        articles= []
        trending_list = requests.get(
            url="https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"
        )
        for id in trending_list.json()[0:top_k]:
            post = requests.get(
                url=f"https://hacker-news.firebaseio.com/v0/item/{id}.json?print=pretty"
            )
            if "url" in post.json(): 
                try: 
                    article = self.html_pipeline.run(
                        {"fetcher": {"urls": [post.json()["url"]]}}
                    )
                    articles.append(article["converter"]["documents"][0])
                except:
                    print(f"Can't download {post}, skipped")
            elif "text" in post.json():
                try:
                    articles.append(Document(content=post.json()["text"], meta= {"title": post.json()["title"]}))
                except:
                    print(f"Can't download {post}, skipped")
        return {"articles": articles}
    
fetcher = HackernewsNewestFetcher()
results = fetcher.run(top_k=3)

# prompt_template = """  
#     You will be provided a few of the top posts in HackerNews.  
#     For each post, provide a brief summary if possible.
    
#     Posts:  
#     {% for article in articles %}
#     Post:\n
#     {{ article.content}}
#     {% endfor %}  
# """

prompt_template = """  
    You will be provided a few of the top posts in HackerNews, followed by their URL.  
    For each post, provide a brief summary followed by the URL the full post can be found at.  
    
    Posts:  
    {% for article in articles %}  
    {{ article.content }}
    URL: {{ article.meta["url"] }}
    {% endfor %}  
"""

prompt_builder = PromptBuilder(template=prompt_template)
fetcher = HackernewsNewestFetcher()
llm = OllamaGenerator(
    model="zephyr",
    generation_kwargs={
        "num_predict": 500,
        "temperature": 0.9,
    },
)

summarizer_pipeline = Pipeline()
summarizer_pipeline.add_component("fetcher", fetcher)
summarizer_pipeline.add_component("prompt", prompt_builder)
summarizer_pipeline.add_component("llm", llm)

summarizer_pipeline.connect("fetcher.articles", "prompt.articles")
summarizer_pipeline.connect("prompt", "llm")

summaries = summarizer_pipeline.run({"fetcher": {"top_k": 3}})

print(summaries["llm"]["replies"][0])