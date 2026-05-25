from dotenv import load_dotenv
from typing import List
from colorama import Fore
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

load_dotenv()

@component
class EntitiesValidator: 

    @component.output_types(entities_to_validate=str, entities=str)
    def run(self, replies: List[str]): 
        if 'DONE' in replies[0]: 
            return {"entities": replies[0].replace('DONE', '')}
        else: 
            print(Fore.RED + 'Reflecting on entities\n', replies[0])
            return {"entities_to_validate": replies[0]}

template = """"
    {% if entities_to_validate %}
        Here was the text you were provided:
        {{ text }}
        Here are the entities you previously extracted: 
        {{ entities_to_validate[0] }}
        Are these the correct entities? 
        Things to check for:
        - Entity categories should exactly be "Person", "Location" and "Date"
        - There should be no extra categories
        - There should be no duplicate entities
        - If there are no appropriate entities for a category, the category should have an empty list
        If you are done say 'DONE' and return your new entities in the next line
        If not, simply return the best entities you can come up with.
        Entities:
    {% else %}
        Extract entities from the following text
        Text: {{ text }} 
        The entities should be presented as key-value pairs in a JSON object.
        Example: 
        {
            "Person": ["value1", "value2"], 
            "Location": ["value3", "value4"],
            "Date": ["value5", "value6"]
        }
        If there are no possibilities for a particular category, return an empty list for this
        category
        Entities:
    {% endif %}
"""

prompt_builder = PromptBuilder(template=template)
llm = OllamaGenerator(model="zephyr",generation_kwargs={"num_predict": 500,"temperature": 0.9,},)
entities_validator = EntitiesValidator()

self_reflecting_agent = Pipeline(max_runs_per_component=10)
self_reflecting_agent.add_component("prompt_builder", prompt_builder)
self_reflecting_agent.add_component("llm", llm)
self_reflecting_agent.add_component("entities_validator", entities_validator)

self_reflecting_agent.connect("prompt_builder.prompt", "llm.prompt")
self_reflecting_agent.connect("llm.replies", "entities_validator.replies")
self_reflecting_agent.connect("entities_validator.entities_to_validate", "prompt_builder.entities_to_validate")

# text = """
#     Istanbul is the largest city in Turkey, straddling the Bosporus Strait, 
#     the boundary between Europe and Asia. It is considered the country's economic, 
#     cultural and historic capital. The city has a population of over 15 million residents, 
#     comprising 19% of the population of Turkey,[4] and is the most populous city in Europe 
#     and the world's fifteenth-largest city.
# """

# result = self_reflecting_agent.run({"prompt_builder": {"text": text}})
# print(Fore.GREEN + result['entities_validator']['entities'])

text = """
Stefano: Hey all, let's start the all hands for June 6th 2024
Geoff: Thanks, I'll kick it off with a request. Could we please add persistent memory to the Chroma document store.
Stefano: Easy enough, I can add that to the feature requests. What else?
Julain: There's a bug, some BM25 algorithms return negative scores and we filter them out from the results by default.
Instead, we should probably check which algorithm is being used and keep results with negative scores accordingly.
Esmail: Before we end this call, we should add a new Generator component for LlamaCpp in the next release.
Tuana: Thanks all, I think we're done here, we can create some issues in GitHub about these."""

result = self_reflecting_agent.run({"prompt_builder": {"text": text}})
print(Fore.GREEN + result['entities_validator']['entities'])

