from haystack.components.builders import PromptBuilder
from haystack import Document, Pipeline, component
from haystack_integrations.components.generators.ollama import OllamaGenerator

@component
class Greeter: 
    @component.output_types(greeting=str)
    def run(self, user_name: str): 
        return {"greeting": f"Hello {user_name}"}
    
greeter = Greeter()
template="""
    You will be given the beginning of a dialouge. Create a short play script using this as the start of the play. 
    Start of dialouge: {{dialouge}}
    Full script: 
"""

prompt = PromptBuilder(template=template)
llm = OllamaGenerator(
    model="zephyr",
    generation_kwargs={
        "num_predict": 500,
        "temperature": 0.9,
    },
)

dialouge_builder = Pipeline()
dialouge_builder.add_component("greeter", greeter)
dialouge_builder.add_component("prompt", prompt)
dialouge_builder.add_component("llm", llm)

dialouge_builder.connect("greeter.greeting", "prompt.dialouge")
dialouge_builder.connect("prompt", "llm")

dialouge = dialouge_builder.run({"greeter": {"user_name": "Tuana"}})

print(dialouge["llm"]["replies"][0])