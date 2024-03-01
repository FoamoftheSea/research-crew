from crewai import Crew, Process
from research_tasks import ResearchTasks
from research_agents import ResearchAgents
# from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

model_name = "gpt-4-1106-preview"
# model_name = "gpt-3.5-turbo-1106"

# Topic for the crew run
topic = 'Quantization of large language models'
research_agents = ResearchAgents(topic=topic, llm=ChatOpenAI(model_name=model_name, temperature=0.5))
research_tasks = ResearchTasks(topic=topic)

lead_researcher = research_agents.lead_researcher()
researcher = research_agents.researcher()
writer = research_agents.writer()

lead_research_task = research_tasks.lead_research(lead_researcher)
research_task = research_tasks.research(researcher)
write_task = research_tasks.write(writer)

# Forming the tech-focused crew with enhanced configurations
crew = Crew(
  agents=[lead_researcher, researcher, writer],
  tasks=[lead_research_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  verbose=2,
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff()
print(result)
