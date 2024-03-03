from crewai import Crew, Process
from research_tasks import ResearchTasks
from research_agents import ResearchAgents
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI, OpenAI

USE_OPENAI = True

# Topic for the crew run
topic = 'Quantization of large language models'

if USE_OPENAI:
    smart_model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.5)
    small_model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.5)
    # model_name = "gpt-3.5-turbo-1106"
    research_agents = ResearchAgents(topic=topic)
else:
    smart_model = Ollama(model="mixtral:instruct")
    small_model = Ollama(model="mistral:instruct")
    research_agents = ResearchAgents(topic=topic)

research_tasks = ResearchTasks(topic=topic)

lead_researcher = research_agents.lead_researcher(llm=smart_model)
research_collector = research_agents.research_collector(llm=smart_model, function_calling_llm=small_model)
research_analyzer = research_agents.research_analyzer(llm=smart_model, function_calling_llm=small_model)
writer = research_agents.writer(llm=smart_model)

lead_research_task = research_tasks.lead_research(lead_researcher)
collect_research_task = research_tasks.collect_research(research_collector)
analyze_research_task = research_tasks.analyze_research(research_analyzer)
write_task = research_tasks.write(writer, [analyze_research_task])

# Forming the tech-focused crew with enhanced configurations
crew = Crew(
  agents=[lead_researcher, research_collector, research_analyzer, writer],
  tasks=[lead_research_task],
  process=Process.sequential,
  verbose=2,
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff()
print(result)
