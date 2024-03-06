from crewai import Crew, Process
from llm import LLM

from research_tasks import ResearchTasks
from research_agents import ResearchAgents

USE_OPENAI = True

# Topic for the crew run
topic = 'Quantization of large language models'

llm = LLM.gpt4_turbo()
fn_llm = LLM.gpt4_turbo()

research_agents = ResearchAgents(topic=topic)
research_tasks = ResearchTasks(topic=topic)

lead_researcher = research_agents.lead_researcher(llm=llm, function_calling_llm=fn_llm)
research_collector = research_agents.research_collector(llm=llm, function_calling_llm=fn_llm)
research_analyzer = research_agents.research_analyzer(llm=llm, function_calling_llm=fn_llm)
writer = research_agents.writer(llm=llm)

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
