from crewai import Crew, Process
from research_tasks import ResearchTasks
from research_agents import ResearchAgents

# Topic for the crew run
topic = 'Quantization of large language models'
research_agents = ResearchAgents(topic=topic)
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
