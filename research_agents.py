from textwrap import dedent

from crewai import Agent
from langchain_community.llms import Ollama
from tools.search_tools import SearchTools
from tools.vector_database_tools import VectorDatabaseTools


class ResearchAgents:
    def __init__(self, topic: str):
        self.topic = topic

    def lead_researcher(self, llm=Ollama(model="mistral:instruct")):
        # Creating a senior researcher agent with memory and verbose mode
        lead_researcher = Agent(
            role="Lead Researcher",
            goal=f"""Lead your team to complete an in-depth research survey of {self.topic}, constructing a linear 
            narrative through the formative works in this field.""",
            verbose=True,
            memory=True,
            backstory=dedent(
                f"""
                You are a senior researcher working at a leading AI think tank who leads a team of researchers in 
                writing comprehensive, in-depth research surveys of various topics in the field of Artificial 
                Intelligence. Your team produces work of the highest standards, since you never give your stamp of 
                approval on shoddy or incomplete work.
                """
            ),
            tools=[
                # SearchTools.search_arxiv,
                # SearchTools.search_internet,
                # SearchTools.search_news,
                # VectorDatabaseTools.store_arxiv_paper,
                # VectorDatabaseTools.chat_with_docs
            ],
            allow_delegation=True,
            llm=llm,
        )
        return lead_researcher

    def research_collector(self, llm=Ollama(model="mistral:instruct"), function_calling_llm=None):
        researcher = Agent(
            role="Research Collector",
            goal=dedent(
                f"""
                Follow instructions from the Lead Researcher to help your team achieve its goal of writing an in-depth 
                and comprehensive research summary on the topic of {self.topic}. Use `search_arxiv` to search for 
                relevant papers, and then store each of them into the vector database with `store_arxiv_paper`.
                """
            ),
            verbose=True,
            memory=True,
            backstory=dedent(
                f"""
                You work at a leading AI think tank as a researcher, known for your ability to effectively conduct 
                comprehensive research on a given topic, store all the formative papers in a local vector database, 
                and write extraordinarily clear and concise research summaries of these papers that your team writer 
                uses to draft exceptional in-depth research surveys.
                """
            ),
            tools=[
                SearchTools.search_arxiv,
                SearchTools.search_internet,
                SearchTools.search_news,
                VectorDatabaseTools.store_arxiv_paper,
            ],
            allow_delegation=False,
            llm=llm,
            function_calling_llm=function_calling_llm or llm,
        )
        return researcher

    def research_analyzer(self, llm=Ollama(model="mistral:instruct"), function_calling_llm=None):
        # Create an agent to analyze information stored in the research database and create summaries
        research_analyzer = Agent(
            role="Research Analyzer",
            goal=f"Analyze the research stored in the vector database to write a set of comprehensive paper summaries on the topic of {self.topic}.",
            verbose=True,
            memory=True,
            backstory=dedent(
                f"""
                You work at a leading AI think tank as a researcher, known for your ability to write extraordinarily 
                clear and concise research summaries of the papers stored in a research database.
                """
            ),
            tools=[VectorDatabaseTools.query_database],
            allow_delegation=False,
            llm=llm,
            function_calling_llm=function_calling_llm or llm,
        )
        return research_analyzer

    def writer(self, llm=Ollama(model="mistral:instruct")):
        # Creating a writer agent with custom tools and delegation capability
        writer = Agent(
            role='Writer',
            goal=dedent(
                f"""
                Use the research summaries provided by your team to complete thorough, detailed, and fully-cited drafts 
                of research surveys on the topic of {self.topic}. Your surveys must cover all of the information 
                provided to you in the research summaries, tying your explanations of their approaches and findings 
                into an engaging chronological narrative. Include all citation details for the papers covered.
                """
            ),
            verbose=True,
            memory=True,
            backstory=dedent(
                f"""
                You work at a leading AI think tank as a writer, known for your ability to draft engaging yet in-depth 
                content on the topic at hand through the creation of a comprehensive linear narrative through the 
                formative works in that topic, providing all citations for the works discussed. You follow the direction 
                of the Senior Researcher.
                """
            ),
            tools=[],
            allow_delegation=False,
            llm=llm,
        )
        return writer
