from crewai import Task
from textwrap import dedent

from tools.search_tools import SearchTools
from tools.vector_database_tools import VectorDatabaseTools


class ResearchTasks:

    def __init__(self, topic):
        self.topic = topic

    def lead_research(self, agent):
        lead_research_task = Task(
            description=dedent(
                f"""
                You are the leader of a research surveying team. You direct the Researcher and the Writer to iteratively 
                perform their roles. The researcher will use tools to search the internet and write research summaries, 
                which you then provide to the Writer to have them produce a draft of the research survey. Each time the 
                Writer completes a draft, you will review it for quality, and if it is not complete or of the highest 
                quality possible, have your team repeat this process until the research survey is complete. The process 
                is enumerated here:
                1. Have the Researcher perform their task of collecting all of the relevant knowledge on the subject 
                of {self.topic}, having them first reflect on the knowledge they already contain about the subject, then 
                use their tools to search for and store the relevant arxiv.org papers into the vector database.
                2. Have the Researcher use the `chat_with_docs` tool to access that database in order to write a 
                chronologically ordered set of thorough 2-paragraph research summary for each of the papers, and return 
                this back to you.
                3. Review the research summaries, and if they are not sufficiently detailed to provide the writer with 
                all the information needed to complete a comprehensive survey, go back to step 1.
                4. Share these research summaries with the Writer, and have the Writer create a draft which fully
                covers the currently available knowledge, being sure that they cite the relevant sources.
                5. Review the draft for quality. If the research survey is not comprehensive or of the highest quality 
                possible, store the current draft, and go back to step 1.
                6. If the draft is ready for review, save it to the output and alert the user. 
                Note: You are responsible for storing the current draft in your memory and supplying this to the 
                writer for each iteration. {self._tip_section}
                """
            ),
            expected_output=dedent(
                f"""
                A comprehensive research survey on the topic of {self.topic}, which follows an engaging linear 
                narrative and must include citations for all works discussed.
                """
            ),
            output_file="survey_draft.md",
            agent=agent,
            tools=[],
        )
        return lead_research_task

    def research(self, agent):
        research_task = Task(
            description=dedent(
                f"""
                Follow the instructions of the Lead Researcher to help your team complete a research survey on the topic 
                of {self.topic}. First reflect on the information you already know about the subject, so that you can 
                determine which knowledge was already in your training data, and which knowledge you will need to search 
                for. You should use the `search_arxiv` tool to locate the web URLs of all formative papers on 
                this subject, and pass these URLs to the `store_arxiv_paper` tool to download and store these papers in 
                the vector database. Use the information stored in this vector database using the `chat_with_docs` tool 
                to construct comprehensive yet concise research summaries of all of the formative work on the subject, 
                complete with the relevant citation information. Your summaries should enable your team's writer to 
                construct a comprehensive, in-depth research survey on the subject.
                Note: {self._tip_section}
                """
            ),
            agent=agent,
            expected_output=dedent(
                f"""
                A long string of concise research summaries covering all of the formative works, theory, and 
                methodologies on the topic of {self.topic} which fully covers the knowledge stored in the vector 
                database, and includes all citation information the Writer will need.
                """
            ),
            tools=[
                SearchTools.search_arxiv,
                SearchTools.search_internet,
                SearchTools.search_news,
                VectorDatabaseTools.store_arxiv_paper,
                VectorDatabaseTools.chat_with_docs
            ]
        )
        return research_task

    def write(self, agent):
        write_task = Task(
            description=dedent(
                f"""
                Follow instructions from the Lead Researcher to complete drafts of a thorough and detailed research 
                survey on the topic of {self.topic}. The Lead Researcher may provide you with the current draft and/or 
                a set of research summaries which you will then edit according to the new information and instructions 
                that you receive, then pass this back to the Lead Researcher for review.
                {self._tip_section}
                """
            ),
            expected_output=dedent(
                f"""
                A comprehensive, long-form survey of formative and current research on the topic of {self.topic}, 
                which follows a clear, engaging, linear narrative through time, and includes citations for all work 
                discussed.
                """
            ),
            agent=agent,
            tools=[VectorDatabaseTools.chat_with_docs]
        )

        return write_task

    @property
    def _tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 bonus!"
