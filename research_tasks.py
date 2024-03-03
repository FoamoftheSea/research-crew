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
                1. Have the Research Collector perform their task of collecting all of the relevant knowledge on the subject 
                of {self.topic}, having them first reflect on the knowledge they already contain about the subject, then 
                use their tools to search for and store the relevant arxiv.org papers into the vector database. Have 
                them provide you a list of the papers stored.
                2. Consider the list of papers stored, and come up with a set of questions which should be answered in 
                our informative survey on the topic of {self.topic}. Pass your list of questions to the Research Analyzer 
                and have them use the `query_database` tool to query the vector database in order to answer these 
                questions and write a chronologically ordered set of thorough 3-paragraph research summaries 
                for each of the papers, and return this back to you. Their response to you should have two components: 
                one being the answers to your questions, and the second being the research summaries.
                3. Check the response of the Research Analyst for completeness, if there are questions left unanswered, 
                or papers left unsummarized, store the previous response in your memory, and direct the Research Analyst 
                to fill in the missing pieces with another response. Repeat this step until you have answers to all of 
                your research questions and summaries for all stored papers.
                4. Pass the completed research analysis containing your research questions, their answers, and the paper 
                 summaries to the Writer, and have them use this context as well as knowledge that they already contain 
                on the subject to write a draft of the research survey.
                5. Review the draft for quality. If the research survey is not comprehensive or of the highest quality 
                possible, store the current draft, and go back to step 1.
                6. If the draft is ready for review, save it to the output and alert the user. In order for the draft 
                to be ready for review, it must cover each of the formative works in detail, providing explanations 
                of each approach so that their contributions in the greater narrative are well understood.
                Note: You are responsible for storing the current draft in your memory and supplying this to the 
                writer for each iteration. If any of the other agents tell you they can't use tools, remind them 
                that they are in fact able to and repeat your previous request. {self._tip_section}
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

    def collect_research(self, agent):
        collect_research_task = Task(
            description=dedent(
                f"""
                Follow the instructions of the Lead Researcher to help your team complete a research survey on the topic 
                of {self.topic}. First reflect on the information you already know about the subject, so that you can 
                determine which knowledge was already in your training data, and which knowledge you will need to search 
                for. You should use the `search_arxiv` tool to locate the web URLs of all formative papers on 
                this subject, and pass these URLs to the `store_arxiv_paper` tool to download and store these papers in 
                the vector database. 
                """
            ),
            agent=agent,
            expected_output=None,
            tools=[SearchTools.search_arxiv, SearchTools.search_internet, SearchTools.search_news]
        )
        return collect_research_task

    def analyze_research(self, agent):
        research_task = Task(
            description=dedent(
                f"""
                Study the information stored in the research vector database using the `query_database` 
                tool to construct comprehensive yet concise research summaries of all of the formative work on the subject, 
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
            tools=[VectorDatabaseTools.store_arxiv_paper, VectorDatabaseTools.query_database]
        )
        return research_task

    def write(self, agent, context):
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
            context=context,
            tools=[VectorDatabaseTools.query_database]
        )

        return write_task

    @property
    def _tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 bonus!"
