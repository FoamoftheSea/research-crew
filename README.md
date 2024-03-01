# Research Crew
![research-crew](research_crew.jpg)

A CrewAI research crew for helping expedite research and writing efforts.

## Installation instructions
1. Make sure you have conda or miniconda, and run `conda env create -f environment.yml`
2. Set your environment variables to include your OpenAI API Key.
   3. Linux: `OPENAI_API_KEY=<your key>`
   4. Windows: `$env:OPENAI_API_KEY=<your key>`
3. Modify the topic in main.py, then run it `python main.py`
4. Output saves to `survey_draft.md`
