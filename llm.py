class LLM:

    @staticmethod
    def claude(model_name="claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)

    @staticmethod
    def claude3_sonnet(temperature=0.2, max_tokens=1024):
        return LLM.claude(model_name="claude-3-sonnet-20240229", temperature=temperature, max_tokens=max_tokens)

    @staticmethod
    def claude3_opus(temperature=0.2, max_tokens=1024):
        return LLM.claude(model_name="claude-3-opus-20240229", temperature=temperature, max_tokens=max_tokens)

    @staticmethod
    def chatgpt(model_name="gpt-4-turbo-preview", temperature=0.5):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model_name=model_name, temperature=temperature)

    @staticmethod
    def gpt4_turbo(temperature=0.5):
        return LLM.chatgpt(model_name="gpt-4-turbo-preview", temperature=temperature)

    @staticmethod
    def gpt3pt5_turbo(temperature=0.5):
        return LLM.chatgpt(model_name="gpt-3.5-turbo", temperature=temperature)

    @staticmethod
    def ollama(model_name="mistral:instruct"):
        from langchain_community.llms import Ollama
        return Ollama(model=model_name)

    @staticmethod
    def mistral(instruct=True):
        model_name = "mistral" if not instruct else "mistral:instruct"
        return LLM.ollama(model_name=model_name)

    @staticmethod
    def mixtral(instruct=True):
        model_name = "mixtral" if not instruct else "mixtral:instruct"
        return LLM.ollama(model_name=model_name)

    @staticmethod
    def nexusraven():
        return LLM.ollama(model_name="nexusraven")
