import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

search_tool = SerperDevTool()

def create_research_agent():

    llm=ChatGroq(temperature=0,
             model_name="llama3-70b-8192",
             api_key=GROQ_API_KEY)
  
  
    return Agent(
        role="Research Specialist",
        goal="Conduct thorough research on given topics",
        backstory="You are an experienced researcher with expertise in finding and synthesizing information from various sources",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )




def create_research_task(agent, topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output = "A detailed summary of the research findings, including key points and insights related to the topic"
    )

def run_research(topic):
    agent = create_research_agent()
    task = create_research_task(agent, topic)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("Welcome to the Research Agent!")
    topic = input("Enter the research topic: ")
    result = run_research(topic)
    print("Research Result:")
    print(result)



# import os
# from langchain_groq import ChatGroq

# llamaChatModel = ChatGroq(
#     model="llama3-70b-8192"
# )


# messages = [
#     ("system", "You are an historian expert in the Kennedy family."),
#     ("human", "How many members of the family died tragically?"),
# ]

# llamaResponse = llamaChatModel.invoke(messages)

# print(llamaResponse.content)
