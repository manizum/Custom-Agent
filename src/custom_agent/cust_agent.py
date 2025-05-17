from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key and model
gemini_api_key = os.getenv("GEMINI_API_KEY")

#Using AsyncOpenAI for async calls
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Agent SDK Configuration
set_default_openai_client(client)
set_tracing_disabled(True)

# Setting up the agent
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Definig Agent with Behavior
financial_agent = Agent(
    name="Financial Advisor",
    instructions="""
    Analyze user's income and expenses to suggest budgeting plans.
    Calculate monthly savings and recommend improvement strategies.
    Estimate emergency fund requirements based on user lifestyle.
    Evaluate debt profile and suggest repayment priorities (e.g., snowball vs avalanche).
    Recommend optimal savings vs investment ratios.
    Explain common investment types (stocks, bonds, ETFs, etc.).
    Compare investment options based on user goals and risk appetite.
    Calculate compound interest on given savings or investment inputs.
    Track and report current stock prices using symbols (e.g., AAPL, TSLA).
    Recommend diversified investment portfolios for short, medium, and long term.
    Analyze historical performance of selected assets or funds.
    Compare loan options (personal, auto, mortgage) by interest and term.
    Calculate EMI for any loan based on principal, rate, and duration.
    Suggest retirement planning based on age, savings, and desired lifestyle.
    Calculate projected retirement corpus using current contribution.
    Offer tax-saving tips based on common deductions or allowances.
    Explain financial jargon in simple language (e.g., "What is an ETF?").
    Flag suspicious financial habits or overspending patterns.
    Recommend financial tools (apps, platforms) for personal use.
    Generate monthly or yearly financial summary reports.
    Identify gaps in insurance coverage (life, health, assets).
    Simulate financial impact of major goals (buying home, car, education).
    Forecast net worth growth based on current financial strategy.
    Monitor economic trends and alert on relevant events (e.g., inflation, interest rates).
    Provide links or summaries to latest financial news (via plugin/API integration).
    Advise on building credit score and managing credit cards wisely.
    Offer personalized financial checklists (e.g., for new job, moving, marriage).
    Recommend side income or passive income ideas tailored to user skills.
    Answer regulatory or compliance-related finance queries in user's region.
    Adapt advice dynamically based on real-time data and user feedback.    
    """,
    model=model,
)

from datetime import datetime

def run_financial_agent():
    log_file = "Output.md"
    
    # Write session start with timestamp
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n##  Session started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("What financial advice do you need? (type 'exit' to stop)")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Exiting the financial agent.")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n**Session ended.**\n")
            break

        result = Runner.run_sync(financial_agent, user_input)
        response = result.final_output
        print("\nFinancial Advisor: ", response)

        # Append both user input and assistant response to the log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n**User**: {user_input}\n\n")
            f.write(f"**Financial Advisor**: {response}\n")
