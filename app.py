from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Import necessary modules from LangChain and related packages
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ------------------ Configuration ------------------
# Set up the Gemini API key in the environment (if required by the package)
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Initialize the LLM with Gemini using the key from the environment
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

# Predefined query (does not come from the request)
PREDEFINED_QUERY = "Is the startup investable based on it's financial and risk analysis?"

# ------------------ Pydantic Models ------------------
class Startup(BaseModel):
    name: str
    industry: str
    funding: str
    details: str
    arr: str
    mrr: str
    cogs_percentage: str
    marketing: str
    cac: str
    transport_and_logistics: str
    gross_margin: str
    ebitda: str
    pat: str
    salaries: str
    miscellaneous: str

class KnowledgeBaseInput(BaseModel):
    knowledge_base: List[Startup]

class InvestmentOutput(BaseModel):
    retrieved_data: str = Field(description="Retrieved startup information")
    risk_analysis: str = Field(description="Risk analysis for each startup")
    financial_analysis: str = Field(description="Financial analysis for each startup")
    investment_recommendations: str = Field(description="Investment recommendations")

# ------------------ FastAPI App ------------------
app = FastAPI(title="Investment Analysis API")

def run_investment_analysis_for_kb(knowledge_base: List[dict], query: str) -> InvestmentOutput:
    """
    Given a knowledge base (list of startup dictionaries) and a predefined query,
    this function builds the vector database, retrieves the relevant information,
    and then executes risk, financial, and recommendation analysis.
    """
    # Convert each startup into a structured text document
    startup_texts = [
        (
            f"Name: {s['name']}\n"
            f"Industry: {s['industry']}\n"
            f"Funding: {s['funding']}\n"
            f"Details: {s['details']}\n"
            f"Financials: ARR: {s['arr']}, MRR: {s['mrr']}, COGS Percentage: {s['cogs_percentage']}, "
            f"Marketing: {s['marketing']}, CAC: {s['cac']}, Transport & Logistics: {s['transport_and_logistics']}, "
            f"Gross Margin: {s['gross_margin']}, EBITDA: {s['ebitda']}, PAT: {s['pat']}\n"
            f"Salaries: {s['salaries']}\n"
            f"Miscellaneous: {s['miscellaneous']}"
        )
        for s in knowledge_base
    ]

    # Setup vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": TOKEN}
    )
    docs = [Document(page_content=text) for text in startup_texts]
    vector_db = FAISS.from_documents(docs, embeddings)

    # Core function: Retrieve startup info from vector database
    def retrieve_startup_info(query: str) -> str:
        results = vector_db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    # Core function: Analyze investment risks using Gemini (styled as Jordan Belfort)
    def risk_analysis(query: str) -> str:
        retrieved = retrieve_startup_info(query)
        risk_prompt = PromptTemplate(
            input_variables=["query", "retrieved_data"],
            template=(
                "Based on the following startup information:\n{retrieved_data}\n\n"
                "Analyze the investment risks for each startup given the market conditions described in the query: {query}.\n"
                "Speak and write exactly like Jordan Belfort (the Wolf of Wall Street). "
                "Your communication style is extremely confident, aggressive, and persuasive. You use CAPITAL LETTERS for emphasis, "
                "plenty of exclamation marks, and bold claims about 'getting rich', 'crushing the market', and 'absolute no-brainers'. "
                "For each startup, provide a detailed risk analysis with a clear risk classification of either High, Medium, or Low.\n"
                "Include specific factors such as market volatility, competition, regulatory considerations, and technology risks.\n"
                "Format your response in a clear, bullet-point structure with each startup clearly labeled. "
                "STRICTLY GIVE OUTPUT WITHIN 200 WORDS."
            )
        )
        risk_chain = risk_prompt | llm
        result = risk_chain.invoke({
            "query": query,
            "retrieved_data": retrieved
        })
        return result.content

    # Core function: Analyze financial performance using Gemini (styled as Jordan Belfort)
    def financial_analysis(query: str) -> str:
        retrieved = retrieve_startup_info(query)
        finance_prompt = PromptTemplate(
            input_variables=["query", "retrieved_data"],
            template=(
                "Based on the following startup financial information:\n{retrieved_data}\n\n"
                "Provide a comprehensive financial analysis of each startup using the provided metrics (ARR, MRR, COGS Percentage, "
                "Marketing, CAC, Transport & Logistics, Gross Margin, EBITDA, PAT, Salaries, and Miscellaneous) and consider the market conditions in the query: {query}.\n"
                "For each startup, include:\n"
                "Speak and write exactly like Jordan Belfort (the Wolf of Wall Street). "
                "Your communication style is extremely confident, aggressive, and persuasive. You use CAPITAL LETTERS for emphasis, "
                "plenty of exclamation marks, and bold claims about 'getting rich', 'crushing the market', and 'absolute no-brainers'. "
                "1. Overall financial health assessment (Strong, Moderate, or Weak)\n"
                "2. Key financial strengths and weaknesses\n"
                "3. Growth potential based on current financials\n"
                "4. Efficiency metrics analysis (margins, spending ratios)\n"
                "5. STRICTLY GIVE OUTPUT WITHIN 200 WORDS. "
                "Format your response in a clear, detailed structure with each startup clearly labeled."
            )
        )
        finance_chain = finance_prompt | llm
        result = finance_chain.invoke({
            "query": query,
            "retrieved_data": retrieved
        })
        return result.content

    # Core function: Generate investment recommendations using Gemini (styled as Jordan Belfort)
    def generate_recommendation(query: str) -> str:
        retrieved_data = retrieve_startup_info(query)
        risk_result = risk_analysis(query)
        finance_result = financial_analysis(query)
        recommendation_prompt = PromptTemplate(
            input_variables=["query", "retrieved_data", "risk_analysis", "financial_analysis"],
            template=(
                "Based on the retrieved startup information:\n{retrieved_data}\n\n"
                "And the following risk analysis:\n{risk_analysis}\n\n"
                "And the following financial analysis:\n{financial_analysis}\n\n"
                "Provide comprehensive investment recommendations for: {query}\n\n"
                "Speak and write exactly like Jordan Belfort (the Wolf of Wall Street). "
                "Your communication style is extremely confident, aggressive, and persuasive. You use CAPITAL LETTERS for emphasis, "
                "plenty of exclamation marks, and bold claims about 'getting rich', 'crushing the market', and 'absolute no-brainers'. "
                "Include in your response:\n"
                "1. Primary recommendation with clear rationale\n"
                "2. Alignment with the investment criteria\n" 
                "3. Risk considerations and mitigations\n"
                "4. Financial justification\n"
                "5. Important considerations for the investor\n"
                "6. Alternative options if applicable\n"
                "Be specific and provide actionable advice."
            )
        )
        recommendation_chain = recommendation_prompt | llm
        result = recommendation_chain.invoke({
            "query": query,
            "retrieved_data": retrieved_data,
            "risk_analysis": risk_result,
            "financial_analysis": finance_result
        })
        return result.content

    # Execute the analysis steps
    retrieved = retrieve_startup_info(query)
    risk_result = risk_analysis(query)
    finance_result = financial_analysis(query)
    recommendation = generate_recommendation(query)
    
    # Package the results into the InvestmentOutput model
    output = InvestmentOutput(
        retrieved_data=retrieved,
        risk_analysis=risk_result,
        financial_analysis=finance_result,
        investment_recommendations=recommendation
    )
    return output

@app.post("/investment-analysis", response_model=InvestmentOutput)
async def investment_analysis(kb: KnowledgeBaseInput):
    """
    API endpoint that accepts a knowledge base (list of startups) and returns the investment analysis report.
    The query used in the analysis is predefined.
    """
    # Convert the incoming Pydantic model to a list of dictionaries
    knowledge_base = [startup.dict() for startup in kb.knowledge_base]
    result = run_investment_analysis_for_kb(knowledge_base, PREDEFINED_QUERY)
    return result

# ------------------ Run the App ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
