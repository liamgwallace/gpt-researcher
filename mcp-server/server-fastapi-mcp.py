"""
GPT Researcher with FastAPI and FastAPI MCP

This script implements a FastAPI server for GPT Researcher, exposing
both standard REST API endpoints and MCP-compatible tools.
"""

import os
import sys
import uuid
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from fastapi_mcp import FastApiMCP
from gpt_researcher import GPTResearcher

# Load environment variables
load_dotenv()

from utils import (
    research_store,
    create_success_response,
    handle_exception,
    get_researcher_by_id,
    format_sources_for_response,
    format_context_with_sources,
    store_research_results,
    create_research_prompt
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

logger = logging.getLogger(__name__)

# Define Pydantic models for request/response validation
class ResearchRequest(BaseModel):
    query: str = Field(..., description="The research query or topic")

class ReportRequest(BaseModel):
    research_id: str = Field(..., description="The ID of the research session")
    custom_prompt: Optional[str] = Field(None, description="Optional custom prompt for report generation")

class SourcesRequest(BaseModel):
    research_id: str = Field(..., description="The ID of the research session")

class ContextRequest(BaseModel):
    research_id: str = Field(..., description="The ID of the research session")

class ResearchQueryRequest(BaseModel):
    topic: str = Field(..., description="The topic to research")
    goal: str = Field(..., description="The goal or specific question to answer")
    report_format: str = Field("research_report", description="The format of the report to generate")

class Source(BaseModel):
    title: str
    url: str
    content: str

class ResearchResponse(BaseModel):
    status: str = "success"
    data: Dict[str, Any]

class ErrorResponse(BaseModel):
    status: str = "error"
    error: str

# Initialize FastAPI app
app = FastAPI(
    title="GPT Researcher API",
    description="API for conducting web research and generating reports using GPT Researcher",
    version="1.0.0",
)

# Store researchers dictionary as state
app.state.researchers = {}

# Resource endpoint (this will be exposed via MCP resources)
@app.get(
    "/api/research/resource/{topic}",
    response_model=str,
    operation_id="research_resource",
    tags=["Research"],
    description="Provide research context for a given topic directly as a resource"
)
async def research_resource(topic: str) -> str:
    """
    Provide research context for a given topic directly as a resource.

    This allows LLMs to access web-sourced information without explicit function calls.

    Args:
        topic: The research topic or query

    Returns:
        String containing the research context with source information
    """
    # Check if we've already researched this topic
    if topic in research_store:
        logger.info(f"Returning cached research for topic: {topic}")
        return research_store[topic]["context"]

    # If not, conduct the research
    logger.info(f"Conducting new research for resource on topic: {topic}")

    # Initialize GPT Researcher
    researcher = GPTResearcher(topic)

    try:
        # Conduct the research
        await researcher.conduct_research()

        # Get the context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()

        # Format with sources included
        formatted_context = format_context_with_sources(topic, context, sources)

        # Store for future use
        store_research_results(topic, context, sources, source_urls, formatted_context)

        return formatted_context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conducting research on '{topic}': {str(e)}")

@app.post(
    "/api/research/deep",
    response_model=ResearchResponse,
    operation_id="deep_research",
    tags=["Research"],
    description="Conduct a deep web research on a given query using GPT Researcher"
)
async def deep_research(request: ResearchRequest) -> Dict[str, Any]:
    """
    Conduct a deep web research on a given query using GPT Researcher.
    Use this API when you need time-sensitive, real-time information like stock prices, news, people, specific knowledge, etc.
    Results include citations that back responses.

    Args:
        request: The research request containing the query

    Returns:
        Dict containing research status, ID, and the actual research context and sources
    """
    query = request.query
    logger.info(f"Conducting research on query: {query}...")

    # Generate a unique ID for this research session
    research_id = str(uuid.uuid4())

    # Initialize GPT Researcher
    researcher = GPTResearcher(query)

    # Start research
    try:
        await researcher.conduct_research()
        app.state.researchers[research_id] = researcher
        logger.info(f"Research completed for ID: {research_id}")

        # Get the research context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()

        # Store in the research store for the resource API
        store_research_results(query, context, sources, source_urls)

        return ResearchResponse(data={
            "research_id": research_id,
            "query": query,
            "source_count": len(sources),
            "context": context,
            "sources": format_sources_for_response(sources),
            "source_urls": source_urls
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conducting research: {str(e)}")

@app.post(
    "/api/research/report",
    response_model=ResearchResponse,
    operation_id="write_report",
    tags=["Research"],
    description="Generate a report based on previously conducted research"
)
async def write_report(request: ReportRequest) -> Dict[str, Any]:
    """
    Generate a report based on previously conducted research.

    Args:
        request: The report request containing research_id and optional custom_prompt

    Returns:
        Dict containing the report content and metadata
    """
    research_id = request.research_id
    custom_prompt = request.custom_prompt

    if research_id not in app.state.researchers:
        raise HTTPException(status_code=404, detail=f"Research ID not found: {research_id}")

    researcher = app.state.researchers[research_id]
    logger.info(f"Generating report for research ID: {research_id}")

    try:
        # Generate report
        report = await researcher.write_report(custom_prompt=custom_prompt)

        # Get additional information
        sources = researcher.get_research_sources()
        costs = researcher.get_costs()

        return ResearchResponse(data={
            "report": report,
            "source_count": len(sources),
            "costs": costs
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.post(
    "/api/research/sources",
    response_model=ResearchResponse,
    operation_id="get_research_sources",
    tags=["Research"],
    description="Get the sources used in the research"
)
async def get_research_sources(request: SourcesRequest) -> Dict[str, Any]:
    """
    Get the sources used in the research.

    Args:
        request: The sources request containing research_id

    Returns:
        Dict containing the research sources
    """
    research_id = request.research_id

    if research_id not in app.state.researchers:
        raise HTTPException(status_code=404, detail=f"Research ID not found: {research_id}")

    researcher = app.state.researchers[research_id]
    sources = researcher.get_research_sources()
    source_urls = researcher.get_source_urls()

    return ResearchResponse(data={
        "sources": format_sources_for_response(sources),
        "source_urls": source_urls
    })

@app.post(
    "/api/research/context",
    response_model=ResearchResponse,
    operation_id="get_research_context",
    tags=["Research"],
    description="Get the full context of the research"
)
async def get_research_context(request: ContextRequest) -> Dict[str, Any]:
    """
    Get the full context of the research.

    Args:
        request: The context request containing research_id

    Returns:
        Dict containing the research context
    """
    research_id = request.research_id

    if research_id not in app.state.researchers:
        raise HTTPException(status_code=404, detail=f"Research ID not found: {research_id}")

    researcher = app.state.researchers[research_id]
    context = researcher.get_research_context()

    return ResearchResponse(data={
        "context": context
    })

@app.post(
    "/api/research/query",
    response_model=str,
    operation_id="research_query",
    tags=["Research"],
    description="Create a research query prompt for GPT Researcher"
)
async def research_query(request: ResearchQueryRequest) -> str:
    """
    Create a research query prompt for GPT Researcher.

    Args:
        request: The query request with topic, goal, and optional report_format

    Returns:
        A formatted prompt for research
    """
    return create_research_prompt(request.topic, request.goal, request.report_format)

# Add an MCP endpoint for resource access
@app.get(
    "/api/research/resource_string/{topic}",
    response_model=str,
    operation_id="research_resource_string",
    tags=["Research"],
    description="Get research as plain text for a given topic"
)
async def research_resource_string(topic: str) -> str:
    """Helper endpoint that exposes the research resource as a standard API endpoint"""
    return await research_resource(topic)

def create_mcp_server():
    """Create and configure the FastAPI MCP server"""
    # Create the MCP server from our FastAPI app with minimal required parameters
    mcp = FastApiMCP(
        app,
        name="GPT Researcher MCP",
        description="MCP server for GPT Researcher, allowing AI assistants to conduct web research"
    )

    # Mount the MCP server to the FastAPI app
    # This automatically discovers all FastAPI routes and registers them as MCP tools
    mcp.mount()

    return mcp

def run_server():
    """Run the FastAPI server using Uvicorn."""
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    # Create the MCP server
    mcp = create_mcp_server()

    # Add startup message
    logger.info("Starting GPT Researcher FastAPI + MCP Server...")
    print("🚀 GPT Researcher FastAPI + MCP Server starting... Check logs for details")

    # Run the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()