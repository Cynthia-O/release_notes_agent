#!/usr/bin/env python3
"""
Databricks AWS Release Notes Agent

This script creates an AI agent that can answer questions about Databricks AWS release notes,
focusing on recent features and capabilities including cost optimization, AI/BI dashboarding,
vector database handling, agentic work, and external data access features.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import re
from typing import Any, Optional, Sequence, Union, List, Dict

# Databricks and MLflow imports
from databricks_langchain import ChatDatabricks
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
import mlflow
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Enable MLflow autologging
mlflow.langchain.autolog()


def scrape_release_notes() -> pd.DataFrame:
    """
    Scrapes release notes from Databricks AWS documentation.
    Returns a DataFrame with release note content and metadata.
    
    For demo purposes, creates synthetic data. In production, implement actual web scraping.
    """
    
    # Base URL for Databricks AWS release notes
    base_url = "https://docs.databricks.com/aws/en/release-notes/product/"
    
    # Synthetic release notes data for demonstration
    synthetic_release_notes = [
        {
            "content": "Databricks Runtime 15.3 LTS introduces enhanced cost optimization features including automatic cluster scaling based on workload patterns, improved spot instance utilization, and real-time cost monitoring dashboards. These features help reduce compute costs by up to 40% while maintaining performance.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/runtime/15.3.html",
            "release_date": "2024-12-15",
            "category": "cost_optimization"
        },
        {
            "content": "New AI-powered dashboard builder allows users to create interactive BI dashboards using natural language queries. The feature includes automatic chart selection, smart data aggregation, and real-time collaboration capabilities. Supports integration with Unity Catalog for secure data access.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/dashboards/ai-builder.html",
            "release_date": "2024-12-10",
            "category": "ai_bi_dashboarding"
        },
        {
            "content": "Vector Search capabilities enhanced with support for hybrid search combining dense vector embeddings with traditional keyword search. New features include automatic embedding generation, similarity threshold controls, and integration with popular vector databases like Pinecone and Weaviate.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/vector-search/enhancements.html",
            "release_date": "2024-12-08",
            "category": "vector_database"
        },
        {
            "content": "Agent Framework now supports multi-step reasoning with memory persistence across sessions. New capabilities include tool chaining, conditional logic execution, and integration with external APIs. Agents can now maintain context and learn from previous interactions.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/agents/multi-step.html",
            "release_date": "2024-12-05",
            "category": "agentic_work"
        },
        {
            "content": "External data access enhanced with new connectors for Snowflake, BigQuery, and Redshift. Features include automatic schema inference, incremental data loading, and unified governance through Unity Catalog. Performance optimizations reduce query times by up to 60%.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/external-data/connectors.html",
            "release_date": "2024-12-01",
            "category": "external_data_access"
        },
        {
            "content": "Cost control features expanded with budget alerts, resource tagging automation, and predictive cost forecasting. New dashboard provides granular cost breakdown by workspace, cluster, and user. Integration with AWS Cost Explorer for comprehensive cost management.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/cost-control/enhanced.html",
            "release_date": "2024-11-28",
            "category": "cost_optimization"
        },
        {
            "content": "AI/ML capabilities enhanced with new AutoML features for time series forecasting and anomaly detection. Improved model registry with version control, A/B testing framework, and automated model deployment pipelines. Support for custom model serving with GPU acceleration.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/ml/automl-enhancements.html",
            "release_date": "2024-11-25",
            "category": "ai_bi_dashboarding"
        },
        {
            "content": "Vector database performance improvements with new indexing algorithms and parallel processing capabilities. Support for real-time vector updates and streaming ingestion. Enhanced query optimization for large-scale vector similarity search operations.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/vector-search/performance.html",
            "release_date": "2024-11-20",
            "category": "vector_database"
        },
        {
            "content": "Agent development tools enhanced with visual workflow builder and debugging capabilities. New agent marketplace for sharing and discovering pre-built agents. Improved monitoring and observability for agent performance and reliability.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/agents/development-tools.html",
            "release_date": "2024-11-18",
            "category": "agentic_work"
        },
        {
            "content": "External data federation capabilities expanded with support for Delta Lake format across multiple cloud providers. New data virtualization features enable querying external data sources without data movement. Enhanced security with row-level security and column-level encryption.",
            "doc_uri": "https://docs.databricks.com/aws/en/release-notes/external-data/federation.html",
            "release_date": "2024-11-15",
            "category": "external_data_access"
        }
    ]
    
    return pd.DataFrame(synthetic_release_notes)


def extract_release_notes_keywords(text: str) -> List[str]:
    """
    Extracts keywords from release notes text, focusing on technical features and capabilities.
    
    Args:
        text (str): Input text from release notes.
    Returns:
        List[str]: List of extracted keywords in order of importance.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import re
    
    # Clean and preprocess text
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    
    def extract_keywords(text, top_n=8):
        """Extracts top keywords from release notes text"""
        # Custom stop words for release notes context
        custom_stop_words = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'new', 'enhanced', 'improved', 'updated', 'added', 'support', 'feature', 'capability'
        ]
        
        vectorizer = TfidfVectorizer(
            stop_words=custom_stop_words,
            ngram_range=(1, 2),  # Include bigrams for technical terms
            max_features=100
        )
        
        tfidf = vectorizer.fit_transform([text])
        scores = tfidf.toarray()[0]
        indices = scores.argsort()[-top_n:][::-1]
        
        return [vectorizer.get_feature_names_out()[i] for i in indices if scores[i] > 0]
    
    return extract_keywords(text)


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    agent_prompt: Optional[str] = None,
) -> CompiledGraph:
    """
    Creates a tool-calling agent using LangGraph.
    
    Args:
        model: The language model to use
        tools: List of tools the agent can use
        agent_prompt: Optional system prompt for the agent
    
    Returns:
        CompiledGraph: The compiled agent workflow
    """
    model = model.bind_tools(tools)

    def routing_logic(state: ChatAgentState):
        last_message = state["messages"][-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if agent_prompt:
        system_message = {"role": "system", "content": agent_prompt}
        preprocessor = RunnableLambda(
            lambda state: [system_message] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}
    
    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        routing_logic,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class ReleaseNotesAgent(ChatAgent):
    """
    ChatAgent wrapper for the release notes agent.
    """
    
    def __init__(self, agent):
        self.agent = agent

    def predict(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}
        output = self.agent.invoke(request)
        return ChatAgentResponse(**output)


def setup_release_notes_agent(
    llm_endpoint: str = "llama-instruct-3-3-70b",
    catalog: str = "main",
    schema: str = "default"
) -> ReleaseNotesAgent:
    """
    Sets up and returns a configured release notes agent.
    
    Args:
        llm_endpoint: The LLM endpoint to use
        catalog: Unity Catalog catalog name
        schema: Unity Catalog schema name
    
    Returns:
        ReleaseNotesAgent: Configured agent ready for use
    """
    
    # Configure LLM
    llm = ChatDatabricks(endpoint=llm_endpoint)
    
    # Load release notes data
    release_notes_df = scrape_release_notes()
    print(f"Loaded {len(release_notes_df)} release notes entries")
    
    # Set up Unity Catalog client
    uc_client = DatabricksFunctionClient()
    set_uc_function_client(uc_client)
    
    # Create keyword extraction function in Unity Catalog
    function_info = uc_client.create_python_function(
        func=extract_release_notes_keywords,
        catalog=catalog,
        schema=schema,
        replace=True,
    )
    print(f"Created function: {function_info}")
    
    # Create toolkit
    uc_tool_names = [f"{catalog}.{schema}.extract_release_notes_keywords"]
    uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
    
    # Prepare documents for search
    documents = release_notes_df
    doc_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = doc_vectorizer.fit_transform(documents["content"])
    
    # Create document retrieval tool
    @tool
    @mlflow.trace(name="ReleaseNotesRetriever", span_type=mlflow.entities.SpanType.RETRIEVER)
    def find_relevant_release_notes(query: str, top_n: int = 5, category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant release notes based on the query.
        
        Args:
            query (str): The search query
            top_n (int): Number of results to return (default: 5)
            category_filter (str): Optional category filter
        
        Returns:
            List[Dict]: List of relevant release notes with metadata
        """
        
        # Filter by category if specified
        if category_filter:
            filtered_docs = documents[documents['category'] == category_filter]
            if len(filtered_docs) == 0:
                return []
            
            # Rebuild TF-IDF matrix for filtered documents
            filtered_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            filtered_tfidf_matrix = filtered_vectorizer.fit_transform(filtered_docs["content"])
            query_tfidf = filtered_vectorizer.transform([query])
            similarities = (filtered_tfidf_matrix @ query_tfidf.T).toarray().flatten()
            ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
            
            result = []
            for idx, score in ranked_docs[:top_n]:
                row = filtered_docs.iloc[idx]
                result.append({
                    "page_content": row["content"],
                    "metadata": {
                        "doc_uri": row["doc_uri"],
                        "release_date": row["release_date"],
                        "category": row["category"],
                        "score": float(score),
                    },
                })
            return result
        
        # Search across all documents
        query_tfidf = doc_vectorizer.transform([query])
        similarities = (tfidf_matrix @ query_tfidf.T).toarray().flatten()
        ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

        result = []
        for idx, score in ranked_docs[:top_n]:
            row = documents.iloc[idx]
            result.append({
                "page_content": row["content"],
                "metadata": {
                    "doc_uri": row["doc_uri"],
                    "release_date": row["release_date"],
                    "category": row["category"],
                    "score": float(score),
                },
            })
        return result
    
    # Create specialized system prompt
    release_notes_system_prompt = """
    You are a specialized assistant that answers questions about Databricks AWS release notes and recent features. 
    You have access to detailed information about recent Databricks capabilities including:

    - Cost optimization and control features
    - AI/BI dashboarding capabilities  
    - Vector database and search features
    - Agentic work and automation capabilities
    - External data access and federation features

    When answering questions:
    1. Focus on recent features and capabilities (last 3-6 months)
    2. Provide specific details about what was released and when
    3. Include performance improvements and benefits where available
    4. Reference the specific release notes and documentation links
    5. If asked about trends, analyze which capabilities Databricks has been building most rapidly

    You have access to tools that can search through release notes and extract relevant keywords. 
    Use these tools to provide accurate, up-to-date information about Databricks AWS features.
    """
    
    # Create the agent
    agent = create_tool_calling_agent(
        llm, 
        tools=[*uc_toolkit.tools, find_relevant_release_notes],
        agent_prompt=release_notes_system_prompt
    )
    
    # Create and return the ChatAgent wrapper
    return ReleaseNotesAgent(agent=agent)


def test_agent(agent: ReleaseNotesAgent):
    """
    Test the agent with sample questions.
    
    Args:
        agent: The configured release notes agent
    """
    test_questions = [
        "What are the most recent features that Databricks on AWS has released to control costs?",
        "What capabilities, such as external data access, or AI/BI dashboarding, or vector database handling, or agentic work, has Databricks been building out most rapidly over the last three months?",
        "What are the latest vector database features in Databricks?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        response = agent.predict({
            "messages": [{"role": "user", "content": question}]
        })
        print(f"\nAnswer: {response.content}")
        print(f"\n{'-'*80}")


def main():
    """
    Main function to set up and test the release notes agent.
    """
    print("Setting up Databricks AWS Release Notes Agent...")
    
    # Set up the agent
    agent = setup_release_notes_agent()
    
    print("Agent setup complete! Testing with sample questions...")
    
    # Test the agent
    test_agent(agent)
    
    print("\nRelease Notes Agent is ready for use!")


if __name__ == "__main__":
    main() 