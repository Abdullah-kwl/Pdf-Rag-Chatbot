"""LangGraph RAG pipeline: condense the question, retrieve context, generate the answer."""

from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import config

CONDENSE_PROMPT = """Rewrite the user's latest message as a standalone search query.
Resolve pronouns and references using the conversation above.
Return the query only, with no preamble."""

ANSWER_PROMPT = """You are a document question-answering assistant.

Answer strictly from the context below. If the context does not contain the answer,
say that it is not covered in the uploaded documents. Do not use outside knowledge.
Cite every supporting passage inline as [file, p.N].

Context:
{context}"""


class RAGState(TypedDict):
    """State carried through the RAG graph."""

    messages: Annotated[list, add_messages]
    question: str
    documents: List[Document]


def format_context(documents):
    """Render retrieved documents as a numbered, source-tagged context block."""
    if not documents:
        return "No relevant passages were retrieved."

    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{i}] [{source}, p.{page}]\n{doc.page_content}")

    return "\n\n".join(blocks)


def build_graph(retriever):
    """Compile the RAG graph against a retriever."""
    llm = ChatGroq(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    def condense(state: RAGState):
        """Turn a follow-up message into a self-contained query."""
        messages = state["messages"]
        latest = messages[-1].content
        history = messages[:-1][-config.HISTORY_WINDOW :]

        if not history:
            return {"question": latest}

        response = llm.invoke(
            [SystemMessage(content=CONDENSE_PROMPT), *history, HumanMessage(content=latest)]
        )
        return {"question": response.content.strip() or latest}

    def retrieve(state: RAGState):
        """Fetch the most relevant chunks for the condensed question."""
        return {"documents": retriever.invoke(state["question"])}

    def generate(state: RAGState):
        """Answer the question using the retrieved context."""
        messages = state["messages"]
        latest = messages[-1].content
        history = messages[:-1][-config.HISTORY_WINDOW :]
        context = format_context(state["documents"])

        response = llm.invoke(
            [
                SystemMessage(content=ANSWER_PROMPT.format(context=context)),
                *history,
                HumanMessage(content=latest),
            ]
        )
        return {"messages": [AIMessage(content=response.content)]}

    builder = StateGraph(RAGState)
    builder.add_node("condense", condense)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "condense")
    builder.add_edge("condense", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())
