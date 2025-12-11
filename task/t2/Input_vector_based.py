import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about users based on provided context.

INSTRUCTIONS:
1. Use ONLY the information from the provided context to answer questions
2. If the context doesn't contain relevant information, say so clearly
3. Be precise and include specific details from the context when available
4. Format your response in a clear, readable manner"""

USER_PROMPT = """## CONTEXT:
{context}

## QUESTION:
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    user_lines = ["User:"]
    for key, value in user.items():
        user_lines.append(f"  {key}: {value}")
    return "\n".join(user_lines)


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        client = UserClient()
        users = client.get_all_users()
        documents = [Document(page_content=format_user_document(user)) for user in users]
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        print(f"📦 Processing {len(documents)} documents in {len(batches)} batches...")
        
        tasks = [
            FAISS.afrom_documents(batch, self.embeddings)
            for batch in batches
        ]
        
        vectorstores = await asyncio.gather(*tasks)
        
        final_vectorstore = vectorstores[0]
        for vs in vectorstores[1:]:
            final_vectorstore.merge_from(vs)
        
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        results = await self.vectorstore.asimilarity_search_with_relevance_scores(query, k=k)
        
        context_parts = []
        
        print(f"\n📋 Retrieved {len(results)} relevant documents:")
        for doc, relevance_score in results:
            if relevance_score >= score:
                context_parts.append(doc.page_content)
                print(f"  Score: {relevance_score:.4f} | {doc.page_content[:50]}...")
        
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        return response.content


async def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model="text-embedding-3-small-1",
        dimensions=384,
        api_version="",
    )
    
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model="gpt-4o-mini",
        api_version="",
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            
            context = await rag.retrieve_context(user_question)
            augmented_prompt = rag.augment_prompt(user_question, context)
            answer = rag.generate_answer(augmented_prompt)
            print(f"\n🤖 Answer:\n{answer}\n")


asyncio.run(main())
