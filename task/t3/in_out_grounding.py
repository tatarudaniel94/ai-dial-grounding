import asyncio
from typing import Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient


# Pydantic models for structured output
class HobbyMatch(BaseModel):
    hobby: str = Field(description="The hobby/interest identified from the search")
    user_ids: list[int] = Field(description="List of user IDs who have this hobby")


class HobbySearchResult(BaseModel):
    matches: list[HobbyMatch] = Field(default_factory=list, description="List of hobby matches with user IDs")


SYSTEM_PROMPT = """You are a hobby extraction assistant. Your task is to analyze user profiles and identify which users match the requested hobbies/interests.

## Instructions:
1. Analyze the user's search query to understand what hobbies/interests they're looking for
2. Review the provided user profiles (containing id and about_me sections)
3. Identify users whose about_me section mentions relevant hobbies or interests
4. Group users by the specific hobby that matched
5. Return ONLY user IDs - do not include any personal information

## Important:
- Be inclusive: if someone mentions related activities, include them
- One user can appear in multiple hobby categories
- Only include users whose about_me clearly relates to the searched interest
- Return empty matches if no users match

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## SEARCH REQUEST:
{query}

## USER PROFILES:
{context}

Find users matching the requested hobbies/interests and group them by hobby."""


def format_user_for_embedding(user: dict[str, Any]) -> str:
    """Format user id and about_me for embedding (reduced context)."""
    return f"User ID: {user['id']}\nAbout: {user.get('about_me', 'No description')}"


class HobbiesWizard:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.user_client = UserClient()
        self.vectorstore: Chroma | None = None
        self.known_user_ids: set[int] = set()

    async def __aenter__(self):
        print("🔮 Initializing Hobbies Wizard...")
        await self._cold_start()
        print("✅ Wizard is ready!")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _cold_start(self):
        """Initialize vectorstore with all current users."""
        print("🔎 Loading all users for cold start...")
        users = self.user_client.get_all_users()
        
        documents = []
        for user in users:
            doc = Document(
                page_content=format_user_for_embedding(user),
                id=str(user['id'])
            )
            documents.append(doc)
            self.known_user_ids.add(user['id'])
        
        print(f"📦 Creating vectorstore with {len(documents)} users...")
        self.vectorstore = await Chroma.afrom_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="users_hobbies"
        )

    async def _sync_vectorstore(self):
        """Sync vectorstore with current users (add new, remove deleted)."""
        print("🔄 Syncing vectorstore...")
        current_users = self.user_client.get_all_users()
        current_ids = {user['id'] for user in current_users}
        
        # Find new and deleted users
        new_ids = current_ids - self.known_user_ids
        deleted_ids = self.known_user_ids - current_ids
        
        # Remove deleted users from vectorstore
        if deleted_ids:
            print(f"🗑️ Removing {len(deleted_ids)} deleted users...")
            await self.vectorstore.adelete(ids=[str(uid) for uid in deleted_ids])
            self.known_user_ids -= deleted_ids
        
        # Add new users to vectorstore
        if new_ids:
            print(f"➕ Adding {len(new_ids)} new users...")
            new_users = [u for u in current_users if u['id'] in new_ids]
            new_docs = [
                Document(
                    page_content=format_user_for_embedding(user),
                    id=str(user['id'])
                )
                for user in new_users
            ]
            await self.vectorstore.aadd_documents(new_docs)
            self.known_user_ids |= new_ids
        
        if not new_ids and not deleted_ids:
            print("✓ Vectorstore is up to date")

    async def search_hobbies(self, query: str, k: int = 50) -> dict[str, list[dict[str, Any]]]:
        """Search for users by hobbies and return grouped results."""
        # Step 1: Sync vectorstore with latest data
        await self._sync_vectorstore()
        
        # Step 2: Retrieve relevant user profiles
        print(f"🔍 Searching for: {query}")
        results = await self.vectorstore.asimilarity_search(query, k=k)
        
        if not results:
            print("No relevant users found")
            return {}
        
        # Step 3: Format context for LLM
        context = "\n\n".join([doc.page_content for doc in results])
        print(f"📋 Found {len(results)} potentially relevant users")
        
        # Step 4: Extract hobbies and user IDs using LLM
        parser = PydanticOutputParser(pydantic_object=HobbySearchResult)
        
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(query=query, context=context)),
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
            format_instructions=parser.get_format_instructions()
        )
        
        search_result: HobbySearchResult = (prompt | self.llm_client | parser).invoke({})
        
        # Step 5: Output grounding - fetch full user info and verify IDs exist
        print("🔒 Grounding output - fetching user details...")
        grouped_results: dict[str, list[dict[str, Any]]] = {}
        
        for match in search_result.matches:
            hobby = match.hobby
            valid_users = []
            
            for user_id in match.user_ids:
                try:
                    user_data = await self.user_client.get_user(user_id)
                    valid_users.append(user_data)
                except Exception as e:
                    print(f"⚠️ User {user_id} not found (possibly deleted): {e}")
            
            if valid_users:
                grouped_results[hobby] = valid_users
        
        return grouped_results


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

    async with HobbiesWizard(embeddings, llm_client) as wizard:
        print("\n🎯 HOBBIES SEARCHING WIZARD")
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find users interested in photography and art")
        print(" - Who likes cooking or baking?")
        
        while True:
            user_question = input("\n> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            
            results = await wizard.search_hobbies(user_question)
            
            if results:
                print("\n🎉 Results:")
                import json
                print(json.dumps(results, indent=2, default=str))
            else:
                print("\n❌ No users found matching the requested hobbies")


if __name__ == "__main__":
    asyncio.run(main())
