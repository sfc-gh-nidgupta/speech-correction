"""
Agent Manager Module

Manages multiple domain-specific agents, each with:
- A semantic terms YAML file for speech correction
- A knowledge base Markdown file for Q&A
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from cerebras.cloud.sdk import Cerebras


AGENTS_DIR = Path(__file__).parent / "agents"


@dataclass
class AgentConfig:
    """Configuration for a domain-specific agent."""
    name: str
    description: str
    icon: str
    terms_file: str
    knowledge_file: str
    folder: Path
    
    @property
    def terms_path(self) -> Path:
        return self.folder / self.terms_file
    
    @property
    def knowledge_path(self) -> Path:
        return self.folder / self.knowledge_file
    
    def load_terms(self) -> str:
        """Load the semantic terms YAML as a string."""
        if self.terms_path.exists():
            return self.terms_path.read_text(encoding="utf-8")
        return ""
    
    def load_knowledge(self) -> str:
        """Load the knowledge base Markdown."""
        if self.knowledge_path.exists():
            return self.knowledge_path.read_text(encoding="utf-8")
        return ""


class AgentManager:
    """Manages loading and accessing domain-specific agents."""
    
    def __init__(self, agents_dir: Path = AGENTS_DIR):
        self.agents_dir = agents_dir
        self._agents: dict[str, AgentConfig] = {}
        self._load_agents()
    
    def _load_agents(self):
        """Scan agents directory and load all agent configs."""
        if not self.agents_dir.exists():
            self.agents_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for folder in self.agents_dir.iterdir():
            if folder.is_dir():
                config_path = folder / "config.yaml"
                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            config_data = yaml.safe_load(f)
                        
                        agent = AgentConfig(
                            name=config_data.get("name", folder.name),
                            description=config_data.get("description", ""),
                            icon=config_data.get("icon", "🤖"),
                            terms_file=config_data.get("terms_file", "terms.yaml"),
                            knowledge_file=config_data.get("knowledge_file", "knowledge.md"),
                            folder=folder
                        )
                        self._agents[folder.name] = agent
                    except Exception as e:
                        print(f"Error loading agent {folder.name}: {e}")
    
    def list_agents(self) -> list[AgentConfig]:
        """Return list of all available agents."""
        return list(self._agents.values())
    
    def get_agent(self, agent_id: str) -> AgentConfig | None:
        """Get agent by folder name/ID."""
        return self._agents.get(agent_id)
    
    def get_agent_names(self) -> list[tuple[str, str, str]]:
        """Return list of (id, name, icon) tuples for dropdown."""
        return [(agent_id, agent.name, agent.icon) 
                for agent_id, agent in self._agents.items()]
    
    def create_agent(self, agent_id: str, name: str, description: str, 
                     icon: str, terms_content: str, knowledge_content: str) -> AgentConfig:
        """Create a new agent with the given configuration."""
        # Create agent folder
        agent_folder = self.agents_dir / agent_id
        agent_folder.mkdir(parents=True, exist_ok=True)
        
        # Write config
        config_data = {
            "name": name,
            "description": description,
            "icon": icon,
            "terms_file": "terms.yaml",
            "knowledge_file": "knowledge.md"
        }
        with open(agent_folder / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Write terms
        with open(agent_folder / "terms.yaml", "w") as f:
            f.write(terms_content)
        
        # Write knowledge
        with open(agent_folder / "knowledge.md", "w") as f:
            f.write(knowledge_content)
        
        # Create and register agent
        agent = AgentConfig(
            name=name,
            description=description,
            icon=icon,
            terms_file="terms.yaml",
            knowledge_file="knowledge.md",
            folder=agent_folder
        )
        self._agents[agent_id] = agent
        
        return agent
    
    def refresh(self):
        """Reload all agents from disk."""
        self._agents.clear()
        self._load_agents()


class DomainAgent:
    """
    An LLM-powered agent that uses domain-specific terms and knowledge.
    """
    
    def __init__(self, config: AgentConfig, api_key: str | None = None):
        self.config = config
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("Cerebras API key is required")
        
        self.client = Cerebras(api_key=self.api_key)
        self.model = "llama-3.3-70b"
        self._knowledge = config.load_knowledge()
        self._terms = config.load_terms()
    
    def get_correction_prompt(self) -> str:
        """Generate the correction system prompt using domain terms."""
        return f"""You are a conservative terminology corrector for {self.config.name}.

IMPORTANT RULES:
1. BE CONSERVATIVE - Only correct when you are highly confident
2. DO NOT change the meaning or intent of the question
3. DO NOT invent or assume what the user meant to ask
4. If uncertain, return the input AS-IS with only minor spelling fixes
5. Preserve the original question structure

KNOWN DOMAIN TERMS (from semantic model):
{self._terms}

Return ONLY the corrected text. If unsure, return the original with minimal changes."""
    
    def correct_transcript(self, raw_text: str) -> str:
        """Correct transcript using domain-specific terms."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_correction_prompt()},
                    {"role": "user", "content": f"Correct this transcript (be conservative): {raw_text}"},
                ],
            )
            if response.choices and response.choices[0].message:
                result = response.choices[0].message.content.strip()
                # Safety check
                if len(result) > len(raw_text) * 3 or len(result) < len(raw_text) * 0.3:
                    return raw_text
                return result
            return raw_text
        except:
            return raw_text
    
    def answer(self, question: str) -> str:
        """Answer a question using the domain knowledge base."""
        system_prompt = f"""You are a helpful {self.config.name} expert assistant.

Answer questions based on this knowledge base:

---
{self._knowledge}
---

Instructions:
1. Answer based on the knowledge base above
2. If not covered, provide your best knowledge but mention it may not be in the official docs
3. Be conversational and helpful
4. Mention related concepts when relevant"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            )
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content.strip()
            return "I couldn't generate a response. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"


# Singleton instance
_manager: AgentManager | None = None

def get_agent_manager() -> AgentManager:
    """Get the singleton agent manager instance."""
    global _manager
    if _manager is None:
        _manager = AgentManager()
    return _manager

