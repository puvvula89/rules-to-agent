# Phase 3: Agent Engine deployment wrapper
# Deploy with:
#   vertexai.init(project="tmeg-working-demos", location="us-central1")
#   reasoning_engines.create(adk_app, requirements=[...])

from vertexai.preview.reasoning_engines import AdkApp
from agents.manager import root_agent

adk_app = AdkApp(agent=root_agent)
