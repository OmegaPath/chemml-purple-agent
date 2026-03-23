import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the ChemML Purple Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill_ml_engineering = AgentSkill(
        id="ml-engineering",
        name="Machine Learning Engineering",
        description=(
            "Solves Kaggle-style ML competitions end-to-end. "
            "Analyzes datasets, engineers features, trains models, "
            "and produces submission files. Handles tabular, image, "
            "text, and time-series data."
        ),
        tags=["machine-learning", "data-science", "kaggle", "ml-engineering"],
        examples=[
            "Train a model on the Titanic dataset and predict survival",
            "Build an image classifier for the given training images",
            "Predict house prices from the provided feature set",
        ],
    )

    skill_cheminformatics = AgentSkill(
        id="cheminformatics",
        name="Cheminformatics & Chemometrics",
        description=(
            "Specialized in chemical data analysis. "
            "Converts molecular representations (SMILES, InChI), "
            "computes molecular fingerprints, predicts molecular properties, "
            "and performs chemometric analysis on spectral/analytical data."
        ),
        tags=["chemistry", "cheminformatics", "chemometrics", "rdkit", "molecular"],
        examples=[
            "Translate molecular structure images to InChI strings",
            "Predict toxicity from SMILES strings using Morgan fingerprints",
            "Analyze spectral data to identify unknown compounds",
        ],
    )

    agent_card = AgentCard(
        name="ChemML Purple Agent",
        description=(
            "A competition-grade ML engineering agent with deep expertise in "
            "analytical chemistry and cheminformatics. Solves Kaggle-style "
            "ML competitions end-to-end: analyzes data, generates code, "
            "trains models, and produces submission files. Specializes in "
            "chemical data (SMILES, InChI, fingerprints, spectral data) "
            "while excelling at general-purpose ML tasks."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill_ml_engineering, skill_cheminformatics],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None,  # Allow large file uploads (competition.tar.gz)
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
