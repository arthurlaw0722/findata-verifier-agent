import asyncio
import os
import json
import tempfile
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from croo import AgentClient, Config, EventType, DeliverableType, DeliverOrderRequest

from agent.analyzer import analyse_dataset
from agent.leakage import detect_possible_leakage
from agent.scoring import calculate_trust_score
from agent.report_generator import generate_markdown_report
from agent.proof import create_proof
from agent.readiness import assess_readiness


client = AgentClient(
    Config(
        base_url=os.environ["CROO_API_URL"],
        ws_url=os.environ["CROO_WS_URL"],
    ),
    os.environ["CROO_SDK_KEY"]
)


async def run_verification_job(requirements_text: str) -> str:
    requirements = json.loads(requirements_text)

    csv_path = requirements["csv_path"]
    dataset_name = requirements.get("dataset_name", "Unknown Dataset")
    target_column = requirements.get("target_column")

    df = pd.read_csv(csv_path)

    analysis = analyse_dataset(csv_path, target_column)
    leakage = detect_possible_leakage(df, target_column)
    score = calculate_trust_score(analysis, leakage)
    readiness = assess_readiness(analysis, leakage, score)

    temp_report = generate_markdown_report(
        dataset_name,
        analysis,
        leakage,
        score,
        {
            "dataset_fingerprint": "pending",
            "report_hash": "pending",
            "execution_timestamp": "pending"
        }
    )

    proof = create_proof(csv_path, temp_report)

    final_report = generate_markdown_report(
        dataset_name,
        analysis,
        leakage,
        score,
        proof
    )

    deliverable = {
        "agent": "FinData Verifier Agent",
        "dataset_name": dataset_name,
        "trust_score": score["trust_score"],
        "trust_grade": score["trust_grade"],
        "leakage": leakage,
        "readiness": readiness,
        "report_markdown": final_report,
        "proof": proof
    }

    return json.dumps(deliverable, indent=2)


async def main():
    stream = await client.connect_websocket()

    def on_negotiation(e):
        async def _handle():
            result = await client.accept_negotiation(e.negotiation_id)
            print(f"Order created: {result.order.order_id}")
        asyncio.create_task(_handle())

    def on_paid(e):
        async def _handle():
            order = await client.get_order(e.order_id)
            result_text = await run_verification_job(order.requirements)

            await client.deliver_order(
                e.order_id,
                DeliverOrderRequest(
                    deliverable_type=DeliverableType.TEXT,
                    deliverable_text=result_text
                )
            )

            print(f"Delivered verification report for order: {e.order_id}")

        asyncio.create_task(_handle())

    stream.on(EventType.NEGOTIATION_CREATED, on_negotiation)
    stream.on(EventType.ORDER_PAID, on_paid)

    print("FinData Verifier Agent provider is running...")
    stop = asyncio.Event()
    await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
