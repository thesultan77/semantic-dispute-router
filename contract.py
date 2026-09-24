# v0.2.16
# {
#   "Seq": [
#     { "Depends": "py-lib-genlayer-embeddings:09h0i209wrzh4xzq86f79c60x0ifs7xcjwl53ysrnw06i54ddxyi" },
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
#   ]
# }
import numpy as np
from genlayer import *
import genlayer_embeddings as gle
import datetime
import typing
import json
from dataclasses import dataclass


@gl.evm.contract_interface
class _Recipient:
    class View:
        pass
    class Write:
        pass


@allow_storage
@dataclass
class Precedent:
    case_id: str
    dispute_text: str
    ruling: str
    reasoning: str


class SemanticDisputeRouter(gl.Contract):
    descriptions: TreeMap[str, str]
    party_a_map: TreeMap[str, Address]
    party_b_map: TreeMap[str, Address]
    status_map: TreeMap[str, str]
    ruling_map: TreeMap[str, str]
    reasoning_map: TreeMap[str, str]
    created_at_map: TreeMap[str, str]
    stakes_a: TreeMap[str, u256]
    stakes_b: TreeMap[str, u256]

    precedents: gle.VecDB[np.float32, typing.Literal[384], Precedent]
    dispute_counter: u256
    precedent_counter: u256

    def __init__(self):
        self.dispute_counter = u256(0)
        self.precedent_counter = u256(0)

    def get_embedding_generator(self):
        return gle.SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(
        self, txt: str
    ) -> np.ndarray[tuple[typing.Literal[384]], np.dtypes.Float32DType]:
        return self.get_embedding_generator()(txt)

    @gl.public.write.payable
    def open_dispute(self, description: str, party_b: Address) -> str:
        dispute_id = f"D{self.dispute_counter}"
        self.dispute_counter += u256(1)

        self.descriptions[dispute_id] = description
        self.party_a_map[dispute_id] = gl.message.sender_address
        self.party_b_map[dispute_id] = Address(party_b.to_bytes(20, byteorder="big"))
        self.status_map[dispute_id] = "OPEN"
        self.ruling_map[dispute_id] = ""
        self.reasoning_map[dispute_id] = ""
        self.created_at_map[dispute_id] = str(datetime.datetime.now())
        self.stakes_a[dispute_id] = gl.message.value
        self.stakes_b[dispute_id] = u256(0)

        return dispute_id

    @gl.public.write.payable
    def accept_dispute(self, dispute_id: str) -> None:
        assert self.status_map[dispute_id] == "OPEN", "Dispute not open"
        assert gl.message.sender_address == self.party_b_map[dispute_id], "Only party_b can accept"
        assert gl.message.value == self.stakes_a[dispute_id], "Stake must match party_a's stake"
        self.stakes_b[dispute_id] = gl.message.value
        self.status_map[dispute_id] = "ACCEPTED"

    @gl.public.view
    def find_similar_precedents(self, description: str, top_n: int) -> list[dict]:
        emb = self.get_embedding(description)
        out = []
        for r in self.precedents.knn(emb, top_n):
            out.append({
                "similarity": str(r.distance),
                "case_id": r.value.case_id,
                "ruling": r.value.ruling,
                "reasoning": r.value.reasoning,
            })
        return out

    @gl.public.write
    def resolve_dispute(self, dispute_id: str) -> str:
        assert self.status_map[dispute_id] == "ACCEPTED", "Dispute must be accepted by both parties first"

        description = self.descriptions[dispute_id]
        emb = self.get_embedding(description)
        matches = list(self.precedents.knn(emb, 3))
        if matches:
            lines = [
                f'- Case {m.value.case_id}: "{m.value.dispute_text}" -> Ruling: {m.value.ruling}. Reasoning: {m.value.reasoning}'
                for m in matches
            ]
            precedent_context = "Relevant precedent cases (rule consistently with these where applicable):\n" + "\n".join(lines)
        else:
            precedent_context = "No precedent cases exist yet. Use first-principles judgment."

        def judge() -> str:
            prompt = f"""You are an impartial dispute arbitrator.

Dispute description:
\"\"\"{description}\"\"\"

{precedent_context}

Rule on this dispute. Choose exactly one outcome:
- PARTY_A (party_a is right, receives full stake pool)
- PARTY_B (party_b is right, receives full stake pool)
- SPLIT (shared responsibility, stake pool split evenly)

Respond with strict JSON only, no markdown, no extra text:
{{"ruling": "PARTY_A", "reasoning": "<one sentence>"}}
"""
            return gl.nondet.exec_prompt(prompt)

        raw = gl.eq_principle.prompt_comparative(
            judge,
            "The ruling field must match exactly (PARTY_A, PARTY_B, or SPLIT). Reasoning wording may vary but must express the same underlying judgment.",
        )
        parsed = json.loads(raw)
        ruling = parsed["ruling"]
        reasoning = parsed["reasoning"]

        self.ruling_map[dispute_id] = ruling
        self.reasoning_map[dispute_id] = reasoning
        self.status_map[dispute_id] = "RESOLVED"

        party_a = self.party_a_map[dispute_id]
        party_b = self.party_b_map[dispute_id]
        stake_a = self.stakes_a[dispute_id]
        stake_b = self.stakes_b[dispute_id]
        total = stake_a + stake_b

        if ruling == "PARTY_A":
            _Recipient(party_a).emit_transfer(value=total)
        elif ruling == "PARTY_B":
            _Recipient(party_b).emit_transfer(value=total)
        else:
            half = total // u256(2)
            _Recipient(party_a).emit_transfer(value=half)
            _Recipient(party_b).emit_transfer(value=total - half)

        case_id = f"P{self.precedent_counter}"
        self.precedent_counter += u256(1)
        self.precedents.insert(emb, Precedent(
            case_id=case_id, dispute_text=description,
            ruling=ruling, reasoning=reasoning,
        ))

        return f"{ruling} - {reasoning}"

    @gl.public.write
    def cancel_dispute(self, dispute_id: str) -> None:
        assert self.status_map[dispute_id] == "OPEN", "Can only cancel before party_b accepts"
        assert gl.message.sender_address == self.party_a_map[dispute_id], "Only party_a can cancel"
        stake_a = self.stakes_a[dispute_id]
        if stake_a > u256(0):
            _Recipient(self.party_a_map[dispute_id]).emit_transfer(value=stake_a)
        self.status_map[dispute_id] = "CANCELLED"

    @gl.public.view
    def get_dispute(self, dispute_id: str) -> dict:
        return {
            "dispute_id": dispute_id,
            "description": self.descriptions[dispute_id],
            "party_a": str(self.party_a_map[dispute_id]),
            "party_b": str(self.party_b_map[dispute_id]),
            "stake_a": str(self.stakes_a[dispute_id]),
            "stake_b": str(self.stakes_b[dispute_id]),
            "status": self.status_map[dispute_id],
            "ruling": self.ruling_map[dispute_id],
            "reasoning": self.reasoning_map[dispute_id],
        }

    @gl.public.view
    def get_dispute_count(self) -> int:
        return int(self.dispute_counter)

    @gl.public.view
    def get_precedent_count(self) -> int:
        return int(self.precedent_counter)
