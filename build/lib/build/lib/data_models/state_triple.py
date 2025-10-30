from typing import TypedDict


class StateTriple(TypedDict):
    initial_state: str
    transformation: str
    final_state: str

