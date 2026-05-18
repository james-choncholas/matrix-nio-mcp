from dataclasses import dataclass, asdict


@dataclass
class MessageRecord:
    event_id: str
    room_id: str
    sender: str
    sender_name: str
    body: str
    timestamp: int  # Unix milliseconds

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    event_id: str
    room_id: str
    sender: str
    sender_name: str
    body: str
    timestamp: int
    score: float

    def to_dict(self) -> dict:
        return asdict(self)
