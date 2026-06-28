from pydantic import BaseModel


class HealthBasicStatus(BaseModel):
    status: str
    role: str


class HealthDatabaseStatus(BaseModel):
    connected: bool
    database_url_scheme: str


class HealthRedisStatus(BaseModel):
    enabled: bool
    connected: bool
    queue_backlog: int


class HealthReadinessStatus(BaseModel):
    status: str
    role: str
    database: HealthDatabaseStatus
    redis: HealthRedisStatus
