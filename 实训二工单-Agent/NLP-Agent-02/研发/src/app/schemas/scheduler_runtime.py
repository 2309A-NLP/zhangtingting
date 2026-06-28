from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SchedulerLeaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    owner_id: str
    locked_until: datetime
    created_at: datetime
    updated_at: datetime
