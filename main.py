from typing import Annotated, Sequence, Optional, Tuple

from fastapi import FastAPI, Depends, HTTPException, Query
from sqlmodel import SQLModel, Field, Session, select, create_engine, Column, DateTime
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from dateutil import parser

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    
def get_session():
    with Session(engine) as session:
        yield session
        
SessionDep = Annotated[Session, Depends(get_session)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: Optional[str] = Field(default="Guest Anonymous")
    operation: Optional[str] = Field(default="Trimming hair length shorter", index=True)
    created_date: str = Field(default=datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    expected_duration: int = Field(default=30)
    appointment_datetime: str
    branch: int
    
@app.get("/order")
def read_orders(session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=100)] = 100) -> Sequence[Appointment]:
    orders = session.exec(select(Appointment).offset(offset).limit(limit)).all()
    return orders
    
@app.post("/order")
def make_order(order: Appointment, session: SessionDep) -> Appointment:
    session.add(order)
    session.commit()
    session.refresh(order)
    return order

@app.get("/check_availability")
def check_appointment_availability(session: SessionDep, dt: str, duration: int):
    response = check_available_time(session, dt, duration)
    if response[0]:
        return { "status": 200, "message": f"Yes, you can order at {dt}. Your operation is expected to take {duration} minutes." }
    else:
        return { "status": 200, "message": f"Sorry, there is a conflict of appointment at {dt}. However we suggest you to order at {response[2]}." }

def check_available_time(db_session: Session, requested_datetime: str, duration: int) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    
    Check if the salon is available for a new appointment.
    
    Args:
        db_session: Database session
        requested_datetime: ISO format datetime string
        duration: Expected duration in minutes
    
    Returns:
        Tuple containing:
        - is_available (bool): Whether the time slot is available
        - conflict_message (str): Message describing any conflicts
        - suggested_time (str): Suggested alternative time if there's a conflict
    
    """
    
    requested_dt = parser.parse(requested_datetime)
    requested_end_dt = requested_dt + timedelta(minutes=duration)
    
    # Potentially conflicting appointment spanning 2 hours either direction
    statement = select(Appointment).where(
        Appointment.appointment_datetime.between(
            (requested_dt - timedelta(hours=2)).isoformat(),
            (requested_dt + timedelta(hours=2)).isoformat()
        )
    )
    appointments = db_session.exec(statement).all()
    
    # Check if there is an overlapping appointment
    for appointment in appointments:
        appointment_dt = parser.parse(appointment.appointment_datetime)
        appointment_end_dt = appointment_dt + timedelta(minutes=appointment.expected_duration)
        
        if ((requested_dt <= appointment_end_dt and requested_end_dt >= appointment_dt) 
            or 
            (appointment_dt <= requested_end_dt and appointment_end_dt >= requested_dt)):
            
            # There is an overlap
            if requested_dt < appointment_dt:
                # Requested time starts before existing appointment
                return (False, f"Conflict arose with {appointment.appointment_datetime}", appointment_end_dt.isoformat())
            
            else:
                # Requested time starts during or after existing appointment
                return (False, f"Conflict arose with {appointment.appointment_datetime}", appointment_end_dt.isoformat())

        # Check if another appointment starts during the requested operation
        if appointment_dt > requested_dt and appointment_dt < requested_end_dt:
            return (False, f"Another appointment starts during {appointment.appointment_datetime}", appointment_end_dt.isoformat())
    
    return (True, None, None)