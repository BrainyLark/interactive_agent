from typing import Annotated, Sequence, Optional, Tuple, Literal, TypedDict
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlmodel import SQLModel, Field, Session, select, create_engine, DateTime
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dateutil import parser

import logging
from dotenv import load_dotenv

_ = load_dotenv()

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
    
@app.get("/appointment")
def retrieve(session: SessionDep, offset: int = 0, limit: Annotated[int, Query(le=100)] = 100) -> Sequence[Appointment]:
    orders = session.exec(select(Appointment).offset(offset).limit(limit)).all()
    return orders
    
@app.post("/appointment")
def insert(order: Appointment, session: SessionDep) -> Appointment:
    session.add(order)
    session.commit()
    session.refresh(order)
    return order

@app.get("/check_conflict")
def check_conflict(session: SessionDep, dt: str, duration: int):
    response = check_datetime(session, dt, duration)
    if response[0]:
        return { "message": f"Yes, you can order at {dt}. Your operation is expected to take {duration} minutes." }
    else:
        return { "message": f"Sorry, there is a conflict of appointment at {dt}. However we suggest you to order at {response[2]}." }

def check_datetime(db_session: Session, requested_datetime: str, duration: int) -> Tuple[bool, Optional[str], Optional[str]]:
    
    check_time_window = 2
    
    requested_dt = parser.parse(requested_datetime)
    requested_end_dt = requested_dt + timedelta(minutes=duration)
    
    # Potentially conflicting appointment spanning 2 hours either direction
    
    statement = select(Appointment).where(
        Appointment.appointment_datetime.between(
            (requested_dt - timedelta(hours=check_time_window)).isoformat(),
            (requested_dt + timedelta(hours=check_time_window)).isoformat()
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
                return (False, f"There is an appointment at {appointment.appointment_datetime}, right after your requested time.", appointment_end_dt.isoformat())
            
            else:
                # Requested time starts during or after existing appointment
                return (False, f"We got an appointment starting at {appointment.appointment_datetime} and through your requested time.", appointment_end_dt.isoformat())

        # Check if another appointment starts during the requested operation
        if appointment_dt > requested_dt and appointment_dt < requested_end_dt:
            return (False, f"Another appointment starts during {appointment.appointment_datetime}", appointment_end_dt.isoformat())
    
    return (True, None, None)