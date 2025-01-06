import requests
import logging
from datetime import datetime

def main():
    logging.basicConfig(filename="insert.log", level=logging.INFO)
    
    appointments = [
        {"appointment_datetime": datetime(2025, 1, 7, 10, 0, 0).isoformat(), "expected_duration": 90, "branch": 1, "operation": "Trimming hair branches"}, 
        {"appointment_datetime": datetime(2025, 1, 7, 11, 30, 0).isoformat(), "expected_duration": 60, "branch": 1, "operation": "Trimming hair branches"}, 
        {"appointment_datetime": datetime(2025, 1, 7, 13, 30, 0).isoformat(), "expected_duration": 30, "branch": 1, "operation": "Trimming hair branches"}, 
        {"appointment_datetime": datetime(2025, 1, 7, 15, 0, 0).isoformat(), "expected_duration": 60, "branch": 1, "operation": "Trimming hair branches"}
    ]
    
    urlstring = "http://localhost:8000/order"
    
    logger = logging.getLogger(__name__)
    
    response = []
    for appointment in appointments:
        data = requests.post(urlstring, json=appointment)
        response.append(data.content)
        
    logger.info(response)

if __name__ == "__main__":
    main()