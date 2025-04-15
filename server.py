from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import json
from datetime import datetime
import os
from threading import Lock
import uvicorn

app = FastAPI(title="문서 평가 시스템 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 제공
app.mount("/static", StaticFiles(directory="."), name="static")

# 동시 접속자 수 제한을 위한 락
active_users_lock = Lock()
active_users = set()
MAX_USERS = 4

# 파일 경로를 절대 경로로 설정
VALIDATION_FILE = os.path.join(os.path.dirname(__file__), 'qlora_finetune_dataset_valid.json')
OPTIMIZATION_FILE = os.path.join(os.path.dirname(__file__), 'direct_preference_optimization.json')
ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), 'account.json')

# 서버 설정
SERVER_IP = '134.185.98.95'
SERVER_PORT = 8000

# Pydantic 모델
class LoginRequest(BaseModel):
    username: str
    password: str

class DisconnectRequest(BaseModel):
    username: str

class EvaluationRequest(BaseModel):
    id: str
    evaluation: int
    username: str

class ModificationRequest(BaseModel):
    id: str
    instruction: str
    input: str
    output: str
    username: str

class ResetEvaluationRequest(BaseModel):
    id: str
    username: str

# 헬퍼 함수
def load_accounts():
    with open(ACCOUNT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def verify_credentials(username: str, password: str) -> bool:
    accounts = load_accounts()
    for user in accounts['users']:
        if user['username'] == username and user['password'] == password:
            return True
    return False

def load_validation_data():
    try:
        with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded data type: {type(data)}")  # 데이터 타입 출력
            print(f"First item type: {type(data[0]) if data else 'empty'}")  # 첫 번째 아이템 타입 출력
            return data
    except Exception as e:
        print(f"Error loading validation data: {str(e)}")
        raise

def save_validation_data(data):
    try:
        with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving validation data: {str(e)}")
        raise

def save_optimization_data(data):
    try:
        if not os.path.exists(OPTIMIZATION_FILE):
            with open(OPTIMIZATION_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
        
        with open(OPTIMIZATION_FILE, 'r+', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
            
            existing_data.append(data)
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            f.truncate()
    except Exception as e:
        print(f"Error saving optimization data: {str(e)}")
        raise

# 라우트 정의
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/{path:path}")
async def serve_static(path: str):
    if os.path.isfile(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/connect")
async def connect(request: LoginRequest):
    try:
        username = request.username
        password = request.password
        
        print(f"Login attempt for user: {username}")  # 로깅 추가
        
        if not username or not password:
            print("Missing credentials")
            raise HTTPException(status_code=400, detail="Username and password are required")
        
        if not verify_credentials(username, password):
            print("Invalid credentials")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        with active_users_lock:
            if len(active_users) >= MAX_USERS:
                print("Max users reached")
                raise HTTPException(status_code=403, detail="Maximum number of users reached")
            active_users.add(username)
        
        print(f"User {username} connected successfully")
        return {"message": "Connected successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error in connect: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/disconnect")
async def disconnect(request: DisconnectRequest):
    username = request.username
    if username:
        with active_users_lock:
            active_users.discard(username)
    return {"message": "Disconnected successfully"}

@app.get("/api/data")
async def get_data():
    try:
        data = load_validation_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

@app.post("/api/evaluate")
async def evaluate(request: EvaluationRequest):
    try:
        print("=== Debug: Starting Evaluation ===")
        
        item_id = request.id
        evaluation = request.evaluation
        username = request.username
        
        print(f"Parsed values - id: {item_id}, evaluation: {evaluation}, username: {username}")
        
        if not all([item_id, evaluation, username]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        try:
            with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
                print(f"Loaded data type: {type(validation_data)}")
                print(f"First item: {validation_data[0] if validation_data else 'Empty'}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to read validation data")
        
        item = None
        for data_item in validation_data:
            if isinstance(data_item, dict) and str(data_item.get('id')) == str(item_id):
                item = data_item
                break
                
        if not item:
            print(f"Item not found for id: {item_id}")
            raise HTTPException(status_code=404, detail="Item not found")
            
        print(f"Found item: {item}")
        
        # metadata 초기화
        if 'metadata' not in item:
            item['metadata'] = {}
            
        # 새로운 평가 데이터 생성
        new_evaluation = {
            'score': evaluation,
            'username': username,
            'timestamp': datetime.now().isoformat(),
            'completed': True  # 평가 완료 상태 추가
        }
        
        # 기존 평가 데이터 업데이트
        item['metadata']['evaluation'] = new_evaluation
        
        print(f"Updated item: {item}")
        
        try:
            with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, ensure_ascii=False, indent=2)
            print("Save successful")
            return {"message": "Evaluation saved successfully"}
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save evaluation")
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/modify")
async def modify(request: ModificationRequest):
    try:
        item_id = request.id
        instruction = request.instruction
        input_text = request.input
        output = request.output
        username = request.username
        
        if not all([item_id, instruction, input_text, output, username]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        optimization_data = {
            'id': item_id,
            'instruction': instruction,
            'input': input_text,
            'output': output,
            'username': username,
            'timestamp': datetime.now().isoformat()
        }
        
        save_optimization_data(optimization_data)
        return {"message": "Modification saved successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Error in modify: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save modification: {str(e)}")

@app.post("/api/reset-evaluation")
async def reset_evaluation(request: ResetEvaluationRequest):
    try:
        item_id = request.id
        username = request.username
        
        if not all([item_id, username]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        try:
            with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to read validation data")
        
        item = None
        for data_item in validation_data:
            if isinstance(data_item, dict) and str(data_item.get('id')) == str(item_id):
                item = data_item
                break
                
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
            
        if 'metadata' not in item:
            item['metadata'] = {}
            
        if 'evaluation' in item['metadata']:
            # 평가 데이터는 유지하되 completed 상태만 변경
            item['metadata']['evaluation']['completed'] = False
        
        try:
            with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, ensure_ascii=False, indent=2)
            return {"message": "Evaluation reset successfully"}
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save evaluation")
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=SERVER_PORT, reload=False) 