from fastapi import FastAPI, HTTPException, Request, Response, status, BackgroundTasks, Body
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
import time
import socket

# 환경 설정
APP_MODE = os.environ.get('APP_MODE', 'dev')  # 기본값은 'dev'

# 모드에 따른 서버 설정
if APP_MODE == 'dev':
    SERVER_IP = 'localhost'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
    print(f"개발 모드에서 실행 중: {SERVER_IP}")
else:  # 'pro'
    SERVER_IP = '134.185.98.95'
    BASE_DIR = '/home/ubuntu/Stage1_DPO'  # 서버 경로
    print(f"프로덕션 모드에서 실행 중: {SERVER_IP}")

SERVER_PORT = 8000

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
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# 동시 접속자 수 제한을 위한 락
active_users_lock = Lock()
active_users = set()
MAX_USERS = 4

# 파일 경로를 절대 경로로 설정
VALIDATION_FILE = os.path.join(BASE_DIR, 'qlora_finetune_dataset_valid.json')
OPTIMIZATION_FILE = os.path.join(BASE_DIR, 'direct_preference_optimization.json')
ACCOUNT_FILE = os.path.join(BASE_DIR, 'account.json')

print(f"모드: {APP_MODE}")
print(f"기본 디렉토리: {BASE_DIR}")
print(f"데이터 파일: {VALIDATION_FILE}")
print(f"계정 파일: {ACCOUNT_FILE}")

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

# 캐시 관련 변수
data_cache = None
cache_last_modified = 0
cache_lock = Lock()
is_saving = False
save_needed = False
last_save_time = 0
SAVE_INTERVAL = 60  # 최소 저장 간격 (초)

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

def load_validation_data(force_reload=False):
    global data_cache, cache_last_modified
    
    # 캐시된 데이터가 있고 강제 리로드가 아니면 캐시 반환
    with cache_lock:
        if data_cache is not None and not force_reload:
            print("반환된 캐시된 데이터:", len(data_cache), "항목")
            return data_cache
            
        try:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Attempting to load validation data from: {VALIDATION_FILE}")
            print(f"File exists: {os.path.exists(VALIDATION_FILE)}")
            print(f"File size: {os.path.getsize(VALIDATION_FILE) if os.path.exists(VALIDATION_FILE) else 'File not found'}")
            
            if not os.path.exists(VALIDATION_FILE):
                print(f"Creating empty validation data file")
                os.makedirs(os.path.dirname(VALIDATION_FILE), exist_ok=True)
                with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                return []
            
            with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
                # 먼저 파일 내용 읽기
                file_content = f.read()
                
                # 파일이 비어있는지 확인
                if not file_content.strip():
                    print("File is empty, returning empty list")
                    return []
                
                # JSON 파싱 시도
                try:
                    data = json.loads(file_content)
                    print(f"Successfully loaded data, type: {type(data)}")
                    print(f"Data length: {len(data)}")
                    print(f"First item type: {type(data[0]) if data else 'empty'}")
                    if data and len(data) > 0:
                        print(f"First item keys: {list(data[0].keys())}")
                    # 파일에서 데이터 로드 후 캐시에 저장
                    data_cache = data
                    cache_last_modified = os.path.getctime(VALIDATION_FILE)
                    return data_cache
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error: {str(json_err)}")
                    print(f"First 100 chars of content: {file_content[:100]}")
                    raise
        except Exception as e:
            print(f"Error loading validation data: {str(e)}")
            # 비상 대책: 빈 배열 반환
            print("Returning empty list as fallback")
            data_cache = []
            return data_cache

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
# API 엔드포인트들
@app.get("/api/data")
async def get_data():
    try:
        print("API 데이터 요청 받음")
        data = load_validation_data()
        
        # ID 필드가 없는 항목에 ID 추가
        for idx, item in enumerate(data):
            if 'id' not in item or not item['id']:
                item['id'] = str(idx + 1)
        
        print(f"데이터 {len(data)}개 반환")
        return data
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

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
        
        # 캐시된 데이터 사용 (읽기 전용)
        validation_data = load_validation_data()
        
        # 먼저 캐시에서 업데이트
        item = None
        for data_item in validation_data:
            if str(data_item.get('id')) == str(item_id):
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
        
        # 캐시 업데이트 - 이제 캐시된 데이터는 최신 상태
        global data_cache
        data_cache = validation_data
        
        # 파일 저장 - 캐시된 데이터를 파일에 즉시 저장
        save_validation_data(validation_data)
        
        return {"message": "Evaluation saved successfully"}
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

# 일반 파일 제공 라우트
@app.get("/")
async def serve_index():
    index_file = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Index file not found")

@app.get("/{path:path}")
async def serve_static(path: str):
    # API 경로는 이미 위에서 처리됨
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"API endpoint not found: {path}")
        
    file_path = os.path.join(BASE_DIR, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail=f"File {path} not found")

# 백그라운드 저장 함수 수정
async def save_data_background(background_tasks: BackgroundTasks, force: bool = False):
    global is_saving, save_needed, last_save_time
    
    current_time = time.time()
    if not force and is_saving:
        save_needed = True
        return {"message": "Save already in progress, will save after current save completes"}
    
    if not force and current_time - last_save_time < SAVE_INTERVAL:
        save_needed = True
        return {"message": "Save requested too soon, will save after interval"}
    
    background_tasks.add_task(save_data_to_file, force)
    return {"message": "Save started in background"}

async def save_data_to_file(force: bool = False):
    global is_saving, save_needed, last_save_time, data_cache
    
    if is_saving and not force:
        save_needed = True
        return
    
    is_saving = True
    try:
        print("데이터 파일에 저장 중...")
        
        if data_cache is None:
            print("No data in cache to save")
            is_saving = False
            return
        
        # 데이터를 JSON 파일에 저장
        with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_cache, f, ensure_ascii=False, indent=2)
        
        print(f"데이터 성공적으로 저장됨 ({len(data_cache)} 항목)")
        last_save_time = time.time()
        save_needed = False
    except Exception as e:
        print(f"저장 중 오류 발생: {str(e)}")
        save_needed = True
    finally:
        is_saving = False
        
    # 저장이 필요하다고 마킹되었으면 다시 저장 시도
    if save_needed and force:
        await save_data_to_file(force=True)

# 새로운 API 엔드포인트 수정
@app.post("/api/save")
async def save_data(background_tasks: BackgroundTasks, request: dict = Body(default={})):
    """데이터를 파일에 저장합니다."""
    force = request.get('force', False)
    return await save_data_background(background_tasks, force=force)

# 서버 시작/종료 이벤트 핸들러 추가
@app.on_event("startup")
async def startup_event():
    print("서버 시작 시 데이터 사전 로드...")
    load_validation_data(force_reload=True)

@app.on_event("shutdown")
async def shutdown_event():
    global data_cache
    print("서버 종료 시 데이터 저장...")
    
    if data_cache is not None:
        try:
            with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(data_cache, f, ensure_ascii=False, indent=2)
            print(f"서버 종료 시 데이터 성공적으로 저장됨 ({len(data_cache)} 항목)")
        except Exception as e:
            print(f"서버 종료 시 데이터 저장 중 오류 발생: {str(e)}")

if __name__ == '__main__':
    print(f"Starting server at http://{SERVER_IP}:{SERVER_PORT}")
    print(f"Documentation available at http://{SERVER_IP}:{SERVER_PORT}/docs")
    print(f"Base directory: {BASE_DIR}")
    uvicorn.run("server:app", host="0.0.0.0", port=SERVER_PORT, reload=(APP_MODE == 'dev')) 