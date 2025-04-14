from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
from datetime import datetime
import os
from threading import Lock

app = Flask(__name__, static_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

# 동시 접속자 수 제한을 위한 락
active_users_lock = Lock()
active_users = set()
MAX_USERS = 4

# 파일 경로를 절대 경로로 설정
VALIDATION_FILE = os.path.join(os.path.dirname(__file__), 'qlora_finetune_dataset_valid.json')
OPTIMIZATION_FILE = os.path.join(os.path.dirname(__file__), 'direct_preference_optimization.json')
ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), 'account.json')

PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('FLASK_ENV') == 'development'

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

def load_accounts():
    with open(ACCOUNT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def verify_credentials(username, password):
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

@app.route('/api/connect', methods=['POST'])
def connect():
    try:
        username = request.json.get('username')
        password = request.json.get('password')
        
        print(f"Login attempt for user: {username}")  # 로깅 추가
        
        if not username or not password:
            print("Missing credentials")
            return jsonify({'error': 'Username and password are required'}), 400
        
        if not verify_credentials(username, password):
            print("Invalid credentials")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        with active_users_lock:
            if len(active_users) >= MAX_USERS:
                print("Max users reached")
                return jsonify({'error': 'Maximum number of users reached'}), 403
            active_users.add(username)
        
        print(f"User {username} connected successfully")
        return jsonify({'message': 'Connected successfully'})
    except Exception as e:
        print(f"Error in connect: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    username = request.json.get('username')
    if username:
        with active_users_lock:
            active_users.discard(username)
    return jsonify({'message': 'Disconnected successfully'})

@app.route('/api/data', methods=['GET'])
def get_data():
    data = load_validation_data()
    return jsonify(data)

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        print("=== Debug: Starting Evaluation ===")
        data = request.json
        print(f"Raw request data: {data}")
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        item_id = data.get('id')
        evaluation = data.get('evaluation')
        username = data.get('username')
        
        print(f"Parsed values - id: {item_id}, evaluation: {evaluation}, username: {username}")
        
        if not all([item_id, evaluation, username]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        try:
            with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
                print(f"Loaded data type: {type(validation_data)}")
                print(f"First item: {validation_data[0] if validation_data else 'Empty'}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return jsonify({'error': 'Failed to read validation data'}), 500
        
        item = None
        for data_item in validation_data:
            if isinstance(data_item, dict) and str(data_item.get('id')) == str(item_id):
                item = data_item
                break
                
        if not item:
            print(f"Item not found for id: {item_id}")
            return jsonify({'error': 'Item not found'}), 404
            
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
            return jsonify({'message': 'Evaluation saved successfully'})
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return jsonify({'error': 'Failed to save evaluation'}), 500
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/api/modify', methods=['POST'])
def modify():
    try:
        data = request.json
        item_id = data.get('id')
        instruction = data.get('instruction')
        input_text = data.get('input')
        output = data.get('output')
        username = data.get('username')
        
        if not all([item_id, instruction, input_text, output, username]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        optimization_data = {
            'id': item_id,
            'instruction': instruction,
            'input': input_text,
            'output': output,
            'username': username,
            'timestamp': datetime.now().isoformat()
        }
        
        save_optimization_data(optimization_data)
        return jsonify({'message': 'Modification saved successfully'})
    except Exception as e:
        print(f"Error in modify: {str(e)}")
        return jsonify({'error': f'Failed to save modification: {str(e)}'}), 500

@app.route('/api/reset-evaluation', methods=['POST'])
def reset_evaluation():
    try:
        data = request.json
        item_id = data.get('id')
        username = data.get('username')
        
        if not all([item_id, username]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        try:
            with open(VALIDATION_FILE, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return jsonify({'error': 'Failed to read validation data'}), 500
        
        item = None
        for data_item in validation_data:
            if isinstance(data_item, dict) and str(data_item.get('id')) == str(item_id):
                item = data_item
                break
                
        if not item:
            return jsonify({'error': 'Item not found'}), 404
            
        if 'metadata' not in item:
            item['metadata'] = {}
            
        if 'evaluation' in item['metadata']:
            # 평가 데이터는 유지하되 completed 상태만 변경
            item['metadata']['evaluation']['completed'] = False
        
        try:
            with open(VALIDATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(validation_data, f, ensure_ascii=False, indent=2)
            return jsonify({'message': 'Evaluation reset successfully'})
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return jsonify({'error': 'Failed to save evaluation'}), 500
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG) 