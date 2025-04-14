import json

def add_ids_to_json():
    # JSON 파일 읽기
    with open('qlora_finetune_dataset_valid.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 항목에 id 추가
    for i, item in enumerate(data, 1):
        item['id'] = i
    
    # 수정된 데이터 저장
    with open('qlora_finetune_dataset_valid.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    add_ids_to_json()
    print("ID가 성공적으로 추가되었습니다.") 