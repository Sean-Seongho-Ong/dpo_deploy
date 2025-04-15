import json

try:
    with open('qlora_finetune_dataset_valid.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f'Successfully loaded {len(data)} items')
        
    # Add ID to items if missing
    for idx, item in enumerate(data):
        if 'id' not in item or not item['id']:
            item['id'] = str(idx + 1)
            print(f'Added ID {item["id"]} to item')
    
    # Save the fixed file
    with open('qlora_finetune_dataset_valid.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('File saved successfully')

except Exception as e:
    print(f'Error: {str(e)}') 